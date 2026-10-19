"""Colab GPU 에서 성분표 OCR 서버를 띄운다.  (nutrition/ocr_remote.py 가 부른다)

노트북에서 assistant.py 를 돌리고, 성분표 OCR 만 Colab T4 GPU 에 맡기기 위한 서버다.
PaddleOCR(PP-OCRv5 server_det + korean_PP-OCRv5_mobile_rec)을 띄우고,
밖에서 닿는 https 주소를 만든다.

주소 만드는 방법은 두 가지다.
- ngrok 고정 주소 (권장): NGROK_AUTHTOKEN · NGROK_DOMAIN 이 있으면 쓴다. 매번 같은 주소라
  OCR_TOKEN 도 정해 두면 노트북 .env 를 처음 한 번만 채우면 된다.
  ngrok.com 무료 가입 → Your Authtoken 복사, Domains 에서 무료 고정 도메인 하나 받기.
- cloudflared 빠른 터널: 위 값이 없으면 쓴다. 가입이 필요 없지만 켤 때마다 주소가 바뀐다.

Colab 에서 (런타임 → 런타임 유형 변경 → T4 GPU):

    # 1) 설치. torch 는 paddle 과 NCCL 이 충돌해서 지운다(이 서버는 torch 를 쓰지 않는다).
    #    지운 뒤에도 "undefined symbol: ncclCommWindowDeregister" 가 나면 폴더째 지운다.
    !pip uninstall -y torch torchvision torchaudio
    !rm -rf /usr/local/lib/python3*/dist-packages/torch
    !pip install -q paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    !pip install -q "paddleocr>=3.3" opencv-python-headless
    # 2) 고정 주소를 쓸 때: 왼쪽 열쇠 아이콘(보안 비밀)에 OCR_TOKEN · NGROK_AUTHTOKEN ·
    #    NGROK_DOMAIN 을 저장하고 "노트북 액세스"를 켠 뒤 환경 변수로 옮긴다.
    from google.colab import userdata
    import os
    for k in ("OCR_TOKEN", "NGROK_AUTHTOKEN", "NGROK_DOMAIN"):
        os.environ[k] = userdata.get(k)
    # 3) 이 파일을 올리거나 저장소에서 받아 실행한다.
    !python colab_ocr_server.py

출력되는 두 줄을 노트북 .env 에 적는다 (고정 주소면 처음 한 번만).

    JARVISEO_OCR_REMOTE_URL=https://내도메인.ngrok-free.app
    JARVISEO_OCR_REMOTE_TOKEN=...

요청: POST /ocr, 본문은 JPEG/PNG 바이트, 헤더 X-Token.
응답: {"lines": [[[[x,y]×4], "글자", 신뢰도], ...], "ms": 걸린 시간}
GET /health 는 토큰 없이 {"ok": true, "device": "gpu:0"} 를 준다.

Colab 은 셀이 멈추거나 세션이 끊기면 서버도 꺼진다. 다시 띄우면 cloudflared 는 주소가 바뀌고,
ngrok 고정 주소는 그대로다.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

PORT = int(os.environ.get("OCR_PORT", "8765"))
TOKEN = os.environ.get("OCR_TOKEN") or secrets.token_urlsafe(16)
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN", "").strip()
NGROK_DOMAIN = re.sub(r"^https?://|/+$", "", os.environ.get("NGROK_DOMAIN", "").strip())
NGROK = Path("/usr/local/bin/ngrok")
NGROK_URL = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
CLOUDFLARED = Path("/usr/local/bin/cloudflared")
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
)
MAX_BYTES = 15 * 1024 * 1024  # 성분표 사진 한 장으로 충분한 크기


def load_ocr():
    import paddle
    from paddleocr import PaddleOCR

    device = "gpu:0" if paddle.device.cuda.device_count() else "cpu"
    ocr = PaddleOCR(
        lang="korean",
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device,
    )
    warm = np.full((320, 640, 3), 255, np.uint8)
    cv2.putText(warm, "warm up", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
    for _ in range(2):  # GPU 첫 추론은 느리다. 미리 한 번 돌려 둔다
        list(ocr.predict(warm))
    return ocr, device


def readtext(ocr, image: np.ndarray) -> list:
    """easyocr.Reader.readtext(detail=1) 와 같은 모양: [(꼭짓점 4개, 글자, 신뢰도), ...]"""
    out = []
    for res in ocr.predict(image):
        r = dict(res)
        for pts, text, score in zip(r["rec_polys"], r["rec_texts"], r["rec_scores"], strict=False):
            out.append([np.asarray(pts).tolist(), str(text), float(score)])
    return out


def make_handler(ocr, device: str):
    lock = threading.Lock()  # 모델 하나를 여러 요청이 동시에 쓰지 않게 한다

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 (http.server 규칙)
            if self.path == "/health":
                self._send(200, {"ok": True, "device": device})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/ocr":
                return self._send(404, {"error": "not found"})
            if not secrets.compare_digest(self.headers.get("X-Token", ""), TOKEN):
                return self._send(401, {"error": "bad token"})
            size = int(self.headers.get("Content-Length") or 0)
            if not 0 < size <= MAX_BYTES:
                return self._send(413, {"error": "bad size"})
            buf = np.frombuffer(self.rfile.read(size), np.uint8)
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if image is None:
                return self._send(400, {"error": "not an image"})
            t0 = time.perf_counter()
            with lock:
                lines = readtext(ocr, image)
            ms = int((time.perf_counter() - t0) * 1000)
            self._send(200, {"lines": lines, "ms": ms})

        def log_message(self, fmt: str, *args) -> None:
            print(f"[ocr] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def _drain(proc: subprocess.Popen) -> None:
    # 나머지 로그는 버린다(안 읽으면 파이프가 차서 터널이 멈춘다)
    threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True).start()


def start_ngrok() -> str:
    """ngrok 고정 도메인으로 터널을 띄운다. 매번 같은 주소다."""
    if not NGROK.exists():
        tgz, _ = urllib.request.urlretrieve(NGROK_URL)
        with tarfile.open(tgz) as tar:
            tar.extract("ngrok", NGROK.parent, filter="data")
        NGROK.chmod(0o755)
    proc = subprocess.Popen(
        [str(NGROK), "http", str(PORT), "--domain", NGROK_DOMAIN,
         "--authtoken", NGROK_AUTHTOKEN, "--log", "stdout", "--log-format", "logfmt"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )  # fmt: skip
    for line in proc.stdout:
        if "started tunnel" in line:
            _drain(proc)
            return f"https://{NGROK_DOMAIN}"
        if "lvl=eror" in line or "ERR_NGROK" in line:
            raise RuntimeError(f"ngrok 터널을 못 띄웠다: {line.strip()}")
    raise RuntimeError("ngrok 이 주소 없이 끝났다 (인증키·도메인을 확인하세요)")


def start_tunnel() -> str:
    """ngrok 값이 있으면 고정 주소, 없으면 cloudflared 빠른 터널. https 주소를 돌려준다."""
    if NGROK_AUTHTOKEN and NGROK_DOMAIN:
        return start_ngrok()
    if not CLOUDFLARED.exists():
        urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED)
        CLOUDFLARED.chmod(0o755)
    proc = subprocess.Popen(
        [str(CLOUDFLARED), "tunnel", "--no-autoupdate", "--url", f"http://localhost:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:  # 주소는 시작 로그에 한 번 찍힌다
        m = re.search(r"https://[\w-]+\.trycloudflare\.com", line)
        if m:
            _drain(proc)
            return m.group(0)
    raise RuntimeError("cloudflared 주소를 못 받았다")


def main() -> None:
    print("PaddleOCR 모델을 올리는 중...", flush=True)
    ocr, device = load_ocr()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), make_handler(ocr, device))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = start_tunnel()
    fixed = bool(NGROK_AUTHTOKEN and NGROK_DOMAIN and os.environ.get("OCR_TOKEN"))
    when = "처음 한 번만 (다음에도 같은 값)" if fixed else "켤 때마다 (주소·토큰이 바뀜)"
    print(f"\n장치: {device}. 아래 두 줄을 노트북 .env 에 적으세요 — {when}.\n", flush=True)
    print(f"JARVISEO_OCR_REMOTE_URL={url}")
    print(f"JARVISEO_OCR_REMOTE_TOKEN={TOKEN}\n", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
