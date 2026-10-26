"""마이크 없는 서버(클라우드)에서 대시보드 질문이 멈추지 않는지 재현한다 (#179).  담당: 문태현

클라우드 대시보드는 페이지를 열 때마다 POST /api/voice/start 로 팀 음성 모드를 켜 보고,
마이크가 없어 실패한다. 그 뒤 POST /api/ask 를 보내 대답이 오는지, 몇 초 걸리는지 잰다.
마이크는 서버처럼 열기에서 실패하게 바꾸고(Microphone), 나머지는 실제 부품(검출기 · 손 모델 ·
VLM · SQLite 기록)을 그대로 쓴다. OpenAI 키(.env)가 있어야 VLM 까지 돈다.

10/26 결과: 고치기 전에는 voice/start 가 손 모델을 만든 뒤 마이크에서 실패해 그 모델이 닫히지 않고
남았고, 첫 질문이 torch 를 불러오는 도중 그 모델의 정리(__del__)가 끼어들어 "대상 검출"에서 영영
멈췄다(timeout 뒤 스택이 찍힌다). 고친 뒤에는 voice/start 0.0s, 첫 질문 19.8s, 두 번째 0.5s.

실행 (레포 맨 위 폴더에서):
    python scripts/check_web_ask_no_mic.py 사진.jpg
    python scripts/check_web_ask_no_mic.py 사진.jpg --starts 3 --timeout 240
"""

from __future__ import annotations

import argparse
import base64
import faulthandler
import os
import sys
import tempfile
import time
from pathlib import Path


class NoMic:
    """서버(오디오 장치 없음)처럼 마이크 열기에서 실패한다."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        raise OSError("PortAudio: 기본 입력 장치 없음 (재현용)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", type=Path, help="질문에 붙일 사진")
    ap.add_argument("--starts", type=int, default=2, help="질문 전에 voice/start 를 부르는 횟수")
    ap.add_argument("--timeout", type=int, default=240, help="이 초가 지나면 스택을 찍고 끝낸다")
    args = ap.parse_args()

    db = Path(tempfile.mkdtemp()) / "check_web_ask.db"
    os.environ["JARVISEO_API_DB"] = f"sqlite:///{db.as_posix()}"
    import cv2

    import jarviseo.voice.mic as mic_mod

    mic_mod.Microphone = NoMic
    from fastapi.testclient import TestClient

    from jarviseo.api.main import create_app

    img = cv2.resize(cv2.imread(str(args.image)), (1920, 1080))
    url = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()
    faulthandler.dump_traceback_later(args.timeout, exit=True, file=sys.stderr)
    with TestClient(create_app()) as c:
        for i in range(args.starts):
            t = time.perf_counter()
            r = c.post("/api/voice/start").json()
            print(f"voice/start {i + 1}: {time.perf_counter() - t:.1f}s {r}", flush=True)
        for n in (1, 2):
            t = time.perf_counter()
            more = {"more": [url, url], "more_ms": [150, 300]}
            body = {"question": "이거 뭐야?", "image": url, **more}
            r = c.post("/api/ask", json=body)
            d = r.json()
            print(
                f"ask {n}: {r.status_code} {time.perf_counter() - t:.1f}s "
                f"{d.get('answer')} {d.get('latency_ms')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
