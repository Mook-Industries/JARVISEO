"""팀 웹 컨테이너(Dockerfile.web)의 메모리 · 응답 시간을 잰다.  담당: 문태현 (배포)

클라우드 서버 크기(#179)를 정하려고 쓴다. 이미지를 docker 로 띄우고 ``docker stats`` 를 계속 읽으며
켜진 직후(대기) · 질문(/api/ask) · 성분 확인(/api/ingredient) 때 메모리 최고치와 응답 시간을 찍는다.
**OpenAI(VLM · 임베딩)와 Google Vision 을 실제로 부른다.** 질문 · 성분 확인 횟수만큼 요금이 나간다.

    docker build -f Dockerfile.web -t jarviseo-web .
    python scripts/measure_web_memory.py --photo <사진.jpg> --label <성분표 사진.jpg>

키는 --env-file(기본 .env)로 컨테이너에 넘긴다. 끝나면 컨테이너를 지운다(--keep 이면 남김).
"""

import argparse
import base64
import json
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

NAME = "jarviseo-web-measure"
UNITS = {"B": 1 / 2**20, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_mib(text: str) -> float | None:
    """``docker stats`` 의 '812.3MiB / 3.697GiB' 에서 앞쪽을 MiB 로."""
    m = re.match(r"\s*([\d.]+)\s*([KMG]?i?B)", text)
    return float(m.group(1)) * UNITS[m.group(2)] if m and m.group(2) in UNITS else None


class MemoryWatch:
    """``docker stats`` 를 띄워 두고 줄마다 메모리를 모은다. phase 로 구간을 나눠 최고치를 본다."""

    def __init__(self, name: str) -> None:
        self.phase = "켜는 중"
        self.samples: list[tuple[str, float]] = []
        self.proc = subprocess.Popen(
            ["docker", "stats", "--format", "{{.MemUsage}}", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            mib = parse_mib(ANSI.sub("", line))  # 줄마다 화면 제어 코드(ESC[H · ESC[K)가 붙는다
            if mib is not None:
                self.samples.append((self.phase, mib))

    def peak(self, phase: str) -> float:
        values = [m for p, m in self.samples if p == phase]
        return max(values) if values else float("nan")

    def last(self) -> float:
        return self.samples[-1][1] if self.samples else float("nan")

    def stop(self) -> None:
        self.proc.terminate()


def post(url: str, body: bytes, content_type: str, timeout: float = 120.0) -> tuple[float, dict]:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": content_type})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return time.perf_counter() - started, data


def wait_health(base: str, timeout: float = 120.0) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            urllib.request.urlopen(f"{base}/health", timeout=2).read()
            return time.perf_counter() - started
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.5)
    raise SystemExit("서버가 뜨지 않았다. docker logs " + NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="팀 웹 컨테이너 메모리 · 응답 시간")
    parser.add_argument("--image", default="jarviseo-web")
    parser.add_argument("--env-file", default=".env", help="OpenAI · Vision 키가 든 파일")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--photo", type=Path, required=True, help="질문에 붙일 사진")
    parser.add_argument("--label", type=Path, required=True, help="성분 확인에 보낼 성분표 사진")
    parser.add_argument("--question", default="이거 뭐야?")
    parser.add_argument("--repeat", type=int, default=3, help="질문 · 성분 확인을 몇 번씩")
    parser.add_argument("--idle-sec", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=120.0, help="요청 하나를 기다릴 최대 초")
    parser.add_argument("--keep", action="store_true", help="끝나도 컨테이너를 남긴다")
    parser.add_argument(
        "-e", "--env", action="append", default=[], metavar="KEY=VALUE",
        help="--env-file 위에 덮어쓸 값. -e GOOGLE_VISION_API_KEY= 면 EasyOCR 로 잰다",
    )  # fmt: skip
    args = parser.parse_args()

    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--name", NAME, "-p", f"{args.port}:8000",
         "--env-file", args.env_file, *[f"-e{e}" for e in args.env], args.image],
        check=True, capture_output=True,
    )  # fmt: skip
    base = f"http://localhost:{args.port}"
    photo = "data:image/jpeg;base64," + base64.b64encode(args.photo.read_bytes()).decode()
    ask = json.dumps({"question": args.question, "image": photo}).encode()
    requests = [
        ("질문", f"{base}/api/ask", ask, "application/json"),
        ("성분", f"{base}/api/ingredient?allergens={quote('모두')}", args.label.read_bytes(),
         "image/jpeg"),  # '모두' = 법정 알레르기 전체
    ]  # fmt: skip
    results: dict[str, list[tuple[float, str]]] = {"질문": [], "성분": []}
    watch = MemoryWatch(NAME)
    try:
        startup = wait_health(base)
        watch.phase = "대기"
        time.sleep(args.idle_sec)
        for kind, url, body, content_type in requests:
            for i in range(args.repeat):
                watch.phase = f"{kind}{i + 1}"
                started = time.perf_counter()
                try:
                    _, data = post(url, body, content_type, args.timeout)
                    text = str(data.get("answer") or data.get("verdict") or data)[:60]
                except Exception as e:  # noqa: BLE001 — 하나가 실패해도 메모리는 끝까지 잰다
                    text = f"실패: {type(e).__name__} {e}"[:60]
                results[kind].append((time.perf_counter() - started, text))
        watch.phase = "끝"
        time.sleep(3)
    finally:
        watch.stop()
        if not args.keep:
            subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)

    peaks = {p: round(watch.peak(p)) for p in dict.fromkeys(p for p, _ in watch.samples)}
    print(f"이미지 {args.image}, 켜기 {startup:.1f}초, 덮어쓴 값 {args.env or '없음'}")
    print(f"메모리 최고(MiB) 구간별: {peaks}")
    for kind, rows in results.items():
        times = [t for t, _ in rows]
        print(f"{kind} 응답(초) " + " ".join(f"{t:.1f}" for t in times)
              + f" · 중앙 {statistics.median(times):.1f} · 예: {rows[0][1]}")  # fmt: skip


if __name__ == "__main__":
    main()
