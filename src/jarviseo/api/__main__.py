"""대시보드를 이 PC 에서 켠다.

    python -m jarviseo.api            # http://localhost:8000 이 브라우저로 열린다
    python -m jarviseo.api --port 8100 --no-browser

서버는 각자 PC 에서만 돈다(localhost). 카메라 · 마이크는 브라우저가 켜고,
성분표 글자 읽기만 .env 의 JARVISEO_OCR_REMOTE_URL(같이 쓰는 Colab)로 보낸다.
Colab 이 꺼져 있으면 이 PC 의 CPU OCR 로 대신 읽는다(느리지만 멈추지 않는다).
"""

from __future__ import annotations

import argparse
import threading
import webbrowser

HOST = "127.0.0.1"  # 다른 PC 에서는 못 들어오게 이 PC 에만 연다


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m jarviseo.api", description="자비서 대시보드")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않는다")
    p.add_argument("--reload", action="store_true", help="코드를 고치면 서버를 다시 띄운다(개발용)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import uvicorn  # 실행할 때만 필요하다

    args = parse_args(argv)
    url = f"http://localhost:{args.port}"
    if not args.no_browser:
        # OCR 모델을 올리는 동안 페이지가 기다리다 열린다
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    print(f"자비서 대시보드: {url}  (끄려면 Ctrl+C)")
    uvicorn.run("jarviseo.api.main:app", host=HOST, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
