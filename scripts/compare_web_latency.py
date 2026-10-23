"""팀 웹의 질문 · 성분 확인 응답 시간을 로컬 · 클라우드에서 같은 사진으로 잰다.  담당: 문태현 (배포)

#179 합격 기준 "같은 질문을 로컬과 클라우드에서 5번씩 해 응답 시간 중앙값을 비교한다" 용이다.
**OpenAI · Google Vision 을 실제로 부른다.** (주소 수 × 횟수 × 2)번만큼 요금이 나간다.

    python -m jarviseo.api --no-browser                 # 다른 창에서 로컬 팀 웹을 켠다
    set JARVISEO_BASIC_AUTH_PASSWORD=...     # 클라우드 접속 비밀번호(Caddy). 화면에 찍지 않는다
    python scripts/compare_web_latency.py --photo <사진.jpg> --label <성분표 사진.jpg> \\
        http://localhost:8000 https://jarviseo.duckdns.org

http 주소에는 비밀번호를 붙이지 않고, https 주소에만 아이디 jarviseo 와 위 비밀번호를 붙인다.
"""

import argparse
import base64
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote


def post(url: str, body: bytes, content_type: str, auth: str | None) -> tuple[float, dict]:
    headers = {"Content-Type": content_type}
    if auth:
        headers["Authorization"] = "Basic " + base64.b64encode(auth.encode()).decode()
    request = urllib.request.Request(url, data=body, headers=headers)
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    return time.perf_counter() - started, data


def main() -> None:
    parser = argparse.ArgumentParser(description="팀 웹 응답 시간 비교(로컬 · 클라우드)")
    parser.add_argument("bases", nargs="+", help="팀 웹 주소들. 예: http://localhost:8000")
    parser.add_argument("--photo", type=Path, required=True, help="질문에 붙일 사진")
    parser.add_argument("--label", type=Path, required=True, help="성분 확인에 보낼 성분표 사진")
    parser.add_argument("--question", default="이거 뭐야?")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--user", default="jarviseo")
    args = parser.parse_args()

    password = os.environ.get("JARVISEO_BASIC_AUTH_PASSWORD", "")
    photo = "data:image/jpeg;base64," + base64.b64encode(args.photo.read_bytes()).decode()
    ask = json.dumps({"question": args.question, "image": photo}).encode()
    label = args.label.read_bytes()
    for base in args.bases:
        auth = f"{args.user}:{password}" if base.startswith("https") and password else None
        rows = {"질문": [], "성분": []}
        for _ in range(args.repeat):
            sec, data = post(f"{base}/api/ask", ask, "application/json", auth)
            rows["질문"].append((sec, str(data.get("answer") or data)[:40]))
        for _ in range(args.repeat):
            url = f"{base}/api/ingredient?allergens={quote('모두')}"  # 법정 알레르기 전체
            sec, data = post(url, label, "image/jpeg", auth)
            rows["성분"].append((sec, str(data.get("verdict") or data)[:40]))
        print(f"== {base}")
        for kind, values in rows.items():
            times = [t for t, _ in values]
            print(f"  {kind} " + " ".join(f"{t:.1f}" for t in times)
                  + f" · 중앙 {statistics.median(times):.1f}초 · 예: {values[0][1]}")  # fmt: skip


if __name__ == "__main__":
    main()
