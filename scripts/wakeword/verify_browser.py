"""브라우저 호출어 2단계 확인을 실제 녹음으로 잰다 (#218).  담당: 문태현

eval 실제 녹음(마이크)을 1차 모델에 흘려 0.97(config)로 깬 곳마다, 브라우저가 보내는 것과 같은 구간
(감지한 80ms 까지 직전 JARVISEO_WAKE_CHECK_SEC 초)을 잘라 팀 웹의 POST /api/wake-check 로 보낸다.
서버 안에서는 노트북과 같은 voice/wakecheck.py 가 OpenAI 로 받아쓴다
(.env 의 OPENAI_API_KEY, 깬 곳마다 1번).
1차 점수는 파이썬으로 낸다. 브라우저 점수와 같다는 것은 compare_browser.py 로 쟀다.

실행:
    python scripts/wakeword/verify_browser.py                 # 실제 녹음 전부
    python scripts/wakeword/verify_browser.py --env noisy tv  # 일부 환경만
    python scripts/wakeword/verify_browser.py --tail 4        # 감지 뒤 80ms 4개를 더 붙여

결과: data/datasets/wakeword/results/browser-check-<날짜시각>.csv
      (파일 · 프레임 · 통과 · 받아쓴 글자 · ms)
"""

import argparse
import csv
import sys
import tempfile
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev  # noqa: E402

from jarviseo import config  # noqa: E402


class _NoChecker:
    synonyms: dict = {}

    def close(self):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="브라우저 호출어 2단계 확인 실측")
    parser.add_argument("--env", nargs="+", help="이 환경만(quiet noisy tv talk ...)")
    parser.add_argument("--tail", type=int, default=0, help="감지 뒤 더 붙일 80ms 수(기본 0)")
    parser.add_argument("--threshold", type=float, default=config.WAKEWORD_THRESHOLD)
    args = parser.parse_args()

    from fastapi.testclient import TestClient

    from jarviseo.api import main as web
    from jarviseo.memory import MemoryStore

    folder = ev.ROOT / "eval"
    with (folder / "metadata.csv").open(encoding="utf-8") as f:
        meta = [
            m
            for m in csv.DictReader(f)
            if m["source"] == "mic"
            and (folder / m["file"]).exists()
            and (not args.env or m["env"] in args.env)
        ]
    model = ev.load_model(config.WAKEWORD_MODEL)
    frames = round(config.WAKE_CHECK_SEC * ev.SAMPLE_RATE / ev.FRAME)  # 1.6초 = 20개
    gap = round(ev.REFRACTORY_SEC * ev.SAMPLE_RATE / ev.FRAME)
    db = Path(tempfile.mkdtemp()) / "check.db"
    store = MemoryStore(database_url=f"sqlite:///{db}")
    store.init_schema()
    rows = []
    with TestClient(web.create_app(store=store, checker=_NoChecker())) as c:
        for i, m in enumerate(meta):
            np.random.seed(i)
            scores = ev.frame_scores(model, folder / m["file"])
            with wave.open(str(folder / m["file"])) as f:
                audio = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
            pad = np.zeros(ev.SAMPLE_RATE, np.int16)  # frame_scores 처럼 앞뒤 무음 1초
            padded = np.concatenate([pad, audio, pad])
            until = -1
            for k in np.flatnonzero(scores >= args.threshold):
                if k < until:
                    continue
                until = k + gap
                end = (k + 1 + args.tail) * ev.FRAME
                clip = padded[max(0, end - (frames + args.tail) * ev.FRAME) : end]
                d = c.post("/api/wake-check", content=clip.astype("<i2").tobytes()).json()
                rows.append(
                    {"file": m["file"], "label": m["label"], "env": m["env"], "frame": int(k)}
                    | {"passed": d["passed"], "text": d["text"] or "", "ms": d["ms"]}
                )
            print(f"\r  {i + 1}/{len(meta)}", end="", flush=True)
    print()

    out = ev.ROOT / "results" / f"browser-check-{datetime.now():%Y%m%d-%H%M}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ev.write_csv(out, list(rows[0]) if rows else ["file"], rows)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["label"], r["env"]), []).append(r)
    for (label, env), rs in sorted(groups.items()):
        ms = sorted(r["ms"] for r in rs)
        print(
            f"{label:10} {env:8} 깨어남 {len(rs):3}  2차 통과 {sum(r['passed'] for r in rs):3}"
            f"  왕복 중앙 {ms[len(ms) // 2]}ms"
        )
        for r in rs:
            if r["passed"] != (label == "wake"):
                print(f"    {'통과' if r['passed'] else '거절'} {r['text']!r}  {r['file']}")
    print(f"결과 {out}")


if __name__ == "__main__":
    main()
