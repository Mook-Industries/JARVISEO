"""웨이크워드 녹음 검사.  담당: 문태현

eval·train 폴더의 wav 를 모두 열어 학습·평가에 쓰면 안 되는 클립을 찾는다.
TTS 가 가끔 거의 무음이거나 말이 아닌 긴 잡음을 돌려줬다(record.problem 의 기준).
그대로 두면 "자비서"라고 붙은 무음을 학습하거나, 평가에서 무음을 놓친 것으로 센다.
팀원 녹음을 합친 뒤에도 한 번 돌린다.

찾는 것
    - 말소리가 0.25초도 안 되는 클립, 6초가 넘는 합성 클립
    - 16kHz 모노가 아닌 wav
    - metadata.csv 에는 있는데 wav 가 없는 줄, wav 는 있는데 metadata.csv 에 없는 파일
      (전원이 꺼지면 끝줄이 NUL 바이트로 깨져 이렇게 됐다)

실행:
    python scripts/wakeword/check.py train          # 찾기만 한다
    python scripts/wakeword/check.py train --fix    # wav 를 지우고 metadata.csv 에서도 뺀다
"""

import argparse
import csv
import os
import wave
from collections import Counter

import numpy as np
from record import FIELDS, ROOT, SAMPLE_RATE, problem, sync


def main() -> None:
    parser = argparse.ArgumentParser(description="웨이크워드 녹음 검사")
    parser.add_argument("split", choices=["eval", "train"])
    parser.add_argument("--fix", action="store_true", help="걸린 wav 를 지우고 metadata 에서 뺀다")
    args = parser.parse_args()

    folder = ROOT / args.split
    meta = folder / "metadata.csv"
    with meta.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    listed = {r["file"] for r in rows}
    keep, bad = [], []
    for r in rows:
        path = folder / (r["file"] or "")
        if not r["file"] or not path.is_file():
            bad.append((r["file"], "wav 없음"))
            continue
        with wave.open(str(path), "rb") as w:
            if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1:
                bad.append((r["file"], "16kHz 모노 아님"))
                continue
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        # background 는 TV 소리를 그대로 길게 녹음한 것이라 말소리 기준을 대지 않는다
        if r["label"] != "background" and (why := problem(audio, r["source"])):
            bad.append((r["file"], why))
            continue
        keep.append(r)
    orphans = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.wav"))
    orphans = [f for f in orphans if f not in listed]
    bad += [(f, "metadata 없음") for f in orphans]

    print(f"{args.split}: metadata {len(rows)}줄, 문제 {len(bad)}개")
    for why, n in Counter(why for _, why in bad).most_common():
        print(f"  {why} {n}개")
    for file, why in bad[:20]:
        print(f"    {file}  ({why})")
    if not args.fix or not bad:
        return

    for file, _ in bad:
        if file and (folder / file).is_file():
            (folder / file).unlink()
    # 다 쓴 뒤에 바꿔치기한다. 쓰는 도중에 꺼져도 원래 metadata.csv 는 남는다.
    tmp = meta.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(keep)
        sync(f)
    os.replace(tmp, meta)
    print(f"지웠다. 남은 클립 {len(keep)}개")


if __name__ == "__main__":
    main()
