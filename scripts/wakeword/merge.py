"""팀원 녹음 zip 합치기.  담당: 문태현

팀원이 record.py 로 녹음한 폴더를 압축해 보낸 zip(train_s02.zip 등)을 이 PC 의 split 폴더에 합친다.
zip 은 풀지 않고 그대로 읽는다. 폴더째 묶었든 안쪽만 묶었든 metadata.csv 가 있는 곳을 기준으로 본다.

- 파일 이름 앞에 화자 번호를 붙인다. 두 사람이 같은 초에 녹음을 시작하면 이름이 같아진다
- wav 가 없는 metadata 줄은 버린다(녹음한 뒤 지운 것). 화자가 번호 꼴(s02)이 아닌 줄도 버린다
- 16kHz 모노가 아닌 wav 는 버린다
- 다른 split 에 이미 있는 화자는 합치지 않고 멈춘다. 같은 사람 목소리로 학습하고 평가하면 안 된다
- 이미 합친 파일은 건너뛰어서, 같은 zip 을 다시 돌려도 두 번 들어가지 않는다

실행:
    python scripts/wakeword/merge.py data/datasets/wakeword/incoming/train_s02.zip ...
    python scripts/wakeword/merge.py --split eval <zip> ...    # 평가용 녹음을 받았을 때

합친 뒤에는 check.py 로 한 번 거른다.
"""

import argparse
import csv
import io
import re
import sys
import wave
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from record import FIELDS, ROOT, SAMPLE_RATE, sync

SPEAKER = re.compile(r"s\d+")


def speakers(folder: Path) -> set[str]:
    meta = folder / "metadata.csv"
    if not meta.exists():
        return set()
    with meta.open(encoding="utf-8", newline="") as f:
        return {r["speaker"] for r in csv.DictReader(f)}


def read_zip(path: Path) -> list[tuple[dict, bytes]]:
    """zip 에서 합칠 (metadata 한 줄, wav 바이트) 를 골라 낸다. 버린 줄은 이유를 찍는다."""
    with zipfile.ZipFile(path) as z:
        (meta_name,) = [n for n in z.namelist() if n.endswith("metadata.csv")]
        base = PurePosixPath(meta_name).parent
        rows = list(csv.DictReader(io.StringIO(z.read(meta_name).decode("utf-8"))))
        names = set(z.namelist())
        keep, dropped = [], Counter()
        for r in rows:
            name = (base / r["file"]).as_posix()
            if name not in names:
                dropped["wav 없음"] += 1
                continue
            if not SPEAKER.fullmatch(r["speaker"]):
                dropped[f"화자 번호가 {r['speaker']!r}"] += 1
                continue
            data = z.read(name)
            with wave.open(io.BytesIO(data), "rb") as w:
                if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                    dropped["16kHz 모노 아님"] += 1
                    continue
            keep.append((r, data))
    print(
        f"{path.name}: metadata {len(rows)}줄 중 쓸 수 있는 클립 {len(keep)}개", dict(dropped) or ""
    )
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description="팀원 녹음 zip 합치기")
    parser.add_argument("zips", nargs="+", type=Path)
    parser.add_argument("--split", choices=["eval", "train"], default="train")
    args = parser.parse_args()

    folder = ROOT / args.split
    other = speakers(ROOT / ("eval" if args.split == "train" else "train"))
    clips = [clip for z in args.zips for clip in read_zip(z)]
    if overlap := {r["speaker"] for r, _ in clips} & other:
        sys.exit(f"{sorted(overlap)} 는 다른 split 에 이미 있는 화자다. 합치지 않았다")

    added = Counter()
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder / "metadata.csv"
    new = not meta.exists()
    with meta.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        if new:
            writer.writeheader()
        for r, data in clips:
            src = PurePosixPath(r["file"])
            rel = (src.parent / f"{r['speaker']}-{src.name}").as_posix()
            dest = folder / rel
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as out:
                out.write(data)
                sync(out)
            writer.writerow({k: r.get(k, "") for k in FIELDS} | {"file": rel})
            sync(f)
            added[(r["speaker"], r["label"])] += 1
    print(f"{args.split} 에 {sum(added.values())}개 더했다", dict(sorted(added.items())))


if __name__ == "__main__":
    main()
