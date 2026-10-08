"""웨이크워드 학습 데이터 준비.  담당: 문태현

train 폴더의 녹음을 openWakeWord 학습 코드(openwakeword/train.py)가 읽는 폴더 4개로 나눈다.
    positive_train / positive_test   "자비서"
    negative_train / negative_test   그 밖의 말(비슷한 발음 등). 이 말에는 깨지 않게 배운다
_test 는 학습 중에 가장 좋은 모델을 고르는 검증용이다. 라벨·화자 묶음마다 10% 씩 떼어 놓는다.
평가용(eval) 녹음은 읽지 않는다. 학습 뒤 평가는 evaluate.py 로 따로 한다.

앞뒤 무음을 잘라서 넣는다. openWakeWord 는 클립을 학습 창(2초)의 끝에 붙이고 그 뒤에 0~0.2초만
다른 소리를 둔다. 합성 클립은 "자비서" 뒤에 무음이 0.6초쯤 붙어 있어서, 그대로 넣으면
"자비서 뒤의 무음"까지 배운다. 그러면 호출어 뒤에 바로 질문을 이어 말할 때 늦게 깨거나 못 깬다.

자르고 나서도 2초(학습 창)가 넘는 클립은 버린다. 소음 속 실제 녹음은 VAD 가 소음을 말로 보고
녹음을 10초까지 끌어서, "자비서"가 여러 번 들어 있거나 어디 있는지 알 수 없었다.

실행:
    python scripts/wakeword/prepare.py

결과: data/datasets/wakeword/oww/
    jarviseo/positive_train/ ... negative_test/   잘라 낸 wav
    jarviseo/split.csv                            원본 파일이 어디로 갔는지
    jarviseo.zip                                  위 폴더를 묶은 것. Drive 에 올려 Colab 에서 푼다
"""

import argparse
import csv
import random
import shutil
import wave
import zipfile
from collections import Counter, defaultdict

import numpy as np
from record import ROOT, SAMPLE_RATE, save

FRAME = 480  # 30ms 씩 소리 크기를 잰다
PAD_SEC = 0.1  # 말소리 앞뒤로 이만큼은 남긴다. 첫 자음·끝 모음이 잘리지 않게
MAX_SEC = 2.0  # openWakeWord 학습 창. 이보다 길면 앞부분만 남기고 잘린다
SPLIT_FIELDS = ["file", "out", "label", "source", "speaker", "split", "seconds"]


def trim(audio: np.ndarray) -> np.ndarray | None:
    """앞뒤 무음을 잘라 낸 소리를 돌려준다. 말소리를 못 찾으면 None.

    말소리로 보는 기준은 "가장 큰 구간보다 30dB 안쪽, 바닥 소음보다 10dB 위"다.
    VAD(vad.voiced_seconds)는 조용한 마이크의 실제 녹음에서 말소리를 거의 못 잡아서 쓰지 않는다.
    소리 크기를 그 클립 안에서 비교하므로 마이크 음량이 달라도 같은 기준이 된다.
    """
    n = len(audio) // FRAME
    if n == 0:
        return None
    frames = audio[: n * FRAME].astype(np.float32).reshape(n, FRAME)
    db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-6)
    level = max(db.max() - 30, np.percentile(db, 10) + 10)
    loud = np.flatnonzero(db > level)
    if len(loud) == 0:
        return None
    pad = int(PAD_SEC * SAMPLE_RATE)
    start = max(0, loud[0] * FRAME - pad)
    end = min(len(audio), (loud[-1] + 1) * FRAME + pad)
    return audio[start:end]


def group(row: dict) -> str:
    """검증용을 고르게 떼어 내려고 묶는 단위. 실제 녹음은 화자별, 합성은 한 묶음."""
    who = "tts" if row["source"] == "tts" else row["speaker"]
    return f"{row['label']}/{who}"


def main() -> None:
    parser = argparse.ArgumentParser(description="웨이크워드 학습 데이터 준비")
    parser.add_argument("--name", default="jarviseo", help="학습 설정의 model_name 과 같게")
    parser.add_argument("--test", type=float, default=0.1, help="검증용으로 떼어 낼 비율")
    args = parser.parse_args()

    src = ROOT / "train"
    with (src / "metadata.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    # background(길게 녹음한 TV 소리)는 클립이 아니라서 여기서 쓰지 않는다
    rows = [r for r in rows if r["label"] != "background"]

    groups = defaultdict(list)
    for r in rows:
        groups[group(r)].append(r)
    split = {}
    for members in groups.values():
        members = sorted(members, key=lambda r: r["file"])
        random.Random(0).shuffle(members)  # 다시 돌려도 같은 클립이 검증용으로 간다
        n_test = round(len(members) * args.test)
        for i, r in enumerate(members):
            split[r["file"]] = "test" if i < n_test else "train"

    out = ROOT / "oww" / args.name
    if out.exists():
        shutil.rmtree(out)  # 이전에 만든 결과. 원본(train)은 건드리지 않는다
    kept, dropped, seconds = [], Counter(), []
    for r in rows:
        with wave.open(str(src / r["file"]), "rb") as w:
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        cut = trim(audio)
        if cut is None:
            dropped["말소리 못 찾음"] += 1
            continue
        if len(cut) > MAX_SEC * SAMPLE_RATE:
            dropped[f"잘라도 {MAX_SEC:.0f}초 넘음"] += 1
            continue
        kind = "positive" if r["label"] == "wake" else "negative"
        path = out / f"{kind}_{split[r['file']]}" / f"{r['label']}-{r['file'].split('/')[-1]}"
        save(path, cut)
        sec = len(cut) / SAMPLE_RATE
        seconds.append(sec)
        kept.append(
            {
                "file": r["file"],
                "out": path.relative_to(out).as_posix(),
                "label": r["label"],
                "source": r["source"],
                "speaker": r["speaker"],
                "split": split[r["file"]],
                "seconds": f"{sec:.2f}",
            }
        )

    with (out / "split.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, SPLIT_FIELDS)
        writer.writeheader()
        writer.writerows(kept)

    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(out.parent).as_posix())

    counts = Counter(row["out"].split("/")[0] for row in kept)
    print(f"원본 {len(rows)}개 → {len(kept)}개 ({out})")
    for folder in ("positive_train", "positive_test", "negative_train", "negative_test"):
        print(f"  {folder:15s} {counts[folder]}")
    for why, n in dropped.most_common():
        print(f"  버림: {why} {n}개")
    print(f"  길이 중앙값 {np.median(seconds):.2f}초, 최대 {max(seconds):.2f}초")
    print(f"묶음 {archive} ({archive.stat().st_size / 1e6:.0f}MB)")


if __name__ == "__main__":
    main()
