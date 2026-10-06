"""웨이크워드 모델 평가.  담당: 문태현

eval 폴더의 녹음(record.py 로 만든 것)을 openWakeWord 모델에 흘려 파일마다 점수를 낸다.
모델은 80ms 마다 0~1 점수를 낸다. 짧은 녹음(wake·similar)은 앞뒤에 무음 1초를 붙여 흘리고
가장 높은 점수를 남긴다. 긴 녹음(background)은 프레임 점수를 그대로 들고 있는다.

평가용 녹음만 읽는다. 학습용(train)은 여기서 읽지 않는다.

실행:
    python scripts/wakeword/evaluate.py                                   # 기성 hey_jarvis
    python scripts/wakeword/evaluate.py --model data/models/jarviseo.onnx # 커스텀 학습 뒤

결과: data/datasets/wakeword/results/<모델>-<날짜시각>-scores.csv
"""

import argparse
import csv
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from jarviseo import config

ROOT = config.DATASETS_DIR / "wakeword"
SAMPLE_RATE = 16000
FRAME = 1280  # openWakeWord 는 80ms(16kHz 1280샘플)씩 받는다
FIELDS = ["file", "label", "source", "env", "speaker", "distance", "text", "seconds", "max_score"]


def load_model(model: str):
    """이름(hey_jarvis 등 기성 모델)이나 .onnx 경로를 받아 openWakeWord 모델을 연다."""
    from openwakeword.model import Model
    from openwakeword.utils import download_models

    if not model.endswith(".onnx"):
        download_models([model])  # 처음 한 번만 받는다(약 1MB). 이미 있으면 건너뛴다
    return Model(wakeword_models=[model], inference_framework="onnx")


def frame_scores(model, path: Path) -> np.ndarray:
    """wav 하나를 처음부터 흘려 80ms 마다 나온 점수를 돌려준다."""
    with wave.open(str(path), "rb") as f:
        assert f.getframerate() == SAMPLE_RATE and f.getnchannels() == 1, path
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    model.reset()  # 앞 파일의 소리가 다음 파일 점수에 섞이지 않게
    (name,) = model.models
    return np.array([p[name] for p in model.predict_clip(audio, padding=1, chunk_size=FRAME)])


def evaluate(model, folder: Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    """metadata.csv 의 파일을 모두 흘린다. (파일별 행, background 파일별 프레임 점수) 를 돌려준다.

    녹음이 잘못돼 wav 를 지웠으면 metadata 에 줄이 남아 있어도 건너뛴다.
    """
    with (folder / "metadata.csv").open(encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    rows, backgrounds = [], {}
    for i, m in enumerate(meta, 1):
        path = folder / m["file"]
        if not path.exists():
            continue
        scores = frame_scores(model, path)
        if m["label"] == "background":
            backgrounds[m["file"]] = scores
        rows.append({k: m.get(k, "") for k in FIELDS} | {"max_score": f"{scores.max():.4f}"})
        print(f"\r  {i}/{len(meta)} {m['file']}", end="", flush=True)
    print()
    return rows, backgrounds


def main() -> None:
    parser = argparse.ArgumentParser(description="웨이크워드 평가")
    parser.add_argument("--model", default="hey_jarvis", help="기성 모델 이름이나 .onnx 경로")
    args = parser.parse_args()

    np.random.seed(0)  # model.reset() 이 버퍼를 난수로 채운다. 다시 돌려도 같은 점수가 나오게
    model = load_model(args.model)
    rows, _ = evaluate(model, ROOT / "eval")

    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"{Path(args.model).stem}-{datetime.now():%Y%m%d-%H%M}"
    path = out / f"{stamp}-scores.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"파일별 점수 {path}")


if __name__ == "__main__":
    main()
