"""웨이크워드 녹음 도구.  담당: 문태현

말할 문구를 띄우고, 한 번 말할 때마다 VAD 로 잘라 16kHz 모노 wav 로 저장한다.
평가용(eval)과 학습용(train)은 폴더를 나눠 저장한다. 저장 위치는 data/datasets/wakeword/ 이고
Git 에는 올라가지 않는다(.gitignore).

파일마다 화자·거리·환경을 그 폴더의 metadata.csv 에 한 줄씩 남긴다. 평가 스크립트가 이걸 보고
조용한 환경·소음 환경을 나눠 센다. 화자는 실명 대신 s01, s02 처럼 번호로 적는다.

실행:
    python scripts/wakeword/record.py wake 30 --env quiet              # "자비서" 30번
    python scripts/wakeword/record.py wake 30 --env noisy              # 소음을 틀어 놓고 30번
    python scripts/wakeword/record.py similar 20                       # 비슷한 발음을 돌아가며
    python scripts/wakeword/record.py background --minutes 30 --env tv # TV 소리를 30분 그대로
    python scripts/wakeword/record.py wake 30 --split train --speaker s02 --distance 2m

label
    wake        호출어 "자비서". 놓치면 FRR 에 잡힌다
    similar     비슷한 발음. 깨어나면 FAR 에 잡힌다
    background  TV·영상 대화를 길게. 깨어난 횟수로 시간당 오탐(FA/h)을 잰다
"""

import argparse
import csv
import itertools
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from jarviseo import config
from jarviseo.voice.mic import BLOCK, SAMPLE_RATE, Microphone
from jarviseo.voice.vad import record_speech

ROOT = config.DATASETS_DIR / "wakeword"
TEXTS = {
    "wake": [config.WAKE_WORD],
    # wakeword.py 의 측정 방법에 적힌 유사어. "자비스"는 STT 가 "자비서"를 잘못 받아쓰던 말이다.
    "similar": ["자비스", "아비서", "자비", "비서"],
}
FIELDS = ["file", "label", "text", "speaker", "distance", "env", "source", "mic", "seconds", "at"]


def save(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.astype("<i2").tobytes())


def write_meta(folder: Path, path: Path, seconds: float, **row: str) -> None:
    """folder/metadata.csv 에 파일 한 개의 정보를 한 줄 더한다."""
    meta = folder / "metadata.csv"
    new = not meta.exists()
    with meta.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(
            {
                "file": path.relative_to(folder).as_posix(),
                "seconds": f"{seconds:.2f}",
                "at": datetime.now().isoformat(timespec="seconds"),
                **row,
            }
        )


def record_clips(
    label: str, count: int, texts: list[str], folder: Path, device: int | None, meta: dict
) -> None:
    """문구를 하나씩 띄우고, 말할 때마다 잘라서 저장한다. 말을 못 잡으면 같은 문구를 다시 띄운다."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prompts = itertools.cycle(texts)
    with Microphone(device) as mic:
        blocks = mic.blocks()
        n = 0
        while n < count:
            text = next(prompts)
            print(f"[{n + 1}/{count}] {text!r} 라고 말하세요")
            speech = None
            while speech is None:
                speech = record_speech(blocks, timeout=10.0)
                if speech is None:
                    print("  말을 못 잡았다. 다시 말하세요")
            n += 1
            path = folder / label / f"{stamp}-{n:03d}.wav"
            seconds = len(speech.audio) / SAMPLE_RATE
            save(path, speech.audio)
            write_meta(folder, path, seconds, label=label, text=text, **meta)
            print(f"  저장 {path.name} ({seconds:.2f}s)")


def record_long(minutes: float, folder: Path, device: int | None, meta: dict) -> None:
    """마이크 소리를 그대로 minutes 분 녹음한다. Ctrl+C 로 멈춰도 그때까지는 남는다."""
    path = folder / "background" / f"{datetime.now():%Y%m%d-%H%M%S}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    total = int(minutes * 60 * SAMPLE_RATE / BLOCK)
    per_minute = 60 * SAMPLE_RATE // BLOCK
    print(f"{minutes}분 녹음한다. 멈추려면 Ctrl+C")
    n = 0
    try:
        with Microphone(device) as mic, wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            for n, (block, _) in enumerate(mic.blocks(), start=1):
                w.writeframes(block.astype("<i2").tobytes())
                if n % per_minute == 0:
                    print(f"  {n // per_minute}분")
                if n >= total:
                    break
    finally:
        write_meta(folder, path, n * BLOCK / SAMPLE_RATE, label="background", text="", **meta)
        print(f"저장 {path}")


def mic_name(device: int | None) -> str:
    import sounddevice as sd

    return sd.query_devices(device, "input")["name"]


def main() -> None:
    parser = argparse.ArgumentParser(description="웨이크워드 녹음")
    parser.add_argument("label", choices=["wake", "similar", "background"])
    parser.add_argument("count", type=int, nargs="?", default=30, help="녹음할 횟수")
    parser.add_argument("--split", choices=["eval", "train"], default="eval")
    parser.add_argument("--text", action="append", help="띄울 문구. 여러 번 주면 돌아가며 띄운다")
    parser.add_argument("--minutes", type=float, default=30, help="background 녹음 길이(분)")
    parser.add_argument("--speaker", default="s01", help="화자 번호. 실명은 쓰지 않는다")
    parser.add_argument("--distance", default="0.5m", help="입과 마이크 사이 거리")
    parser.add_argument("--env", default="quiet", help="quiet / noisy / tv 등")
    parser.add_argument("--device", type=int, help="sounddevice 입력 장치 번호")
    args = parser.parse_args()

    folder = ROOT / args.split
    meta = {
        "speaker": args.speaker,
        "distance": args.distance,
        "env": args.env,
        "source": "mic",
        "mic": mic_name(args.device),
    }
    try:
        if args.label == "background":
            record_long(args.minutes, folder, args.device, meta)
        else:
            texts = args.text or TEXTS[args.label]
            record_clips(args.label, args.count, texts, folder, args.device, meta)
    except KeyboardInterrupt:
        print("\n멈췄다. 그때까지 저장한 파일은 남아 있다")


if __name__ == "__main__":
    main()
