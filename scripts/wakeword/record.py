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
    python scripts/wakeword/record.py wake 100 --tts                   # TTS 로 합성 (키 필요)

--tts 는 마이크 대신 OpenAI TTS 로 목소리 11종 × 속도 3종 × 문장부호(억양) 3종 × 말투 9종을
바꿔 가며 합성한다. 실제 녹음과 섞지 않도록 metadata.csv 의 source 가 tts 로 남는다.

label
    wake        호출어 "자비서". 놓치면 FRR 에 잡힌다
    similar     비슷한 발음. 깨어나면 FAR 에 잡힌다
    background  TV·영상 대화를 길게. 깨어난 횟수로 시간당 오탐(FA/h)을 잰다
"""

import argparse
import csv
import itertools
import os
import random
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from jarviseo import config
from jarviseo.voice.mic import BLOCK, SAMPLE_RATE, Microphone
from jarviseo.voice.vad import record_speech, voiced_seconds

ROOT = config.DATASETS_DIR / "wakeword"
TEXTS = {
    "wake": [config.WAKE_WORD],
    # wakeword.py 의 측정 방법에 적힌 유사어. "자비스"는 STT 가 "자비서"를 잘못 받아쓰던 말이다.
    "similar": ["자비스", "아비서", "자비", "비서"],
}
# gpt-4o-mini-tts 목소리. 속도는 TextToSpeech 의 SLOW 0.85 / NORMAL 1.0 / FAST 1.2 를 쓴다.
VOICES = "alloy ash ballad coral echo fable nova onyx sage shimmer verse".split()
SPEEDS = ["SLOW", "NORMAL", "FAST"]
# 말투는 instructions 로 바꾼다(속도는 instructions 로 안 바뀌어서 speed 를 따로 준다).
# 목소리가 11종뿐이라 수천 개를 만들면 같은 소리가 반복된다. 말투를 섞어 조합을 9배로 늘린다.
STYLES = [
    "",
    "멀리 있는 사람을 부르듯 크고 또렷하게",
    "작은 목소리로 속삭이듯",
    "급하게 빨리",
    "졸리고 피곤한 목소리로",
    "밝고 들뜬 목소리로",
    "무뚝뚝하고 낮게",
    "혼잣말하듯 웅얼거리며",
    "짜증 난 목소리로",
]
FIELDS = ["file", "label", "text", "speaker", "distance", "env", "source", "mic", "seconds", "at"]
# 학습·평가에 쓰면 안 되는 클립의 기준. TTS 가 가끔 거의 무음이거나 말이 아닌 긴 잡음을 돌려줬다.
MIN_VOICED_SEC = 0.25  # 실제 녹음은 짧게 불러도 0.39초 이상, 망가진 합성은 0.21초 이하였다
MAX_TTS_SEC = 6.0  # 한 단어 합성은 길어도 5초 안쪽. 8~9초짜리는 받아써도 빈 글자인 잡음이었다


def problem(audio: np.ndarray, source: str) -> str | None:
    """학습·평가에 쓰면 안 되는 클립이면 그 이유를, 괜찮으면 None 을 돌려준다."""
    if voiced_seconds(audio) < MIN_VOICED_SEC:
        return "말소리 없음"
    if source == "tts" and len(audio) / SAMPLE_RATE > MAX_TTS_SEC:
        return "합성이 너무 김"
    return None


def sync(f) -> None:
    """파일을 디스크에 바로 쓴다. 안 하면 전원이 꺼질 때 wav 소리가 0 으로 남거나,
    metadata.csv 끝이 NUL 바이트로 채워졌다(10/7 노트북이 꺼졌을 때 실제로 생김)."""
    f.flush()
    os.fsync(f.fileno())


def save(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        with wave.open(f, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(audio.astype("<i2").tobytes())
        sync(f)


def write_meta(folder: Path, path: Path, seconds: float, **row: str) -> None:
    """folder/metadata.csv 에 파일 한 개의 정보를 한 줄 더한다. wav 를 다 쓴 뒤에 부른다."""
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
        sync(f)


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


def synth_clips(label: str, count: int, texts: list[str], folder: Path) -> None:
    """마이크 대신 TTS 로 목소리·속도·문장부호·말투를 바꿔 가며 합성해 16kHz 로 낮춰 저장한다."""
    import httpx2
    from openai import OpenAI, OpenAIError
    from scipy.signal import resample_poly

    from jarviseo.voice.tts import SAMPLE_RATE as TTS_RATE
    from jarviseo.voice.tts import SPEEDS as SPEED_VALUES

    # TextToSpeech 는 instructions 를 받지 않아서 API 를 바로 부른다.
    client = OpenAI(api_key=config.OPENAI_API_KEY or None, timeout=30.0, max_retries=2)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 같은 글자라도 "!" "?" 를 붙이면 억양이 달라진다.
    variants = [t + mark for t in texts for mark in ("", "!", "?")]
    combos = list(itertools.product(VOICES, SPEEDS, variants, STYLES))
    random.Random(0).shuffle(combos)  # count 가 조합 수보다 적어도 목소리가 골고루 섞이게
    # 다시 돌리면 이미 만든 개수만큼 건너뛰고 이어서 만든다. 앞에서 만든 조합과 겹치지 않게.
    done = len(list((folder / label).glob("*-tts-*.wav")))
    todo = itertools.islice(itertools.cycle(combos), done, None)
    n = 0
    while n < count:
        voice, speed, text, style = next(todo)
        extra = {"instructions": style} if style else {}
        if speed != "NORMAL":  # NORMAL 은 speed 를 안 보낸다(TextToSpeech 와 같게)
            extra["speed"] = SPEED_VALUES[speed]
        try:
            with client.audio.speech.with_streaming_response.create(
                model=config.TTS_MODEL, voice=voice, input=text, response_format="pcm", **extra
            ) as response:
                pcm = np.frombuffer(response.read(), dtype="<i2")
        except (OpenAIError, httpx2.HTTPError) as e:
            # 받는 도중에 끊기면(ReadTimeout) 반쪽 소리라 버리고 다음 조합으로 넘어간다.
            print(f"  합성 실패, 건너뛴다: {text!r} {voice} {speed} ({e})")
            continue
        audio = resample_poly(pcm.astype(np.float32), SAMPLE_RATE, TTS_RATE)
        audio = np.clip(audio, -32768, 32767).astype(np.int16)
        if why := problem(audio, "tts"):
            print(f"  {why}, 버리고 다음 조합으로: {text!r} {voice} {speed} {style or '기본'}")
            continue
        n += 1
        path = folder / label / f"{stamp}-tts-{n:03d}.wav"
        seconds = len(audio) / SAMPLE_RATE
        save(path, audio)
        meta = {"speaker": f"tts-{voice}", "distance": "-", "env": "synthetic", "source": "tts"}
        # 마이크 칸에는 무엇으로 만들었는지(모델·속도·말투)를 적는다.
        how = f"{config.TTS_MODEL} {speed} {style or '기본'}"
        write_meta(folder, path, seconds, label=label, text=text, mic=how, **meta)
        print(f"[{n}/{count}] {text!r} {voice} {speed} {style or '기본'} → {path.name}")


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
    parser.add_argument("--tts", action="store_true", help="마이크 대신 TTS 로 합성한다")
    args = parser.parse_args()
    if args.tts and args.label == "background":
        parser.error("background 는 실제 소리를 녹음해야 한다. --tts 와 같이 쓸 수 없다")

    folder = ROOT / args.split
    texts = args.text or TEXTS.get(args.label, [])
    try:
        if args.tts:
            synth_clips(args.label, args.count, texts, folder)
            return
        meta = {
            "speaker": args.speaker,
            "distance": args.distance,
            "env": args.env,
            "source": "mic",
            "mic": mic_name(args.device),
        }
        if args.label == "background":
            record_long(args.minutes, folder, args.device, meta)
        else:
            record_clips(args.label, args.count, texts, folder, args.device, meta)
    except KeyboardInterrupt:
        print("\n멈췄다. 그때까지 저장한 파일은 남아 있다")


if __name__ == "__main__":
    main()
