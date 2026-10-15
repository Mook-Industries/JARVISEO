"""받아쓰기 모델을 같은 녹음으로 비교한다 (지연·정확도·말끝 대기별 잘림).  담당: 문태현

녹음은 data/datasets/stt/<이름>/ 아래 16kHz 모노 wav 와 metadata.csv(file, text) 다.

- --record 이름 : QUESTIONS 를 한 문장씩 띄우고 마이크로 녹음한다(사람 목소리)
- --tts         : 같은 문장을 TTS 목소리 몇 개로 합성해 tts/ 에 둔다.
                  합성 목소리라 정확도는 사람보다 후하게 나온다
- --mix 소리    : tts/ 에 배경 소리(16kHz 모노 wav)를 --snr dB 로 섞어 tts_noise<snr>/ 에 둔다
- (기본) 비교   : 폴더의 wav 를 모델마다 받아쓴다. 지연 중앙값·p95, 맞은 문장 수, CER 을 낸다.
                  문장마다 모델 순서를 돌려 가며 보내서, 그사이 API 속도가 바뀌어도 고르게 섞인다.
                  --hint 를 주면 모델마다 호출어 힌트와 그 힌트를 둘 다 잰다(되묻기 대답용).
                  --questions 로 QUESTIONS 번호를 골라 그 문장만 잴 수 있다
- --silence     : 말끝 대기(silence_sec)별로 VAD 가 말을 중간에 끊는지 센다. API 는 안 쓴다

맞음은 clean() 으로 호출어·필러를 뺀 뒤 띄어쓰기·문장부호를 빼고 글자가 같은지로 본다.
CER 은 그 글자 기준 편집 거리 / 정답 글자 수다. 결과 CSV 는 data/datasets/stt/results/ 에 남는다.

실행:
    python scripts/stt_compare.py --tts
    python scripts/stt_compare.py --mix data/datasets/wakeword/eval/background/x.wav --snr 10
    python scripts/stt_compare.py --record s01
    python scripts/stt_compare.py tts tts_noise10 s01
    python scripts/stt_compare.py tts --questions 15 16 17 18 --hint "컵, 병, 왼쪽, 오른쪽"
    python scripts/stt_compare.py s01 --silence 0.5 0.6 0.7
"""

import argparse
import csv
import re
import statistics
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from jarviseo.voice import SpeechToText
from jarviseo.voice.mic import BLOCK, SAMPLE_RATE
from jarviseo.voice.transcript import clean
from jarviseo.voice.vad import record_speech, voiced_seconds

ROOT = Path("data/datasets/stt")
MODELS = ["gpt-transcribe", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"]
VOICES = ["coral", "onyx"]
# 실제로 물을 만한 말. 호출어를 붙인 질문, 기억·성분 질문, 상표·숫자, 되묻기 대답(짧은 말)을 섞었다.
QUESTIONS = [
    "자비서, 이거 뭐야?",
    "자비서, 저거 뭐야?",
    "자비서, 내가 가리키는 거 뭐야?",
    "이 과자에 우유 들어 있어?",
    "이거 땅콩 들어갔는지 봐 줘.",
    "자비서, 아까 본 컵 어디 있었지?",
    "아까 본 음료 캔 이름이 뭐였지?",
    "어제 지갑 어디에 뒀지?",
    "핸드폰 어디 있었는지 기억나?",
    "갈아만든 배 캔이 몇 밀리리터야?",
    "바나프레소 컵에 뭐라고 적혀 있어?",
    "이 약 하루에 몇 번 먹어야 돼?",
    "지금 앞에 뭐가 있어?",
    "이거 유통기한 언제까지야?",
    "왼쪽 컵이요.",
    "오른쪽에 있는 거요.",
    "컵.",
    "병이요.",
    "자비서, 이 글자 읽어 줘.",
    "나트륨이 얼마나 들어 있어?",
]
# 녹음 앞뒤에 붙이는 무음(초). 뒤는 --silence 가 말끝 대기를 끝까지 잴 수 있게 길게 둔다.
LEAD_SEC, TAIL_SEC = 0.3, 1.5


def letters(text: str) -> str:
    return re.sub(r"[^가-힣A-Za-z0-9]", "", clean(text))


def cer(want: str, got: str) -> float:
    """글자 편집 거리 / 정답 글자 수."""
    prev = list(range(len(got) + 1))
    for i, a in enumerate(want, 1):
        cur = [i]
        for j, b in enumerate(got, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / max(len(want), 1)


def save(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.astype("<i2").tobytes())


def load(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels()) == (SAMPLE_RATE, 1), path
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def write_meta(folder: Path, rows: list[dict]) -> None:
    with (folder / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, ["file", "text", "voice"])
        writer.writeheader()
        writer.writerows(rows)


def read_meta(folder: Path) -> list[dict]:
    with (folder / "metadata.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pad(audio: np.ndarray) -> np.ndarray:
    lead, tail = np.zeros(round(LEAD_SEC * SAMPLE_RATE)), np.zeros(round(TAIL_SEC * SAMPLE_RATE))
    return np.concatenate([lead, audio, tail]).astype(np.int16)


def synth(tts, text: str) -> np.ndarray:
    """합성 한 번. 스트림이 도중에 끊기는 일이 있어서(read 타임아웃) 두 번까지 다시 한다."""
    for attempt in range(3):
        try:
            return np.frombuffer(b"".join(tts.synthesize(text)), dtype="<i2")
        except Exception as e:
            print(f"  합성 실패 {attempt + 1}번: {type(e).__name__}")
    raise RuntimeError(f"합성 실패: {text}")


def make_tts() -> None:
    """QUESTIONS 를 VOICES 로 합성해 tts/ 에 둔다."""
    from scipy.signal import resample_poly

    from jarviseo.voice import TextToSpeech

    rows = []
    for voice in VOICES:
        tts = TextToSpeech(voice=voice)
        for i, text in enumerate(QUESTIONS, 1):
            audio = pad(resample_poly(synth(tts, text).astype(np.float32), 2, 3))  # 24k → 16k
            name = f"{voice}-{i:02d}.wav"
            save(ROOT / "tts" / name, audio)
            rows.append({"file": name, "text": text, "voice": voice})
            print(f"  {name} {text}")
    write_meta(ROOT / "tts", rows)


def mix(noise: Path, snr_db: float) -> None:
    """tts/ 의 녹음마다 배경 소리의 다른 구간을 snr_db 로 섞어 tts_noise<snr>/ 에 둔다."""
    background = load(noise).astype(np.float32)
    rng = np.random.default_rng(0)
    out = ROOT / f"tts_noise{snr_db:g}"
    rows = []
    for row in read_meta(ROOT / "tts"):
        audio = load(ROOT / "tts" / row["file"]).astype(np.float32)
        start = rng.integers(0, len(background) - len(audio))
        noise_part = background[start : start + len(audio)]
        # SNR 은 앞뒤에 붙인 무음을 뺀 말소리 크기로 맞춘다
        lead, tail = round(LEAD_SEC * SAMPLE_RATE), round(TAIL_SEC * SAMPLE_RATE)
        speech = audio[lead : len(audio) - tail]
        gain = np.sqrt(np.mean(speech**2) / (np.mean(noise_part**2) + 1e-9))
        mixed = audio + noise_part * gain / 10 ** (snr_db / 20)
        save(out / row["file"], np.clip(mixed, -32768, 32767))
        rows.append({**row, "voice": f"{row['voice']}+noise{snr_db:g}"})
    write_meta(out, rows)
    print(f"{out} {len(rows)}개")


def record(name: str, device: int | None) -> None:
    """QUESTIONS 를 한 문장씩 띄우고, 말할 때마다 잘라서 저장한다."""
    from jarviseo.voice.mic import Microphone

    folder, rows = ROOT / name, []
    with Microphone(device) as mic:
        blocks = mic.blocks()
        for i, text in enumerate(QUESTIONS, 1):
            print(f"[{i}/{len(QUESTIONS)}] {text!r} 라고 말하세요")
            speech = None
            while speech is None:
                # 말 중간에 쉬어도 끊기지 않게 길게 기다린다. 짧은 대기는 --silence 로 따로 잰다.
                speech = record_speech(blocks, timeout=10.0, silence_sec=TAIL_SEC)
                if speech is None:
                    print("  말을 못 잡았다. 다시 말하세요")
            file = f"{i:02d}.wav"
            save(folder / file, pad(speech.audio))
            rows.append({"file": file, "text": text, "voice": name})
    write_meta(folder, rows)


def compare(
    folders: list[str], models: list[str], hint: str | None, questions: list[int] | None
) -> None:
    clients = {m: SpeechToText(model=m) for m in models}
    # (이름, 모델, 힌트). 힌트가 비면 transcribe() 가 호출어를 힌트로 준다
    variants = [(m, m, []) for m in models]
    if hint:
        variants += [(f"{m}+힌트", m, [w.strip() for w in hint.split(",")]) for m in models]
    out = []
    for name in folders:
        folder = ROOT / name
        rows = read_meta(folder)
        if questions:
            rows = [r for r in rows if int(Path(r["file"]).stem[-2:]) in questions]
        print(f"\n== {name} ({len(rows)}개)")
        for i, row in enumerate(rows):
            audio = load(folder / row["file"])
            want = letters(row["text"])
            k = i % len(variants)
            for label, model, hints in variants[k:] + variants[:k]:
                asked = time.perf_counter()
                try:
                    got = clients[model].transcribe(audio, 0.0, 0.0, hints).text
                except Exception as e:  # 한 번 실패해도 나머지는 잰다
                    print(f"  {label} {row['file']} 실패: {e}")
                    continue
                ms = (time.perf_counter() - asked) * 1000
                out.append(
                    {
                        "set": name,
                        "file": row["file"],
                        "model": label,
                        "ms": round(ms),
                        "ok": letters(got) == want,
                        "cer": round(cer(want, letters(got)), 3),
                        "want": row["text"],
                        "got": got,
                    }
                )
    report(out, folders, [label for label, _, _ in variants])
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"compare-{datetime.now():%Y%m%d-%H%M%S}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"\n저장 {path}")


def report(out: list[dict], folders: list[str], models: list[str]) -> None:
    print(f"\n{'세트':10} {'모델':24} {'중앙':>6} {'p95':>6} {'맞음':>7} {'CER':>6}")
    for name in folders:
        for model in models:
            rs = [r for r in out if r["set"] == name and r["model"] == model]
            if not rs:
                continue
            ms = sorted(r["ms"] for r in rs)
            p95 = ms[min(len(ms) - 1, round(0.95 * (len(ms) - 1)))]
            ok = sum(r["ok"] for r in rs)
            mean_cer = statistics.mean(r["cer"] for r in rs)
            print(
                f"{name:10} {model:24} {statistics.median(ms):6.0f} {p95:6.0f}"
                f" {ok:3}/{len(rs):<3} {mean_cer:6.3f}"
            )
    print("\n틀린 것")
    for r in out:
        if not r["ok"]:
            print(f"  [{r['set']} {r['model']}] {r['want']!r} → {r['got']!r}")


def silence(folders: list[str], waits: list[float]) -> None:
    """말끝 대기별로, VAD 가 말을 마치기 전에 끊은 녹음 수를 센다(끊은 뒤에 말소리가 또 있으면)."""
    print(f"\n{'세트':10} " + " ".join(f"{w:>5.1f}초" for w in waits))
    for name in folders:
        folder = ROOT / name
        rows = read_meta(folder)
        cuts, cut_files = [], {}
        for wait in waits:
            n = 0
            for row in rows:
                audio = load(folder / row["file"])
                blocks = (
                    (audio[i : i + BLOCK], i / SAMPLE_RATE) for i in range(0, len(audio), BLOCK)
                )
                speech = record_speech(blocks, timeout=5.0, silence_sec=wait)
                rest = audio[round(speech.ended_at * SAMPLE_RATE) :] if speech else audio
                if speech is None or voiced_seconds(rest) >= 0.15:
                    n += 1
                    cut_files.setdefault(wait, []).append(f"{row['file']} {row['text']}")
            cuts.append(n)
        print(f"{name:10} " + " ".join(f"{c:3}/{len(rows):<3}" for c in cuts))
        for wait, files in cut_files.items():
            print(f"  {wait:.1f}초에서 끊김: " + ", ".join(files))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sets", nargs="*", help="비교할 녹음 폴더 이름 (data/datasets/stt/ 아래)")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--tts", action="store_true", help="QUESTIONS 를 TTS 로 합성")
    parser.add_argument("--mix", type=Path, help="tts/ 에 섞을 배경 소리 wav (16kHz 모노)")
    parser.add_argument("--snr", type=float, default=10.0, help="배경 소리를 섞는 SNR(dB)")
    parser.add_argument("--hint", help="받아쓰기 힌트도 따로 잰다 (예: 컵, 병, 왼쪽, 오른쪽)")
    parser.add_argument("--questions", nargs="+", type=int, help="이 QUESTIONS 번호만 (1부터)")
    parser.add_argument("--record", metavar="이름", help="마이크로 QUESTIONS 녹음")
    parser.add_argument("--device", type=int, help="마이크 장치 번호")
    parser.add_argument("--silence", nargs="+", type=float, help="말끝 대기(초)별 끊김만 센다")
    args = parser.parse_args()

    if args.tts:
        make_tts()
    elif args.mix:
        mix(args.mix, args.snr)
    elif args.record:
        record(args.record, args.device)
    elif args.silence:
        silence(args.sets, args.silence)
    else:
        compare(args.sets, args.models, args.hint, args.questions)


if __name__ == "__main__":
    main()
