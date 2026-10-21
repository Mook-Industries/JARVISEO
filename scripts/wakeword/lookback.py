"""호출어에 질문을 이어 말한 소리로 listen(since=감지 시각)을 실제 STT 에 돌려 본다.  담당: 문태현

평가용 실제 "자비서" 녹음(말투 정상) 뒤에 일상 문장 합성(train speech, NORMAL 기본)을
gap 초 띄워 붙이고, 공유 마이크(MicStream)로 흘린다. 감지 시각 t 를 찾아
SpeechToText.listen(since=t) 를 부르고 원문·정리된 질문·말 시작 시각을 찍는다.
config.WAKE_LOOKBACK_SEC 를 바꿔 가며 비교할 수 있다.

    gap 0.05  "자비서오늘…" 처럼 붙여 말함. 앞부터 덜 읽으면 질문 앞부분이 빠지거나("날씨 어때?"),
              호출어 끝만 들어가 "그래서 …"처럼 받아써진다.
    gap 1.0   "자비서." 하고 쉬었다가 질문. 호출어만 받아써지면 listen() 이 한 번 더 듣는다.

WAKE_LOOKBACK_SEC=1.0 의 근거다(docs/experiments.md "끼어들기·턴 음성 기록 (D11)").

--chime 은 시작음(#104)이 마이크에 같이 들어간 소리도 받아써 본다. 감지 CHIME_DELAY_SEC 뒤에
시작음을 섞고, 크기는 질문 말소리보다 몇 dB 작게 할지로 준다(0 이면 같은 크기, 음수면 더 크게).
시작음은 게이트를 닫지 않아서 이렇게 질문과 겹친다. 시작음 없는 결과도 늘 같이 잰다.
섞은 구간은 Speaker 가 하듯 MIC_GATE 에 표시해서 VAD 가 말로 보지 않게 한다(gate.sounding).
--unmarked 를 주면 표시하지 않는다(고치기 전).
OpenAI 받아쓰기를 (n × gap 수 × lookback 수 × 시작음 조건 수)번 넘게 부른다. 키가 필요하다.

실행:
    python scripts/wakeword/lookback.py          # 지금 설정(1초), 붙여·쉬고 4개씩
    python scripts/wakeword/lookback.py --lookback 0 0.4 1.0 --gap 0.05 --n 10
    python scripts/wakeword/lookback.py --n 10 --chime 10 0 -6
    python scripts/wakeword/lookback.py --gap 0.05 --lag 0.9 --lookback 1.0 1.5 --n 6

--lag 는 감지를 모델로 찾지 않고 호출어 끝 + lag 초에 둔다. 실제로 "자비서 이거 뭐야"를 붙여 말하면
감지가 질문 끝까지 늦어서(감지 직전 1.3초에 호출어와 질문이 다 있었다, #149) 그 경우를 재현한다.
"""

import argparse
import csv
import math
import re

import numpy as np
from lag import EVAL, first_detect, read_wav, real_wakes, speech_span
from record import ROOT, SAMPLE_RATE

from jarviseo import config
from jarviseo.voice.chime import START_WAV, load_sound
from jarviseo.voice.gate import MIC_GATE, SOUND_TAIL_SEC
from jarviseo.voice.mic import BLOCK, MicStream
from jarviseo.voice.speaker import SAMPLE_RATE as SPEAKER_RATE
from jarviseo.voice.stt import SpeechToText

TRAIN = ROOT / "train"
LEAD_SEC, TAIL_SEC = 1.5, 1.0  # 호출어 앞 무음, 질문 뒤 무음(말 끝 판정 0.7초보다 길게)
PAD_SEC = 0.03  # 말소리 구간 앞뒤로 남기는 길이. 첫 자음·끝 모음이 잘리지 않게
NOISE = 20  # 실제 마이크처럼 아주 작은 잡음을 깐다(int16 표준편차)
# 호출 감지 → 시작음 첫 소리. 노트북 스피커(MME)에서 스트림 열기 약 0.04초 + 출력 지연 약 0.09초.
CHIME_DELAY_SEC = 0.15


def tight(audio: np.ndarray) -> np.ndarray:
    """앞뒤 무음을 거의 다 잘라 말소리만 남긴다."""
    start, end = speech_span(audio)
    pad = PAD_SEC * SAMPLE_RATE
    return audio[max(0, round(start * SAMPLE_RATE - pad)) : round(end * SAMPLE_RATE + pad)]


def questions() -> list[dict]:
    """일상 문장 합성 중 기본 말투로 읽은 것."""
    with (TRAIN / "metadata.csv").open(encoding="utf-8") as f:
        rows = csv.DictReader(f)
        return [r for r in rows if r["label"] == "speech" and r["mic"].endswith("NORMAL 기본")]


def same(a: str, b: str) -> bool:
    """띄어쓰기·문장부호만 다르면 같은 문장으로 본다."""
    return re.sub(r"[\s.,!?~]", "", a) == re.sub(r"[\s.,!?~]", "", b)


def start_chime() -> np.ndarray:
    """시작음(24kHz)을 마이크와 같은 16kHz 로 바꾼다. 834Hz 소리라 선형 보간으로 충분하다."""
    pcm = np.frombuffer(load_sound(START_WAV), np.int16).astype(np.float64)
    n = round(len(pcm) * SAMPLE_RATE / SPEAKER_RATE)
    return np.interp(np.arange(n) * SPEAKER_RATE / SAMPLE_RATE, np.arange(len(pcm)), pcm)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def with_chime(
    audio: np.ndarray, chime: np.ndarray, at: float, question: np.ndarray, snr: float
) -> np.ndarray:
    """at 초부터 시작음을 섞는다. 시작음 크기는 질문 말소리보다 snr dB 작다."""
    at_sample = round(at * SAMPLE_RATE)
    end = min(len(audio), at_sample + len(chime))
    scaled = chime * rms(question) / rms(chime) / 10 ** (snr / 20)
    mixed = audio.astype(np.float64)
    mixed[at_sample:end] += scaled[: end - at_sample]
    return mixed.clip(-32768, 32767).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description="호출어 + 질문을 listen(since) 로 받아써 보기")
    parser.add_argument("--lookback", type=float, nargs="+", default=[config.WAKE_LOOKBACK_SEC])
    parser.add_argument("--gap", type=float, nargs="+", default=[0.05, 1.0])
    parser.add_argument("--n", type=int, default=4, help="gap 마다 몇 쌍")
    parser.add_argument("--start", type=int, default=0, help="몇 번째 녹음·문장부터 쓸지")
    parser.add_argument("--threshold", type=float, default=config.WAKEWORD_THRESHOLD)
    parser.add_argument(
        "--lag",
        type=float,
        help="감지 시각을 모델로 찾지 않고 호출어 끝 + 이 초로 둔다 (붙여 말할 때 늦은 감지, #149)",
    )
    parser.add_argument(
        "--chime",
        type=float,
        nargs="+",
        default=[],
        help="시작음을 질문보다 몇 dB 작게 섞을지 (여러 개, 시작음 없는 것은 늘 잰다)",
    )
    parser.add_argument(
        "--unmarked", action="store_true", help="시작음 구간을 VAD 에 알리지 않는다 (고치기 전)"
    )
    args = parser.parse_args()

    wakes, texts = real_wakes("정상"), questions()
    chime = start_chime() if args.chime else None
    chimes: list[float | None] = [None, *args.chime]
    rng = np.random.default_rng(0)
    right: dict[tuple[float, float, float | None], list[bool]] = {}
    k = args.start
    for gap in args.gap:
        for _ in range(args.n):
            wake, row = tight(read_wav(EVAL / wakes[k]["file"])), texts[k]
            k += 1
            question = tight(read_wav(TRAIN / row["file"]))
            parts = [LEAD_SEC, wake, gap, question, TAIL_SEC]
            audio = np.concatenate(
                [
                    p if isinstance(p, np.ndarray) else np.zeros(round(p * SAMPLE_RATE), np.int16)
                    for p in parts
                ]
            )
            noise = rng.normal(0, NOISE, len(audio))
            audio = (audio + noise).clip(-32768, 32767).astype(np.int16)
            asked_at = LEAD_SEC + (len(wake) + round(gap * SAMPLE_RATE)) / SAMPLE_RATE
            if args.lag is None:
                t = first_detect(audio, args.threshold, config.WAKEWORD_MODEL)
            else:
                t = LEAD_SEC + len(wake) / SAMPLE_RATE + args.lag
            print(f"\n쉼 {gap}s, 질문 {row['text']!r}")
            if t is None:
                print("  호출어 감지 못 함")
                continue
            for snr in chimes:
                heard = audio
                MIC_GATE.mark_sound(-math.inf, -math.inf)
                if snr is not None:
                    at = t + CHIME_DELAY_SEC
                    heard = with_chime(audio, chime, at, question, snr)
                    if not args.unmarked:
                        MIC_GATE.mark_sound(at, at + len(chime) / SAMPLE_RATE + SOUND_TAIL_SEC)
                blocks = [
                    (heard[i : i + BLOCK], i / SAMPLE_RATE)
                    for i in range(0, len(heard) - BLOCK + 1, BLOCK)
                ]
                for lookback in args.lookback:
                    config.WAKE_LOOKBACK_SEC = lookback  # listen() 이 부를 때마다 읽는다
                    with MicStream(source=blocks, history_sec=len(heard) / SAMPLE_RATE + 1) as mic:
                        stt = SpeechToText(mic=mic)
                        utterance = stt.listen(since=t, timeout=2.0)
                    ok = utterance is not None and same(utterance.text, row["text"])
                    right.setdefault((gap, lookback, snr), []).append(ok)
                    got = "None" if utterance is None else repr(utterance.text)
                    when = (
                        ""
                        if utterance is None
                        else f", 말 시작 {utterance.started_at - asked_at:+.2f}s"
                    )
                    label = "" if snr is None else f" 시작음 {snr:+g}dB"
                    print(f"  앞 {lookback}s{label}  원문 {stt.last_raw_text!r} → {got}{when}")

    print("\n글자가 같은 것 (띄어쓰기·문장부호 무시, STT 가 비슷한 말로 받아쓴 것은 틀림으로 셈)")
    for (gap, lookback, snr), oks in sorted(right.items(), key=lambda kv: str(kv[0])):
        label = "시작음 없음" if snr is None else f"시작음 {snr:+g}dB"
        print(f"  쉼 {gap}s, 앞 {lookback}s, {label}: {sum(oks)}/{len(oks)}")


if __name__ == "__main__":
    main()
