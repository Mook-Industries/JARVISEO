"""호출어 감지가 "자비서" 끝보다 얼마나 늦는지 잰다.  담당: 문태현

평가용 실제 녹음(eval, 잘림 제외)을 앞뒤 무음 2초를 붙여 감지기처럼 80ms 씩 모델에 넣고,
점수가 처음 임계값을 넘은 프레임이 끝난 시각(= on_detect 의 t)과 말소리가 끝난 시각을 비교한다.
말소리 구간은 prepare.trim 과 같은 기준(가장 큰 곳 -30dB, 바닥 소음 +10dB)으로 잡는다.

listen(since=t) 를 감지 시각부터 읽으면 이만큼 늦은 사이에 시작한 질문 앞부분이 빠진다.
config.WAKE_LOOKBACK_SEC(얼마나 앞부터 읽을지)와 stt.WAKE_LAG_SEC(말 시작 시각 하한)의 근거다
(docs/experiments.md "끼어들기·턴 음성 기록 (D11)").

실행:
    python scripts/wakeword/lag.py
    python scripts/wakeword/lag.py --threshold 0.99 --verbose

소음 속 녹음은 말 끝이 소음까지 늘어나 크게 음수(감지가 말 끝보다 이름)로 나오는 것이 섞인다.
중앙값과 최대값을 본다.
"""

import argparse
import csv
import statistics
import wave
from pathlib import Path

import numpy as np
from record import ROOT, SAMPLE_RATE

from jarviseo import config
from jarviseo.voice.wakeword import FRAME, load_scorer

EVAL = ROOT / "eval"
LEVEL_BLOCK = 480  # 30ms 씩 소리 크기를 잰다(prepare.py 와 같음)
PAD_SEC = 2.0  # 모델이 앞뒤 소리 없이 시작·끝나지 않게 붙이는 무음


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        assert f.getframerate() == SAMPLE_RATE and f.getnchannels() == 1, path
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")


def speech_span(audio: np.ndarray) -> tuple[float, float]:
    """말소리가 시작·끝나는 시각(초). prepare.trim 과 같은 기준이다."""
    n = len(audio) // LEVEL_BLOCK
    blocks = audio[: n * LEVEL_BLOCK].astype(np.float32).reshape(n, LEVEL_BLOCK)
    db = 20 * np.log10(np.sqrt((blocks**2).mean(axis=1)) + 1e-6)
    loud = np.flatnonzero(db > max(db.max() - 30, np.percentile(db, 10) + 10))
    return loud[0] * LEVEL_BLOCK / SAMPLE_RATE, (loud[-1] + 1) * LEVEL_BLOCK / SAMPLE_RATE


def real_wakes(speech: str | None = None, folder: Path = EVAL) -> list[dict]:
    """평가용 실제 "자비서" 녹음 행. speech 를 주면 그 말투(정상·끊음·늘임)만, 잘림은 늘 뺀다.

    folder 는 eval 이나 최종 평가용 test 폴더다. speech.csv(말투)가 없으면 말투는 "?" 다.
    """
    kinds = {}
    if (folder / "speech.csv").exists():
        with (folder / "speech.csv").open(encoding="utf-8") as f:
            kinds = {r["file"]: r["speech"] for r in csv.DictReader(f)}
    with (folder / "metadata.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] == "wake" and r["source"] != "tts"]
    rows = [
        {**r, "speech": kinds.get(r["file"], "?")}
        for r in rows
        if (folder / r["file"]).exists() and kinds.get(r["file"]) != "잘림"
    ]
    return [r for r in rows if speech is None or r["speech"] == speech]


def first_detect(audio: np.ndarray, threshold: float, model: str) -> float | None:
    """감지기처럼 80ms 씩 넣어, 처음 임계값을 넘은 프레임이 끝난 시각(초). 못 넘으면 None."""
    score = load_scorer(model)  # 파일마다 새로 연다. 앞 파일 소리가 섞이지 않게
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        if score(audio[i : i + FRAME]) >= threshold:
            return (i + FRAME) / SAMPLE_RATE
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="호출어 감지가 말 끝보다 늦는 시간")
    parser.add_argument("--model", default=config.WAKEWORD_MODEL)
    parser.add_argument("--threshold", type=float, default=config.WAKEWORD_THRESHOLD)
    parser.add_argument("--verbose", action="store_true", help="파일마다 찍는다")
    args = parser.parse_args()

    pad = np.zeros(round(PAD_SEC * SAMPLE_RATE), np.int16)
    lags: dict[str, list[float]] = {}
    rows = real_wakes()
    for r in rows:
        audio = read_wav(EVAL / r["file"])
        start, end = (PAD_SEC + s for s in speech_span(audio))
        t = first_detect(np.concatenate([pad, audio, pad]), args.threshold, args.model)
        if t is not None:
            lags.setdefault(r["env"], []).append(t - end)
        if args.verbose:
            shown = "못 넘음" if t is None else f"{t - end:+.2f}s"
            print(f"  {r['file']} {r['env']} {r['speech']} 말 {end - start:.2f}s 감지-말끝 {shown}")

    every = [lag for env in lags.values() for lag in env]
    print(f"실제 녹음 {len(rows)}개 중 {args.threshold} 에서 감지 {len(every)}개")
    if len(every) < 2:
        return
    deciles = statistics.quantiles(every, n=10)
    print("감지 - 말 끝 (초)  최소 / p10 / 중앙 / p90 / 최대")
    values = (min(every), deciles[0], statistics.median(every), deciles[-1], max(every))
    print("  전체  " + " / ".join(f"{v:+.2f}" for v in values))
    for env, values in sorted(lags.items()):
        middle, top = statistics.median(values), max(values)
        print(f"  {env}  중앙 {middle:+.2f}, 최대 {top:+.2f} ({len(values)}개)")
    lengths = [b - a for a, b in (speech_span(read_wav(EVAL / r["file"])) for r in rows)]
    print(f"말 길이 중앙 {statistics.median(lengths):.2f}s")


if __name__ == "__main__":
    main()
