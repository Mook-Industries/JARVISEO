"""호출어 2단계 확인(받아쓰기)을 녹음 파일로 잰다.  담당: 문태현

1차(호출어 모델)가 깨어난 시각 t 마다 [t − 구간, t] 를 잘라 voice/wakecheck.py 와 같은 방법으로
받아쓰고, 호출어가 받아써졌는지(transcript.has_wake_word) 센다(#149).

    진짜 호출  eval 실제 "자비서"(잘림 제외)에 앞뒤 2초 잡음을 붙여 흘리고, 지금 임계값에서
               깬 것만 본다. 2차에서 떨어지면 진짜 호출을 놓친 것이다.
    오탐 후보  eval TV 30분·일반 대화 30분, train 방송 1시간에서 깬 곳.
               2차를 통과하면 최종 오탐이다.
               지금 임계값(0.97)에서는 몇 개뿐이라 0.5 에서도 뽑아 표본을 늘린다.
               train 방송은 1차 학습에 오답으로 들어갔지만 2차는 학습이 없어서 그대로 쓴다.

구간 길이(--sec)와 받아쓰기 힌트 있음·없음(--hint)을 바꿔 가며 비교한다. OpenAI 받아쓰기를
(진짜 호출 + 오탐 후보) × 구간 수 × 힌트 수 번 부른다. 키가 필요하다.

실행:
    python scripts/wakeword/verify.py                      # 1.0·1.3·1.6초 × 힌트 있음·없음
    python scripts/wakeword/verify.py --sec 1.3 --hint on --verbose
    python scripts/wakeword/verify.py --wake-only --sec 1.3 --hint on --workers 1   # 왕복 시간
    python scripts/wakeword/verify.py --rescore data/datasets/wakeword/results/verify-….csv

--split test 는 고를 때 안 쓴 최종 평가용 녹음(record.py --split test)만 읽는다. 진짜 호출과
그 폴더의 background 를 같은 방법으로 잰다. 구간·힌트는 정한 값(1.6초, 힌트 있음)으로 고정해서 준다.
    python scripts/wakeword/verify.py --split test --sec 1.6 --hint on --verbose

결과: data/datasets/wakeword/results/verify-<날짜시각>.csv  (조각마다 받아쓴 글자와 통과 여부)
호출어로 칠 표기(transcript.has_wake_word)를 바꾸면 --rescore 로 받아쓰기 없이 다시 센다.
"""

import argparse
import csv
import math
import statistics
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from lag import EVAL, read_wav, real_wakes
from openai import OpenAI
from record import ROOT, SAMPLE_RATE

from jarviseo import config
from jarviseo.voice.transcript import has_wake_word
from jarviseo.voice.wakecheck import WakeCheck
from jarviseo.voice.wakeword import FRAME, REFRACTORY_SEC, load_scorer

TRAIN = ROOT / "train"
RESULTS = ROOT / "results"
PAD_SEC = 2.0  # 짧은 녹음 앞뒤에 붙이는 길이. 모델이 소리 없이 시작·끝나지 않게
NOISE = 20  # 붙이는 구간에 까는 아주 작은 잡음(int16 표준편차). 완전 무음은 받아쓰기가 지어낸다
FIELDS = "kind file env speech threshold t score sec hint text ms passed".split()


def frame_scores(audio: np.ndarray, model: str) -> np.ndarray:
    """감지기처럼 80ms 씩 넣은 점수. 파일마다 모델을 새로 열어 앞 파일 소리가 섞이지 않게 한다."""
    score = load_scorer(model)
    return np.array([score(audio[i : i + FRAME]) for i in range(0, len(audio) - FRAME + 1, FRAME)])


def detections(scores: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    """깨어난 (시각(초), 점수). 시각은 프레임이 끝난 때이고, 깨면 REFRACTORY_SEC 는 다시 안 깬다."""
    found, quiet_until = [], -math.inf
    for i in np.flatnonzero(scores >= threshold):
        t = (i + 1) * FRAME / SAMPLE_RATE
        if t >= quiet_until:
            found.append((t, float(scores[i])))
            quiet_until = t + REFRACTORY_SEC
    return found


def backgrounds(split: str = "eval") -> list[tuple[str, dict]]:
    """(종류, metadata 행). eval 이면 eval 의 TV·대화와 train 의 방송, test 면 test 폴더 것만."""
    rows = []
    folders = ((EVAL, "eval"), (TRAIN, "train")) if split == "eval" else ((ROOT / split, split),)
    for folder, name in folders:
        if not (folder / "metadata.csv").exists():
            continue
        with (folder / "metadata.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["label"] == "background" and (folder / r["file"]).exists():
                    rows.append((f"{name} {r['env']}", {**r, "path": folder / r["file"]}))
    return rows


def candidates(
    model: str, thresholds: list[float], background: bool = True, split: str = "eval"
) -> tuple[list[dict], dict[str, float]]:
    """1차가 깨어난 곳들과, 오탐 후보 녹음 종류별 길이(시간). background 가 False 면 진짜 호출만."""
    rng = np.random.default_rng(0)
    found = []
    wakes = real_wakes(folder=ROOT / split)
    for i, r in enumerate(wakes, 1):
        audio = read_wav(ROOT / split / r["file"])
        pad = round(PAD_SEC * SAMPLE_RATE)
        audio = np.concatenate([np.zeros(pad, np.int16), audio, np.zeros(pad, np.int16)])
        audio[:pad] = rng.normal(0, NOISE, pad).astype(np.int16)
        audio[-pad:] = rng.normal(0, NOISE, pad).astype(np.int16)
        hits = detections(frame_scores(audio, model), config.WAKEWORD_THRESHOLD)
        row = {"kind": "wake", "file": r["file"], "env": r["env"], "speech": r["speech"]}
        if hits:
            t, score = hits[0]
            found.append(row | {"threshold": config.WAKEWORD_THRESHOLD, "t": t, "score": score})
            found[-1]["audio"] = audio
        else:
            found.append(row | {"threshold": config.WAKEWORD_THRESHOLD, "t": None, "score": None})
        print(f"\r  진짜 호출 {i}/{len(wakes)}", end="", flush=True)
    print()
    hours: dict[str, float] = {}
    for kind, r in backgrounds(split) if background else []:
        audio = read_wav(r["path"])
        hours[kind] = hours.get(kind, 0.0) + len(audio) / SAMPLE_RATE / 3600
        print(f"  {kind} {r['file']} ({len(audio) / SAMPLE_RATE / 60:.0f}분) 점수 내는 중")
        scores = frame_scores(audio, model)
        for th in thresholds:
            for t, score in detections(scores, th):
                found.append(
                    {"kind": kind, "file": r["file"], "env": r["env"], "speech": ""}
                    | {"threshold": th, "t": t, "score": score, "audio": audio}
                )
    return found, hours


def transcribe_all(
    found: list[dict], secs: list[float], hints: list[str], workers: int
) -> list[dict]:
    """깬 곳마다 구간 길이 × 힌트별로 받아쓴 행. 왕복 시간(ms)도 남긴다."""
    # 재는 동안에는 한 번 더 보내 본다. 실패가 섞이면 2차의 성능이 아니라 인터넷을 재게 된다.
    check = WakeCheck(client=OpenAI(api_key=config.OPENAI_API_KEY or None, max_retries=2))
    jobs = []
    for c in found:
        if c["t"] is None:
            continue
        for sec in secs:
            for hint in hints:
                start = max(0, round((c["t"] - sec) * SAMPLE_RATE))
                jobs.append((c, sec, hint, c["audio"][start : round(c["t"] * SAMPLE_RATE)]))
    print(f"받아쓰기 {len(jobs)}번")

    def timed(job):
        sent_at = time.perf_counter()
        text = check.transcribe(job[3], hint=job[2] == "on")
        return text, (time.perf_counter() - sent_at) * 1000

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(timed, jobs))
    return [
        {k: c[k] for k in FIELDS[:5]}
        | {"t": c["t"], "score": c["score"], "sec": sec, "hint": hint, "text": text, "ms": ms}
        for (c, sec, hint, _), (text, ms) in zip(jobs, results, strict=True)
    ]


def rescore(path: Path, split: str = "eval") -> tuple[list[dict], dict[str, float]]:
    """저장한 CSV 의 받아쓴 글자를 다시 읽는다. 받아쓰기는 부르지 않는다. 오탐 후보 녹음 길이는
    wav 머리만 읽어 다시 센다."""
    with path.open(encoding="utf-8") as f:
        rows = [
            r | {"threshold": float(r["threshold"]), "t": float(r["t"]), "sec": float(r["sec"])}
            for r in csv.DictReader(f)
        ]
    hours: dict[str, float] = {}
    for kind, r in backgrounds(split):
        with wave.open(str(r["path"]), "rb") as w:
            hours[kind] = hours.get(kind, 0.0) + w.getnframes() / SAMPLE_RATE / 3600
    return rows, hours


def summarize(rows: list[dict], hours: dict[str, float], n_wakes: int, verbose: bool) -> None:
    """2차 통과 수를 진짜 호출(환경별)과 오탐 후보(녹음·임계값별)로 찍는다.

    통과 여부는 저장한 값이 아니라 지금 has_wake_word 로 다시 정한다(규칙을 바꾸면 --rescore).
    """
    for r in rows:
        r["passed"] = int(has_wake_word(r["text"]))
    secs = sorted({r["sec"] for r in rows})
    hints = sorted({r["hint"] for r in rows}, reverse=True)  # on 먼저
    conditions = [(sec, hint) for sec in secs for hint in hints]

    def picked(kind_ok, sec, hint, **match):
        return [
            r
            for r in rows
            if kind_ok(r["kind"])
            and r["sec"] == sec
            and r["hint"] == hint
            and all(r[k] == v for k, v in match.items())
        ]

    is_wake = "wake".__eq__
    woke = {r["file"] for r in rows if r["kind"] == "wake"}
    th = config.WAKEWORD_THRESHOLD
    print(f"\n진짜 호출 {n_wakes}개 중 1차({th})에서 깬 것 {len(woke)}개. 2차 통과 (깬 것 중)")
    for sec, hint in conditions:
        line = []
        for env in ("quiet", "noisy"):
            got = picked(is_wake, sec, hint, env=env)
            line.append(f"{env} {sum(r['passed'] for r in got)}/{len(got)}")
        got = picked(is_wake, sec, hint)
        line.append(f"전체 {sum(r['passed'] for r in got)}/{len(got)}")
        times = [float(r["ms"]) for r in got if r.get("ms")]
        if len(times) >= 2:
            p90 = statistics.quantiles(times, n=10)[-1]
            line.append(f"왕복 중앙 {statistics.median(times):.0f} · p90 {p90:.0f}ms")
        print(f"  {sec}초 힌트 {hint}: " + ", ".join(line))

    print("\n오탐 후보: 1차에서 깬 수 → 2차까지 통과한 수 (시간당)")
    thresholds = sorted({r["threshold"] for r in rows if r["kind"] != "wake"}, reverse=True)
    for kind, h in hours.items():
        for th in thresholds:
            first = {r["t"] for r in rows if r["kind"] == kind and r["threshold"] == th}
            print(f"  {kind} {h * 60:.0f}분 @{th}: 1차 {len(first)}회 ({len(first) / h:.1f}/h)")
            for sec, hint in conditions:
                n = sum(r["passed"] for r in picked(kind.__eq__, sec, hint, threshold=th))
                print(f"      {sec}초 힌트 {hint}: 2차 통과 {n}회 ({n / h:.1f}/h)")

    if verbose:
        print("\n2차에서 떨어진 진짜 호출")
        for r in rows:
            if r["kind"] == "wake" and not r["passed"]:
                what = f"{r['file']} {r['env']} {r['speech']}"
                print(f"  {what} {r['sec']}초 {r['hint']} {r['text']!r}")
        print("오탐 후보에서 받아쓴 글자 (통과 표시 ✓)")
        for r in rows:
            if r["kind"] != "wake":
                where = f"{r['kind']} @{r['threshold']} {r['t'] / 60:.1f}분"
                mark = "✓" if r["passed"] else " "
                print(f"  {mark} {where} {r['sec']}초 {r['hint']} {r['text']!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="호출어 2단계 확인을 녹음 파일로 잰다")
    parser.add_argument("--model", default=config.WAKEWORD_MODEL)
    parser.add_argument("--sec", type=float, nargs="+", default=[1.0, 1.3, 1.6])
    parser.add_argument("--hint", choices=["on", "off"], nargs="+", default=["on", "off"])
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[config.WAKEWORD_THRESHOLD, 0.5]
    )
    parser.add_argument(
        "--wake-only", action="store_true", help="진짜 호출만 잰다 (오탐 후보 점수 내기를 건너뜀)"
    )
    parser.add_argument(
        "--workers", type=int, default=6, help="받아쓰기를 같이 보낼 수 (왕복 시간을 잴 때는 1)"
    )
    parser.add_argument(
        "--split", choices=["eval", "test"], default="eval", help="test = 고를 때 안 쓴 녹음"
    )
    parser.add_argument(
        "--rescore", type=Path, help="받아쓰기 없이 저장한 CSV 를 지금 규칙으로 다시 센다"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="떨어진 진짜 호출·오탐 후보 글자를 찍는다"
    )
    args = parser.parse_args()

    n_wakes = len(real_wakes(folder=ROOT / args.split))
    if args.rescore is not None:
        rows, hours = rescore(args.rescore, args.split)
        summarize(rows, hours, n_wakes, args.verbose)
        return
    found, hours = candidates(
        args.model, args.thresholds, background=not args.wake_only, split=args.split
    )
    rows = transcribe_all(found, args.sec, args.hint, args.workers)
    RESULTS.mkdir(parents=True, exist_ok=True)
    split = "" if args.split == "eval" else f"{args.split}-"
    out = RESULTS / f"verify-{split}{datetime.now():%Y%m%d-%H%M%S}.csv"
    summarize(rows, hours, n_wakes, args.verbose)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(
            r | {"t": f"{r['t']:.2f}", "score": f"{r['score']:.4f}", "ms": f"{r['ms']:.0f}"}
            for r in rows
        )
    print(f"\n결과: {out}")


if __name__ == "__main__":
    main()
