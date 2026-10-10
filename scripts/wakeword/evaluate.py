"""웨이크워드 모델 평가.  담당: 문태현

eval 폴더의 녹음(record.py 로 만든 것)을 openWakeWord 모델에 흘려 파일마다 점수를 낸다.
모델은 80ms 마다 0~1 점수를 낸다. 짧은 녹음(wake·similar)은 앞뒤에 무음 1초를 붙여 흘리고
가장 높은 점수를 남긴다. 긴 녹음(background)은 프레임 점수를 그대로 들고 있는다.

평가용 녹음만 읽는다. 학습용(train)은 여기서 읽지 않는다.

임계값 0.1~0.9 마다 세 가지를 센다. 실제 녹음(mic)과 합성(tts), 환경(env)별로 따로 낸다.
    FRR    wake 녹음 중 점수가 임계값에 못 미친 비율 (불렀는데 안 깨어남)
    FAR    similar 녹음 중 점수가 임계값을 넘은 비율 (비슷한 말에 깨어남)
    FA/h   background 녹음에서 깨어난 횟수 ÷ 시간. 한 번 깨어나면 2초는 다시 세지 않는다

eval/speech.csv 가 있으면 실제 "자비서" 녹음을 사람이 들어 보고 적은 말투로 나눈다.
    정상  한 번에 이어 말함        끊음  "자, 비서"처럼 중간에 쉼
    늘임  "자~비~서"처럼 길게      잘림  일부만 녹음됨 → 녹음 오류라 평가에서 뺀다
끊음·늘임도 호출어로 친다(이슈 #40). FRR 은 전체와 말투별로 따로 낸다.
마지막에 합격 기준을 한 임계값에서 모두 맞춘 임계값을 출력한다.

실행:
    python scripts/wakeword/evaluate.py                                   # 기성 hey_jarvis
    python scripts/wakeword/evaluate.py --model data/models/jarviseo.onnx # 커스텀 학습 뒤

결과: data/datasets/wakeword/results/ 에
    <모델>-<날짜시각>-scores.csv    파일별 최고 점수
    <모델>-<날짜시각>-summary.csv   임계값별 FRR·FAR·FA/h
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
FIELDS = "file label source env speaker distance text speech seconds max_score".split()
SUMMARY = ["threshold", "metric", "source", "env", "speech", "n", "count", "value"]
THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]
SHOWN = (0.5, 0.8)  # 기본 임계값, 재생 중(소프트 게이트) 임계값
REFRACTORY_SEC = 2.0  # 한 번 깨어나면 이만큼은 다시 세지 않는다
# 합격 기준(이슈 #40). 실제 녹음으로 잰 값이 같은 임계값 하나에서 모두 이 안이어야 한다.
# TV 는 30분뿐이라 한 번만 깨어나도 2회/h 가 되므로 0회를 기준으로 둔다.
CRITERIA = {
    ("FRR", "quiet"): 0.10,
    ("FRR", "noisy"): 0.20,
    ("FAR", "quiet"): 0.10,
    ("FA/h", "tv"): 0.0,
}
DROPPED = "잘림"  # speech.csv 에서 이 말투는 녹음 오류로 보고 평가에서 뺀다


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


def read_speech(folder: Path) -> dict[str, str]:
    """speech.csv(file, speech)를 읽어 파일별 말투를 돌려준다. 없으면 빈 dict."""
    path = folder / "speech.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["file"]: r["speech"] for r in csv.DictReader(f)}


def evaluate(model, folder: Path) -> tuple[list[dict], dict[str, np.ndarray]]:
    """metadata.csv 의 파일을 모두 흘린다. (파일별 행, background 파일별 프레임 점수) 를 돌려준다.

    녹음이 잘못돼 wav 를 지웠거나 speech.csv 에 잘림으로 적힌 파일은 건너뛴다.
    """
    with (folder / "metadata.csv").open(encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    speech = read_speech(folder)
    rows, backgrounds, dropped = [], {}, 0
    for i, m in enumerate(meta, 1):
        path = folder / m["file"]
        if not path.exists():
            continue
        if speech.get(m["file"]) == DROPPED:
            dropped += 1
            continue
        scores = frame_scores(model, path)
        if m["label"] == "background":
            backgrounds[m["file"]] = scores
        rows.append(
            {k: m.get(k, "") for k in FIELDS}
            | {"speech": speech.get(m["file"], ""), "max_score": f"{scores.max():.4f}"}
        )
        print(f"\r  {i}/{len(meta)} {m['file']}", end="", flush=True)
    print()
    if dropped:
        print(f"  speech.csv 에 {DROPPED}으로 적힌 {dropped}개는 뺐다")
    return rows, backgrounds


def wakeups(scores: np.ndarray, threshold: float) -> int:
    """프레임 점수에서 깨어난 횟수. 한 번 깨어나면 REFRACTORY_SEC 동안은 다시 세지 않는다."""
    gap = round(REFRACTORY_SEC * SAMPLE_RATE / FRAME)
    count, until = 0, -1
    for i in np.flatnonzero(scores >= threshold):
        if i >= until:
            count, until = count + 1, i + gap
    return count


def summarize(rows: list[dict], backgrounds: dict[str, np.ndarray]) -> list[dict]:
    """임계값마다 FRR·FAR·FA/h 를 출처(mic/tts)·환경(env)별로 센다.

    말투가 적힌 "자비서" 녹음은 말투별 FRR 도 따로 센다(speech 칸). 전체 행은 speech 가 비어 있다.
    """
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["label"], r["source"], r["env"], ""), []).append(r)
        if r["label"] == "wake" and r["speech"]:
            groups.setdefault((r["label"], r["source"], r["env"], r["speech"]), []).append(r)
    out = []
    for th in THRESHOLDS:
        for (label, source, env, speech), rs in sorted(groups.items()):
            if label == "background":
                hours = sum(len(backgrounds[r["file"]]) for r in rs) * FRAME / SAMPLE_RATE / 3600
                count = sum(wakeups(backgrounds[r["file"]], th) for r in rs)
                metric, n, value = "FA/h", round(hours, 3), count / hours if hours else 0.0
            else:
                over = sum(float(r["max_score"]) >= th for r in rs)
                # wake 는 못 넘은 것이 실패(FRR), similar 는 넘은 것이 실패(FAR)
                metric, count = ("FRR", len(rs) - over) if label == "wake" else ("FAR", over)
                n, value = len(rs), count / len(rs)
            out.append(
                {"threshold": th, "metric": metric, "source": source, "env": env, "speech": speech}
                | {"n": n, "count": count, "value": round(value, 4)}
            )
    return out


def passing(summary: list[dict]) -> list[float]:
    """합격 기준(CRITERIA)을 실제 녹음 전체 행에서 모두 맞춘 임계값들."""
    ok = []
    for th in THRESHOLDS:
        values = {
            (s["metric"], s["env"]): s["value"]
            for s in summary
            if s["threshold"] == th and s["source"] == "mic" and not s["speech"]
        }
        if all(values.get(key, float("inf")) <= limit for key, limit in CRITERIA.items()):
            ok.append(th)
    return ok


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="웨이크워드 평가")
    parser.add_argument("--model", default="hey_jarvis", help="기성 모델 이름이나 .onnx 경로")
    args = parser.parse_args()

    np.random.seed(0)  # model.reset() 이 버퍼를 난수로 채운다. 다시 돌려도 같은 점수가 나오게
    model = load_model(args.model)
    rows, backgrounds = evaluate(model, ROOT / "eval")
    summary = summarize(rows, backgrounds)

    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"{Path(args.model).stem}-{datetime.now():%Y%m%d-%H%M}"
    write_csv(out / f"{stamp}-scores.csv", FIELDS, rows)
    write_csv(out / f"{stamp}-summary.csv", SUMMARY, summary)

    for th in SHOWN:
        print(f"\n임계값 {th}")
        for s in summary:
            if s["threshold"] != th:
                continue
            if s["metric"] == "FA/h":
                where, value = f"{s['n']}시간 중 {s['count']}번", f"{s['value']:.2f}"
            else:
                where, value = f"{s['n']}개 중 {s['count']}개", f"{s['value']:.1%}"
            group = f"{s['source']}·{s['env']}" + (f"  └ {s['speech']}" if s["speech"] else "")
            print(f"  {s['metric']:5} {group}  {where}  {value}")
    ok = passing(summary)
    print(f"\n합격 기준을 모두 맞춘 임계값: {', '.join(map(str, ok)) if ok else '없음'}")
    print(f"결과 {out / stamp}-scores.csv, -summary.csv")


if __name__ == "__main__":
    main()
