"""브라우저 호출어가 노트북(파이썬)과 같은 점수를 내는지 잰다 (#218).  담당: 문태현

eval 녹음을 파일마다 두 길로 흘려 80ms 점수를 비교한다.
    파이썬    evaluate.py 와 같다(openWakeWord Model.predict_clip, 앞뒤 무음 1초)
    브라우저  대시보드가 쓰는 src/jarviseo/api/pages/wakeword.js 를
              Node 에서 onnxruntime-web(wasm)으로(browser_scores.mjs)
켤 때 임베딩 버퍼를 채우는 난수(openWakeWord reset)는 파일마다 같은 값을 양쪽에 넣는다.

출력: 프레임 점수 차이(최대 · 중앙), 임계값에서 감지 여부가 다른 파일 수,
      임계값마다 FRR · FAR · FA/h 를 파이썬 / 브라우저 나란히.

준비(한 번): Node 18 이상과 onnxruntime-web. 레포에는 node 패키지를 두지 않는다.
    npm install --prefix <폴더> onnxruntime-web@1.20.1    # dashboard.js ORT_URL 과 같은 버전

실행:
    python scripts/wakeword/compare_browser.py --ort-web <폴더>/node_modules/onnxruntime-web
    python scripts/wakeword/compare_browser.py --ort-web ... --limit 20      # 앞 20개만 빨리
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import evaluate as ev  # noqa: E402

from jarviseo import config  # noqa: E402

HERE = Path(__file__).parent
NOISE_SAMPLES = 16000 * 4  # openWakeWord AudioFeatures.reset 이 만드는 난수 길이


def feature_models() -> dict[str, Path]:
    """openWakeWord 공용 특징 모델 2개(scripts/download_models.py 가 받아 둔 곳)."""
    import openwakeword

    folder = Path(openwakeword.__file__).parent / "resources" / "models"
    return {n: folder / n for n in ("melspectrogram.onnx", "embedding_model.onnx")}


def files_of(folder: Path, limit: int | None) -> list[dict]:
    """evaluate.evaluate 와 같은 파일을 고른다(지운 wav · 잘림 말투는 뺀다)."""
    with (folder / "metadata.csv").open(encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    speech = ev.read_speech(folder)
    picked = [
        m | {"speech": speech.get(m["file"], "")}
        for m in meta
        if (folder / m["file"]).exists() and speech.get(m["file"]) != ev.DROPPED
    ]
    return picked[:limit] if limit else picked


def main() -> None:
    parser = argparse.ArgumentParser(description="브라우저 호출어와 파이썬 점수 비교")
    parser.add_argument("--ort-web", required=True, type=Path, help="node_modules/onnxruntime-web")
    parser.add_argument("--model", default=config.WAKEWORD_MODEL)
    parser.add_argument("--split", choices=["eval", "test"], default="eval")
    parser.add_argument("--limit", type=int, help="앞에서부터 이만큼만")
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(ev.SHOWN))
    args = parser.parse_args()

    folder = ev.ROOT / args.split
    metas = files_of(folder, args.limit)
    model = ev.load_model(args.model)
    tmp = Path(tempfile.mkdtemp(prefix="wakeword-browser-"))
    py: dict[str, np.ndarray] = {}
    jobs = []
    for i, m in enumerate(metas):
        # 같은 시드로 난수를 뽑아 브라우저에 주고, 다시 같은 시드로 파이썬 reset 이
        # 같은 난수를 뽑게 한다
        np.random.seed(i)
        noise = np.random.randint(-1000, 1000, NOISE_SAMPLES).astype(np.int16)
        np.random.seed(i)
        py[m["file"]] = ev.frame_scores(model, folder / m["file"])
        (tmp / f"{i}.noise").write_bytes(noise.tobytes())
        wav = str((folder / m["file"]).resolve())
        jobs.append({"wav": wav, "noise": str(tmp / f"{i}.noise"), "out": str(tmp / f"{i}.out")})
        print(f"\r  파이썬 {i + 1}/{len(metas)}", end="", flush=True)
    print()

    models = {k: str(v) for k, v in feature_models().items()}
    models["jarviseo.onnx"] = str(Path(args.model).resolve())
    ort = args.ort_web.resolve() / "dist" / "ort.wasm.min.mjs"
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({"ort": str(ort), "models": models, "files": jobs}), "utf-8")
    subprocess.run(
        ["node", "--no-warnings", str(HERE / "browser_scores.mjs"), str(manifest)], check=True
    )
    js = {
        m["file"]: np.fromfile(j["out"], dtype=np.float32) for m, j in zip(metas, jobs, strict=True)
    }

    diffs = np.concatenate([np.abs(py[f] - js[f]) for f in py])
    worst = max(py, key=lambda f: float(np.abs(py[f] - js[f]).max()))
    print(f"\n파일 {len(py)}개, 프레임 {len(diffs)}개")
    print(f"프레임 점수 차이  최대 {diffs.max():.6f} ({worst})  중앙 {np.median(diffs):.6f}")

    rows, backs = {}, {}
    for name, scores in (("파이썬", py), ("브라우저", js)):
        rows[name] = [
            {k: m.get(k, "") for k in ev.FIELDS} | {"max_score": f"{scores[m['file']].max():.4f}"}
            for m in metas
        ]
        backs[name] = {m["file"]: scores[m["file"]] for m in metas if m["label"] == "background"}
    for th in args.thresholds:
        differ = sum((py[f].max() >= th) != (js[f].max() >= th) for f in py)
        print(f"\n임계값 {th}  감지 여부가 다른 파일 {differ}개")
        sums = {n: ev.summarize(rows[n], backs[n], [th]) for n in rows}
        for a, b in zip(sums["파이썬"], sums["브라우저"], strict=True):
            if a["source"] != "mic" or a["speech"] or a["mic"]:
                continue
            fmt = "{:.2f}" if a["metric"] == "FA/h" else "{:.1%}"
            print(
                f"  {a['metric']:5} {a['env']:8} n={a['n']}  파이썬 {a['count']}"
                f" ({fmt.format(a['value'])})  브라우저 {b['count']} ({fmt.format(b['value'])})"
            )


if __name__ == "__main__":
    main()
