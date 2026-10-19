"""가리킴 ablation — 정답 표시한 사진으로 v1~v5 를 채점하고 기록한다.  담당: 최홍묵

tools/label_targets.py 가 만든 labels.csv 의 사진마다 한 번 검출하고, 버전마다 켤 단서만 켜서
resolve_target 으로 고른 박스를 정답 박스와 비교한다(IoU 0.5 이상이면 맞음).
결과는 DB eval_run(버전마다 한 줄, 지표 묶음) · eval_sample(사진마다 맞음/틀림)과
사진별 CSV(실패 분류 · margin · 지연)에 남기고, 이전 실행과 비교표를 찍는다.

버전은 단서를 하나씩 쌓아 켠다(types.CueKind 순서). 아직 구현 안 된 단서가 든 버전은 건너뛴다.
검출은 사진마다 한 번만 해서, 버전 차이가 단서에서만 나오게 한다.
STT · VLM 없이 촬영 때 적은 발화로 채점해서, 가리킴 모델 실력만 따로 잰다.

실행 (레포 맨 위 폴더에서):
    python scripts/run_ablation.py --note "v1 기준선"
    python scripts/run_ablation.py --variants v1 v2 --note "v2 손 방향 추가"
    python scripts/run_ablation.py --split test --note "최종"     # 10/24 최종 측정 때만
"""

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select

from jarviseo import config
from jarviseo.memory import EvalRun, EvalSample, MemoryStore
from jarviseo.pointing.resolver import IMPLEMENTED_CUES, resolve_target
from jarviseo.types import BBox, CueKind, Detection, TargetResolution

POINTING_DIR = config.DATASETS_DIR / "pointing"
# 단서를 하나씩 쌓아 켠다. ablation 표의 한 줄이 한 버전이다.
VARIANTS: dict[str, frozenset[CueKind]] = {
    "v1": frozenset({CueKind.CENTER}),
    "v2": frozenset({CueKind.CENTER, CueKind.POINT}),
    "v3": frozenset({CueKind.CENTER, CueKind.POINT, CueKind.GAZE}),
    "v4": frozenset({CueKind.CENTER, CueKind.POINT, CueKind.GAZE, CueKind.LANG}),
    "v5": frozenset(CueKind),
}
# 검출 평가의 표준 기준. 고른 박스와 정답 박스가 절반 넘게 겹치면 맞음.
IOU_MATCH = 0.5
# 틀렸는데 margin 이 이 이상이면 "높은 확신으로 틀림". 되묻지 않고 단정한 오답이다.
HIGH_CONFIDENCE = 0.5
# 실패 6분류(멘토링 2차) + 정답. 위에서 먼저 걸리는 것으로 정한다(judge).
OUTCOMES = (
    "정답",
    "되묻는 게 맞음",
    "후보 없음",
    "불필요한 되묻기",
    "방향 오판",
    "높은 확신으로 틀림",
    "다른 후보 1위",
)
CSV_COLUMNS = [
    "file",
    "gt_label",
    "pred_label",
    "outcome",
    "correct",
    "margin",
    "has_hand",
    "ambiguous",
    "candidate_hit",
    "detect_ms",
    "resolve_ms",
]


def load_samples(path: Path, split: str) -> list[dict]:
    """labels.csv 에서 그 split 의 쓸 수 있는 사진만 읽어, 정답 박스 · 표시를 풀어 둔다."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    samples = []
    for row in rows:
        if row["split"] != split or row["excluded"] == "1":
            continue
        samples.append(
            {
                "file": row["file"],
                "utterance": row["utterance"],
                "gt": BBox(*(float(row[k]) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2"))),
                "gt_label": row["gt_label"],
                # 0/1 로 저장했다. bool("0") 은 참이라 글자로 비교한다.
                "has_hand": row["has_hand"] == "1",
                "ambiguous": row["ambiguous"] == "1",
                "prev_label": row["prev_label"] or None,
            }
        )
    return samples


def implemented_variants(names: list[str]) -> tuple[list[str], list[str]]:
    """(돌릴 버전, 미구현이라 건너뛸 버전). 구현 안 된 단서를 켜면 resolve_target 이 멈춘다."""
    ready = [v for v in names if VARIANTS[v] <= IMPLEMENTED_CUES]
    return ready, [v for v in names if v not in ready]


def judge(
    sample: dict, resolution: TargetResolution, candidate_hit: bool, uses_point: bool
) -> tuple[bool, str]:
    """(맞음 여부, 결과 분류). 위에서 먼저 걸리는 것으로 정한다.

    정답이 후보에 없으면 단서가 아무리 좋아도 못 맞히므로, 되묻기보다 "후보 없음"을 먼저 본다.
    검출기 몫의 실패를 단서 몫으로 세면 v2~v5 의 개선이 흐려진다.
    """
    chosen = resolution.chosen
    if chosen is not None and chosen.detection.bbox.iou(sample["gt"]) >= IOU_MATCH:
        return True, "정답"
    if resolution.needs_clarify and sample["ambiguous"]:
        return True, "되묻는 게 맞음"  # 사람도 못 고르는 장면에서는 묻는 게 정답
    if not candidate_hit:
        return False, "후보 없음"
    if resolution.needs_clarify:
        return False, "불필요한 되묻기"
    if sample["has_hand"] and uses_point:
        return False, "방향 오판"
    if resolution.margin >= HIGH_CONFIDENCE:
        return False, "높은 확신으로 틀림"
    return False, "다른 후보 1위"


def evaluate(
    samples: list[dict],
    candidates: dict[str, list[Detection]],
    detect_ms: dict[str, float],
    variant: str,
    frame_size: dict[str, tuple[int, int]],
) -> list[dict]:
    """한 버전으로 모든 사진을 채점한다. 검출은 밖에서 한 번 해 둔 후보를 쓴다."""
    cues = VARIANTS[variant]
    rows = []
    for s in samples:
        found = candidates[s["file"]]
        hit = any(d.bbox.iou(s["gt"]) >= IOU_MATCH for d in found)
        started = time.perf_counter()
        resolution = resolve_target(
            candidates=found,
            fingertip=None,  # 손 단서(v2)를 붙일 때 손 관절 결과를 넣는다
            gaze_vector=None,
            text=s["utterance"],
            frame_size=frame_size[s["file"]],
            prev_target_label=s["prev_label"],
            enabled_cues=set(cues),
        )
        resolve_ms = (time.perf_counter() - started) * 1000
        correct, outcome = judge(s, resolution, hit, CueKind.POINT in cues)
        chosen = resolution.chosen
        rows.append(
            {
                "file": s["file"],
                "gt_label": s["gt_label"],
                "pred_label": chosen.detection.label if chosen is not None else None,
                "outcome": outcome,
                "correct": correct,
                "margin": round(resolution.margin, 3),
                "has_hand": s["has_hand"],
                "ambiguous": s["ambiguous"],
                "candidate_hit": hit,
                "answered": chosen is not None,
                "clarified": resolution.needs_clarify,
                "detect_ms": round(detect_ms[s["file"]], 1),
                "resolve_ms": round(resolve_ms, 3),
            }
        )
    return rows


def percentiles(values: list[float]) -> tuple[float, float]:
    """p50 · p95. measure_latency.py · store.latency_percentiles 와 같은 nearest-rank."""
    ordered = sorted(values)

    def pick(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return pick(0.50), pick(0.95)


def _rate(part: list[dict], key: str) -> float | None:
    """part 에서 key 가 참인 비율. part 가 비면 None(0 으로 두면 "전부 틀림"으로 읽힌다)."""
    return round(sum(r[key] for r in part) / len(part), 3) if part else None


def summarize(rows: list[dict]) -> dict:
    """한 버전의 지표. eval_run.metrics 에 그대로 들어간다."""
    counts = Counter(r["outcome"] for r in rows)
    detect_p50, detect_p95 = percentiles([r["detect_ms"] for r in rows])
    resolve_p50, resolve_p95 = percentiles([r["resolve_ms"] for r in rows])
    return {
        "accuracy": _rate(rows, "correct"),
        "answered_accuracy": _rate([r for r in rows if r["answered"]], "correct"),
        "reask_rate": _rate(rows, "clarified"),
        "candidate_recall": _rate(rows, "candidate_hit"),
        "accuracy_with_hand": _rate([r for r in rows if r["has_hand"]], "correct"),
        "accuracy_without_hand": _rate([r for r in rows if not r["has_hand"]], "correct"),
        "high_conf_error_rate": round(counts["높은 확신으로 틀림"] / len(rows), 3),
        "outcomes": {name: counts[name] for name in OUTCOMES},
        "detect_p50_ms": detect_p50,
        "detect_p95_ms": detect_p95,
        "resolve_p50_ms": resolve_p50,
        "resolve_p95_ms": resolve_p95,
    }


def save_run(store: MemoryStore, variant: str, rows: list[dict], metrics: dict, note: str) -> int:
    """eval_run 한 줄과 eval_sample 사진 수만큼을 같은 트랜잭션으로 남기고 run_id 를 돌려준다."""
    with store.session() as db:
        run = EvalRun(variant=variant, sample_count=len(rows), metrics=metrics, note=note or None)
        db.add(run)
        db.flush()  # run_id 를 받아야 eval_sample 이 가리킬 수 있다
        for r in rows:
            # turn_id 는 비운다. 실제 대화 턴이 아니라 평가 사진이다.
            db.add(
                EvalSample(
                    run_id=run.run_id,
                    gt_label=r["gt_label"],
                    pred_label=r["pred_label"],
                    is_correct=r["correct"],
                )
            )
        return run.run_id


def previous_accuracy(store: MemoryStore, variant: str, before: int) -> float | None:
    """같은 버전의 바로 이전 실행의 정답률. 없으면 None."""
    with store.session() as db:
        stmt = (
            select(EvalRun.metrics)
            .where(EvalRun.variant == variant, EvalRun.run_id < before)
            .order_by(EvalRun.run_id.desc())
            .limit(1)
        )
        metrics = db.scalar(stmt)
    return metrics.get("accuracy") if metrics else None


def write_csv(out_dir: Path, run_id: int, variant: str, rows: list[dict]) -> Path:
    """사진별 결과. 파일 이름의 run_id 로 DB 의 실행과 짝지어 실패 사진을 찾아간다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}_{variant}.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}"


def format_row(variant: str, metrics: dict, n: int, previous: float | None) -> str:
    """비교표 한 줄. 이전 실행이 있으면 정답률 옆에 차이(%p)를 붙인다."""
    accuracy = _pct(metrics["accuracy"])
    if previous is not None:
        accuracy += f" ({(metrics['accuracy'] - previous) * 100:+.1f})"
    hand = f"{_pct(metrics['accuracy_with_hand'])}/{_pct(metrics['accuracy_without_hand'])}"
    return (
        f"{variant} | {n} | {accuracy} | {_pct(metrics['answered_accuracy'])} | "
        f"{_pct(metrics['reask_rate'])} | {hand} | {_pct(metrics['candidate_recall'])} | "
        f"{metrics['resolve_p50_ms']:.2f}/{metrics['resolve_p95_ms']:.2f}"
    )


def read_image(path: Path) -> np.ndarray | None:
    """cv2.imread 는 윈도우에서 경로에 한글이 있으면 조용히 실패해서, 바이트로 읽어 푼다."""
    data = np.fromfile(path, dtype=np.uint8) if path.exists() else np.empty(0, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description="가리킴 v1~v5 ablation")
    parser.add_argument("--labels", type=Path, default=POINTING_DIR / "labels.csv")
    parser.add_argument("--raw", type=Path, default=POINTING_DIR / "raw")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--db", default=config.DATABASE_URL, help="결과를 남길 DB")
    parser.add_argument("--note", default="", help="이번 실행 메모, 예: v2 손 방향 추가")
    parser.add_argument("--out", type=Path, default=POINTING_DIR / "runs", help="사진별 CSV 폴더")
    args = parser.parse_args()

    if args.split == "test":
        print("주의: test 를 연다 — 최종 측정 때만. 보면서 고치면 test 숫자가 실력이 아니게 된다.")
    samples = load_samples(args.labels, args.split)
    if not samples:
        raise SystemExit(f"{args.split} 로 표시된 사진이 없다: {args.labels}")
    ready, skipped = implemented_variants(args.variants)
    if skipped:
        print(f"미구현이라 건너뜀: {', '.join(skipped)}")
    if not ready:
        raise SystemExit("돌릴 수 있는 버전이 없다")

    # 사진마다 한 번만 검출한다. ultralytics 는 CI 에 없어서 여기서 불러온다.
    from jarviseo.pointing.detector import ObjectDetector

    detector = ObjectDetector()
    # 첫 호출은 모델을 올리느라 느리다(실측 861ms, 그다음 30ms 안팎). 재기 전에 한 번 버린다.
    detector.detect(np.zeros((640, 640, 3), np.uint8))
    candidates, detect_ms, frame_size = {}, {}, {}
    for s in samples:
        image = read_image(args.raw / s["file"])
        if image is None:
            raise SystemExit(f"사진을 못 읽었다: {args.raw / s['file']}")
        started = time.perf_counter()
        candidates[s["file"]] = detector.detect(image)
        detect_ms[s["file"]] = (time.perf_counter() - started) * 1000
        height, width = image.shape[:2]
        frame_size[s["file"]] = (width, height)

    # 쓰는 쪽이라 테이블이 없으면 만든다(이미 있으면 아무것도 안 한다).
    store = MemoryStore(database_url=args.db)
    store.init_schema()
    # 검출은 모든 버전이 같은 결과를 쓰므로 버전 줄마다가 아니라 한 번만 찍는다.
    detect_p50, detect_p95 = percentiles(list(detect_ms.values()))
    print(
        f"\n{args.split} {len(samples)}장 · 검출 p50 {detect_p50:.1f}ms · p95 {detect_p95:.1f}ms\n"
    )
    print(
        "버전 | 장수 | 정답률 | 답한 것 정답률 | 되묻기율 | "
        "손있음/손없음 | 후보재현율 | resolve p50/p95 ms"
    )
    for variant in ready:
        rows = evaluate(samples, candidates, detect_ms, variant, frame_size)
        metrics = summarize(rows)
        run_id = save_run(store, variant, rows, metrics, args.note)
        previous = previous_accuracy(store, variant, run_id)
        print(format_row(variant, metrics, len(rows), previous))
        path = write_csv(args.out, run_id, variant, rows)
        failed = {k: v for k, v in metrics["outcomes"].items() if v and k not in OUTCOMES[:2]}
        print(f"    run {run_id} · 실패 {failed or '없음'} · {path}")


if __name__ == "__main__":
    main()
