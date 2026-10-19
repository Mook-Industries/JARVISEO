"""가리킴 ablation — 정답 표시 읽기, 7분류 채점, 지표, DB 기록, 이전 실행과 비교.

YOLO 없이 정해 둔 후보로 채점한다. DB 기록은 store fixture 로 SQLite 와 실제 Postgres 에서 돈다.
"""

import csv
import importlib.util
from pathlib import Path

from jarviseo.memory import EvalRun, EvalSample
from jarviseo.types import BBox, Detection, TargetCandidate, TargetResolution

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_ablation.py"
_spec = importlib.util.spec_from_file_location("run_ablation", _SCRIPT)
run_ablation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_ablation)

FRAME = (1920, 1080)
GT = BBox(860, 440, 1060, 640)  # 화면 가운데 컵
LEFT = BBox(400, 440, 600, 640)
RIGHT = BBox(1320, 440, 1520, 640)  # LEFT 와 화면 가운데를 두고 좌우 대칭


def _det(bbox: BBox, label: str = "cup") -> Detection:
    return Detection(label=label, confidence=0.9, bbox=bbox)


def _sample(gt=GT, *, has_hand=False, ambiguous=False, file="s01/a.jpg") -> dict:
    return {
        "file": file,
        "utterance": "저거 뭐야?",
        "gt": gt,
        "gt_label": "cup",
        "has_hand": has_hand,
        "ambiguous": ambiguous,
        "prev_label": None,
    }


def _resolution(chosen: BBox | None, *, clarify=False, margin=0.3) -> TargetResolution:
    picked = TargetCandidate(_det(chosen)) if chosen is not None else None
    return TargetResolution([], picked, margin, clarify)


def _row(**overrides) -> dict:
    row = {
        "file": "s01/a.jpg",
        "gt_label": "cup",
        "pred_label": "cup",
        "outcome": "정답",
        "correct": True,
        "margin": 0.3,
        "has_hand": False,
        "ambiguous": False,
        "candidate_hit": True,
        "answered": True,
        "clarified": False,
        "detect_ms": 40.0,
        "resolve_ms": 0.2,
    }
    return {**row, **overrides}


# --- 정답 표시 읽기 -----------------------------------------------------------------


def test_load_samples_keeps_split_and_drops_excluded(tmp_path):
    """그 split 의 쓸 수 있는 사진만 읽고, 0/1 표시와 빈 앞 대상을 풀어 둔다."""
    path = tmp_path / "labels.csv"
    columns = ["file", "split", "utterance", "gt_x1", "gt_y1", "gt_x2", "gt_y2"]
    columns += ["gt_label", "has_hand", "ambiguous", "prev_label", "excluded"]
    rows = [
        ["s01/a.jpg", "valid", "저거 뭐야?", "1", "2", "3", "4", "cup", "1", "0", "", "0"],
        ["s01/b.jpg", "valid", "흐림", "", "", "", "", "", "0", "0", "", "1"],
        ["s03/c.jpg", "test", "이거 뭐야?", "1", "2", "3", "4", "cup", "0", "0", "", "0"],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    samples = run_ablation.load_samples(path, "valid")

    assert [s["file"] for s in samples] == ["s01/a.jpg"]
    assert samples[0]["gt"] == BBox(1, 2, 3, 4)
    assert (samples[0]["has_hand"], samples[0]["ambiguous"]) == (True, False)
    assert samples[0]["prev_label"] is None


def test_load_samples_without_file_is_empty(tmp_path):
    """표시 파일이 없으면 빈 목록이다."""
    assert run_ablation.load_samples(tmp_path / "none.csv", "valid") == []


def test_unimplemented_variants_are_skipped():
    """지금 구현된 단서는 center 하나라 v1 만 돌고 v2~v5 는 건너뛴다."""
    ready, skipped = run_ablation.implemented_variants(["v1", "v2", "v5"])
    assert (ready, skipped) == (["v1"], ["v2", "v5"])


# --- 7분류 채점 ---------------------------------------------------------------------


def test_judge_correct_when_iou_over_half():
    """고른 박스가 정답과 절반 넘게 겹치면 맞음."""
    assert run_ablation.judge(_sample(), _resolution(GT), True, False) == (True, "정답")


def test_judge_clarify_on_ambiguous_is_correct():
    """사람도 못 고르는 장면에서 되물으면 맞은 것으로 센다."""
    sample, resolution = _sample(ambiguous=True), _resolution(None, clarify=True)
    assert run_ablation.judge(sample, resolution, True, False) == (True, "되묻는 게 맞음")


def test_judge_no_candidate_comes_before_clarify():
    """정답이 후보에 없으면 되물었어도 '후보 없음'(검출기 몫)이다."""
    result = run_ablation.judge(_sample(), _resolution(None, clarify=True), False, False)
    assert result == (False, "후보 없음")


def test_judge_needless_clarify():
    """모호하지 않은데 되물었으면 '불필요한 되묻기'."""
    result = run_ablation.judge(_sample(), _resolution(None, clarify=True), True, False)
    assert result == (False, "불필요한 되묻기")


def test_judge_direction_miss_only_with_hand_and_point_cue():
    """손이 보이고 손 단서를 켠 버전에서 틀리면 '방향 오판'. 손 단서를 안 켰으면 아니다."""
    sample = _sample(has_hand=True)
    assert run_ablation.judge(sample, _resolution(LEFT), True, True) == (False, "방향 오판")
    assert run_ablation.judge(sample, _resolution(LEFT), True, False)[1] != "방향 오판"


def test_judge_high_confidence_and_other_top():
    """틀렸는데 margin 이 0.5 이상이면 '높은 확신으로 틀림', 아니면 '다른 후보 1위'."""
    sample = _sample()
    high = run_ablation.judge(sample, _resolution(LEFT, margin=1.0), True, False)
    low = run_ablation.judge(sample, _resolution(LEFT, margin=0.2), True, False)
    assert (high, low) == ((False, "높은 확신으로 틀림"), (False, "다른 후보 1위"))


# --- 한 버전 채점 (실제 resolve_target) ------------------------------------------------


def test_evaluate_v1_picks_center_and_records_row():
    """v1(가운데)은 가운데 컵을 고른다. 사진별 결과에 분류 · 후보 적중 · 지연이 남는다."""
    samples = [_sample()]
    candidates = {"s01/a.jpg": [_det(BBox(0, 0, 200, 200), "bottle"), _det(GT)]}
    detect_ms, frames = {"s01/a.jpg": 40.0}, {"s01/a.jpg": FRAME}

    [row] = run_ablation.evaluate(samples, candidates, detect_ms, "v1", frames)

    assert (row["outcome"], row["correct"], row["pred_label"]) == ("정답", True, "cup")
    assert row["candidate_hit"] is True
    assert row["detect_ms"] == 40.0


def test_evaluate_symmetric_tie_clarifies():
    """좌우 대칭 컵 둘은 v1 이 되묻는다. 모호하다고 표시했으면 맞음."""
    samples = [_sample(gt=LEFT, ambiguous=True)]
    candidates = {"s01/a.jpg": [_det(LEFT), _det(RIGHT)]}
    detect_ms, frames = {"s01/a.jpg": 40.0}, {"s01/a.jpg": FRAME}

    [row] = run_ablation.evaluate(samples, candidates, detect_ms, "v1", frames)

    assert (row["outcome"], row["clarified"], row["answered"]) == ("되묻는 게 맞음", True, False)


# --- 지표 ---------------------------------------------------------------------------


def test_summarize_rates_and_outcome_counts():
    """정답률 · 답한 것 정답률 · 되묻기율 · 손 있음/없음 · 분류 개수를 낸다."""
    rows = [
        _row(),
        _row(has_hand=True, correct=False, outcome="다른 후보 1위", pred_label="bottle"),
        _row(answered=False, clarified=True, correct=False, outcome="불필요한 되묻기"),
        _row(candidate_hit=False, correct=False, outcome="후보 없음"),
    ]

    m = run_ablation.summarize(rows)

    assert m["accuracy"] == 0.25
    assert m["answered_accuracy"] == round(1 / 3, 3)  # 고른 3장 중 1장
    assert m["reask_rate"] == 0.25
    assert m["candidate_recall"] == 0.75
    assert (m["accuracy_with_hand"], m["accuracy_without_hand"]) == (0.0, round(1 / 3, 3))
    assert m["outcomes"]["후보 없음"] == 1
    assert set(m["outcomes"]) == set(run_ablation.OUTCOMES)


def test_summarize_without_hand_photos_is_none():
    """손 없는 사진이 0장이면 그 정답률은 0 이 아니라 None 이다."""
    m = run_ablation.summarize([_row(has_hand=True)])
    assert m["accuracy_without_hand"] is None


# --- DB 기록 · 비교 -----------------------------------------------------------------


def test_save_run_writes_run_and_samples(store):
    """eval_run 한 줄과 사진 수만큼의 eval_sample 이 남는다."""
    rows = [_row(), _row(correct=False, pred_label=None, outcome="후보 없음")]
    metrics = run_ablation.summarize(rows)

    run_id = run_ablation.save_run(store, "v1", rows, metrics, "기준선")

    with store.session() as db:
        run = db.get(EvalRun, run_id)
        samples = db.query(EvalSample).filter_by(run_id=run_id).all()
        assert (run.variant, run.sample_count, run.note) == ("v1", 2, "기준선")
        assert run.metrics["accuracy"] == 0.5
        assert sorted(s.is_correct for s in samples) == [False, True]


def test_previous_accuracy_is_same_variant_before(store):
    """같은 버전의 바로 이전 실행 정답률을 찾는다. 다른 버전 · 첫 실행이면 None."""
    first = run_ablation.save_run(store, "v1", [_row()], {"accuracy": 0.6}, "")
    run_ablation.save_run(store, "v2", [_row()], {"accuracy": 0.9}, "")
    second = run_ablation.save_run(store, "v1", [_row()], {"accuracy": 0.7}, "")

    assert run_ablation.previous_accuracy(store, "v1", second) == 0.6
    assert run_ablation.previous_accuracy(store, "v1", first) is None


def test_format_row_shows_delta_from_previous():
    """이전 실행이 있으면 정답률 옆에 차이(%p)를 붙이고, 끝에 대상 결정 p50/p95 를 둔다."""
    m = run_ablation.summarize([_row(), _row(correct=False, outcome="다른 후보 1위")])

    line = run_ablation.format_row("v1", m, 2, previous=0.4)

    assert line.startswith("v1 | 2 | 50.0 (+10.0) |")
    assert line.endswith("| 0.20/0.20")  # 대상 결정 p50/p95


def test_write_csv_names_file_by_run(tmp_path):
    """사진별 CSV 는 run_id_버전.csv 이고 한 줄에 한 사진이다."""
    path = run_ablation.write_csv(tmp_path / "runs", 7, "v1", [_row(), _row(file="s01/b.jpg")])

    assert path.name == "7_v1.csv"
    with path.open(encoding="utf-8", newline="") as f:
        assert [r["file"] for r in csv.DictReader(f)] == ["s01/a.jpg", "s01/b.jpg"]
