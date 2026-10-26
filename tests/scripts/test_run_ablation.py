"""가리킴 ablation — 정답 표시 읽기(세션 고르기), 8분류 채점, v1 · v2 채점, v2+C(손 우선 크롭),
v2+lang · v2+lang+ctx(말 · 이어 묻기 임시 버전, 색은 raw 사진에서),
확신 오답 · 크롭 담김, 지표, DB 기록, 실행 조건(커밋 · 가중치 지문 · 세션) 기록과
같은 조건끼리만 비교, 가중치끼리 후보 적중 변화.

YOLO · MediaPipe 없이 정해 둔 후보와 손으로 채점한다. DB 기록은 store fixture 로 SQLite 와
실제 Postgres 에서 돈다.
"""

import csv
import hashlib
import importlib.util
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from jarviseo.memory import EvalRun, EvalSample
from jarviseo.pointing.hands import HandPose
from jarviseo.types import BBox, CueKind, Detection, TargetCandidate, TargetResolution

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


def _hand(base: tuple[float, float], tip: tuple[float, float]) -> HandPose:
    """검지 뿌리(5) → 끝(8)만 의미 있는 손. 채점은 방향만 쓴다."""
    landmarks = [(0.0, 0.0)] * 21
    landmarks[5], landmarks[8] = base, tip
    return HandPose(tuple(landmarks), ((0.0, 0.0, 0.0),) * 21, "Right", 0.9)


# 화면 아래에서 RIGHT 중심(1420, 540) · LEFT 중심(500, 540)을 똑바로 가리키는 손
TO_RIGHT = _hand(base=(1068.0, 936.0), tip=(1100.0, 900.0))
TO_LEFT = _hand(base=(852.0, 936.0), tip=(820.0, 900.0))


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
        "hand_used": False,
        "ambiguous": False,
        "candidate_hit": True,
        "crop_covered": 1.0,
        "crop_area": 0.05,
        "crop_contained": True,
        "hand_crop": False,
        "answered": True,
        "clarified": False,
        "confident_error": False,
        "detect_ms": 40.0,
        "resolve_ms": 0.2,
    }
    return {**row, **overrides}


# --- 정답 표시 읽기 -----------------------------------------------------------------


def _labels_csv(tmp_path) -> Path:
    """세션 1 · 3 · 4 가 섞인 정답표. 세션 4 는 말 · 맥락 세션처럼 나중에 valid 로 붙은 것."""
    path = tmp_path / "labels.csv"
    columns = ["file", "session", "split", "utterance", "gt_x1", "gt_y1", "gt_x2", "gt_y2"]
    columns += ["gt_label", "has_hand", "ambiguous", "prev_label", "excluded"]
    rows = [
        ["s01/a.jpg", "1", "valid", "저거 뭐야?", "1", "2", "3", "4", "cup", "1", "0", "", "0"],
        ["s01/b.jpg", "1", "valid", "흐림", "", "", "", "", "", "0", "0", "", "1"],
        ["s03/c.jpg", "3", "test", "이거 뭐야?", "1", "2", "3", "4", "cup", "0", "0", "", "0"],
        ["s04/d.jpg", "4", "valid", "빨간 거 뭐야?", "1", "2", "3", "4", "cup", "0", "0", "", "0"],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def test_load_samples_keeps_split_and_drops_excluded(tmp_path):
    """그 split · 세션의 쓸 수 있는 사진만 읽고, 0/1 표시와 빈 앞 대상을 풀어 둔다."""
    samples = run_ablation.load_samples(_labels_csv(tmp_path), "valid", [1, 2])

    assert [s["file"] for s in samples] == ["s01/a.jpg"]
    assert samples[0]["gt"] == BBox(1, 2, 3, 4)
    assert (samples[0]["has_hand"], samples[0]["ambiguous"]) == (True, False)
    assert samples[0]["prev_label"] is None


def test_load_samples_default_sessions_keep_62_photo_set(tmp_path):
    """기본 세션(valid 1 · 2)이면 나중에 붙은 세션 4 가 안 섞인다. 세션 4 는 따로 골라야 읽힌다."""
    path = _labels_csv(tmp_path)
    default = run_ablation.DEFAULT_SESSIONS["valid"]

    assert [s["file"] for s in run_ablation.load_samples(path, "valid", default)] == ["s01/a.jpg"]
    assert [s["file"] for s in run_ablation.load_samples(path, "valid", [4])] == ["s04/d.jpg"]


def test_load_samples_without_file_is_empty(tmp_path):
    """표시 파일이 없으면 빈 목록이다."""
    assert run_ablation.load_samples(tmp_path / "none.csv", "valid", [1, 2]) == []


def test_unimplemented_variants_are_skipped():
    """지금 구현된 단서는 center · point 라 v1 · v2 만 돌고 v3~v5 는 건너뛴다."""
    ready, skipped = run_ablation.implemented_variants(["v1", "v2", "v5"])
    assert (ready, skipped) == (["v1", "v2"], ["v5"])


# --- 8분류 채점 ---------------------------------------------------------------------


def test_judge_correct_when_iou_over_half():
    """고른 박스가 정답과 절반 넘게 겹치면 맞음."""
    assert run_ablation.judge(_sample(), _resolution(GT), True, False) == (True, "정답")


def test_judge_clarify_on_ambiguous_is_correct():
    """사람도 못 고르는 장면에서 되물으면 맞은 것으로 센다."""
    sample, resolution = _sample(ambiguous=True), _resolution(None, clarify=True)
    assert run_ablation.judge(sample, resolution, True, False) == (True, "되묻는 게 맞음")


def test_judge_no_candidate_comes_before_clarify():
    """정답이 후보에 없으면 '불필요한 되묻기'가 아니라 후보 쪽(검출기 몫)으로 센다.

    되물었으면 안전한 쪽이다.
    """
    result = run_ablation.judge(_sample(), _resolution(None, clarify=True), False, False)
    assert result == (False, "후보 없어 되물음")


def test_judge_no_candidate_but_answered_is_separate():
    """정답이 후보에 없는데 엉뚱한 물건을 단정했으면 '후보 없는데 답함'(위험한 쪽)으로 따로 센다."""
    high = run_ablation.judge(_sample(), _resolution(LEFT, margin=1.0), False, False)
    low = run_ablation.judge(_sample(), _resolution(LEFT, margin=0.2), False, False)
    assert high == low == (False, "후보 없는데 답함")


def test_outcomes_split_no_candidate_into_two():
    """결과 분류는 judge 의 8가지 + 손 방향 크롭이다. 옛 '후보 없음' 하나는 없다."""
    assert len(run_ablation.OUTCOMES) == 9
    assert run_ablation.OUTCOMES[-1] == run_ablation.HAND_CROP
    assert "후보 없음" not in run_ablation.OUTCOMES


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

    [row] = run_ablation.evaluate(samples, candidates, detect_ms, "v1", frames, {"s01/a.jpg": None})

    assert (row["outcome"], row["correct"], row["pred_label"]) == ("정답", True, "cup")
    assert row["candidate_hit"] is True
    assert row["hand_used"] is False
    assert row["detect_ms"] == 40.0


def test_evaluate_symmetric_tie_clarifies():
    """좌우 대칭 컵 둘은 v1 이 되묻는다. 모호하다고 표시했으면 맞음."""
    samples = [_sample(gt=LEFT, ambiguous=True)]
    candidates = {"s01/a.jpg": [_det(LEFT), _det(RIGHT)]}
    detect_ms, frames = {"s01/a.jpg": 40.0}, {"s01/a.jpg": FRAME}

    [row] = run_ablation.evaluate(samples, candidates, detect_ms, "v1", frames, {"s01/a.jpg": None})

    assert (row["outcome"], row["clarified"], row["answered"]) == ("되묻는 게 맞음", True, False)


def _evaluate_one(variant, sample, candidates, hand, raw=None):
    """사진 한 장을 한 버전으로 채점한다. 후보와 손은 정해 둔 것을 쓴다."""
    file = sample["file"]
    [row] = run_ablation.evaluate(
        [sample], {file: candidates}, {file: 40.0}, variant, {file: FRAME}, {file: hand}, raw=raw
    )
    return row


def test_evaluate_v2_hand_picks_pointed_cup():
    """좌우 대칭 컵 둘: v1 은 되묻고, v2 는 손이 가리킨 쪽을 고른다. 두 버전이 같은 손을 쓴다."""
    sample = _sample(gt=RIGHT, has_hand=True)
    candidates = [_det(LEFT), _det(RIGHT)]

    v1 = _evaluate_one("v1", sample, candidates, TO_RIGHT)
    v2 = _evaluate_one("v2", sample, candidates, TO_RIGHT)

    assert (v1["outcome"], v1["hand_used"]) == ("불필요한 되묻기", False)
    assert (v2["outcome"], v2["hand_used"]) == ("정답", True)


def test_evaluate_v2_wrong_direction_is_direction_miss():
    """손 단서가 점수를 냈는데 다른 쪽을 골랐으면 '방향 오판'이다."""
    sample = _sample(gt=RIGHT, has_hand=True)

    row = _evaluate_one("v2", sample, [_det(LEFT), _det(RIGHT)], TO_LEFT)

    assert (row["outcome"], row["hand_used"]) == ("방향 오판", True)


def test_evaluate_v2_without_hand_is_not_direction_miss():
    """손 있음 사진이라도 손을 못 찾아 기권했으면, 틀려도 '방향 오판'이 아니다(손 검출 몫)."""
    corner = BBox(0, 0, 200, 200)
    sample = _sample(gt=corner, has_hand=True)

    row = _evaluate_one("v2", sample, [_det(corner, "bottle"), _det(GT)], None)

    assert row["hand_used"] is False
    assert row["correct"] is False
    assert row["outcome"] != "방향 오판"


# --- 확신 오답 · 크롭 담김 ------------------------------------------------------------


def test_crop_coverage_is_share_of_answer_inside_crop():
    """정답 박스 중 크롭 안에 든 비율과 크롭이 사진에서 차지하는 비율. IoU 가 아니다."""
    gt = BBox(0, 0, 100, 100)
    assert run_ablation.crop_coverage(gt, BBox(0, 0, 200, 100), (400, 250)) == (1.0, 0.2)
    assert run_ablation.crop_coverage(gt, BBox(50, 0, 200, 100), (400, 250))[0] == 0.5
    assert run_ablation.crop_coverage(gt, BBox(300, 0, 400, 100), (400, 250))[0] == 0.0


def test_crop_coverage_with_empty_answer_box_is_zero():
    """정답 박스 넓이가 0 이면 0 으로 나누지 않고 0.0 이다."""
    assert run_ablation.crop_coverage(BBox(5, 5, 5, 5), BBox(0, 0, 10, 10), (10, 10))[0] == 0.0


@pytest.mark.parametrize(
    ("covered", "area", "expected"),
    [
        (0.6, 0.5, True),  # 경계는 담김
        (0.599, 0.1, False),  # 정답을 60% 못 담음
        (1.0, 0.501, False),  # 사진 절반을 넘게 덮으면 무엇이든 담겨서 세지 않는다
        (None, None, False),  # 되물어서 크롭이 없음
    ],
)
def test_is_crop_contained_boundaries(covered, area, expected):
    assert run_ablation.is_crop_contained(covered, area) is expected


def test_evaluate_records_crop_for_chosen_target():
    """고른 박스에 실제 비서와 같은 여백 15% 를 붙여 잰다. 200×200 컵 → 260×260 크롭."""
    row = _evaluate_one("v1", _sample(), [_det(GT)], None)

    assert (row["crop_covered"], row["crop_contained"]) == (1.0, True)
    assert row["crop_area"] == round(260 * 260 / (1920 * 1080), 3)
    assert row["confident_error"] is False


def test_evaluate_clarified_photo_has_no_crop():
    """되물었으면 VLM 에 보낼 크롭이 없다 — 비율 · 넓이는 비우고 담김 아님, 확신 오답도 아님."""
    row = _evaluate_one("v1", _sample(gt=LEFT), [_det(LEFT), _det(RIGHT)], None)

    assert row["clarified"] is True
    assert (row["crop_covered"], row["crop_area"], row["crop_contained"]) == (None, None, False)
    assert row["confident_error"] is False


def test_evaluate_screen_covering_box_is_confident_error_not_contained():
    """화면을 다 덮는 헛 박스(냉장고) 하나뿐이면 단정한다. 정답을 다 담아도 크롭 담김이 아니다."""
    fridge = _det(BBox(0, 0, 1920, 1080), "refrigerator")

    row = _evaluate_one("v1", _sample(), [fridge], None)

    assert row["outcome"] == "후보 없는데 답함"
    assert row["confident_error"] is True
    assert (row["crop_covered"], row["crop_area"], row["crop_contained"]) == (1.0, 1.0, False)


# --- 손 우선 크롭 (v2+C) ----------------------------------------------------------------

# 화면 아래 가운데에서 똑바로 위를 가리키는 손. LEFT · RIGHT 는 선에서 45° 벗어나 손 점수가 0.3 미만
TO_UP = _hand(base=(960.0, 1040.0), tip=(960.0, 1000.0))
# 손끝 위쪽 물건. 후보에는 없다(검출기가 놓친 물건)
ABOVE = BBox(860, 600, 1060, 800)


def test_hand_crop_variant_is_ready_and_separate():
    """v2+C 는 단서가 v2 와 같아 바로 돌고, 손 크롭을 켜는 버전은 v2+C 하나다."""
    assert run_ablation.implemented_variants(["v2", "v2+C"]) == (["v2", "v2+C"], [])
    assert run_ablation.VARIANTS["v2+C"] == run_ablation.VARIANTS["v2"]
    assert run_ablation.HAND_CROP_VARIANTS == {"v2+C"}


def test_evaluate_v2c_crops_fingertip_when_no_candidate_on_line():
    """선 위에 후보가 없으면 v2 는 되묻고(후보 없어 되물음), v2+C 는 손끝 방향을 잘라 정답을 담는다.

    손 크롭은 IoU 로 맞을 수 없어 정답 · 답함 · 확신 오답에 넣지 않고 크롭 담김으로만 센다.
    """
    sample = _sample(gt=ABOVE, has_hand=True)
    candidates = [_det(LEFT), _det(RIGHT)]

    v2 = _evaluate_one("v2", sample, candidates, TO_UP)
    v2c = _evaluate_one("v2+C", sample, candidates, TO_UP)

    assert (v2["outcome"], v2["clarified"], v2["hand_crop"]) == ("후보 없어 되물음", True, False)
    assert (v2c["outcome"], v2c["hand_crop"], v2c["crop_contained"]) == ("손 방향 크롭", True, True)
    assert (v2c["correct"], v2c["answered"], v2c["clarified"]) == (False, False, False)
    assert (v2c["confident_error"], v2c["pred_label"]) == (False, None)
    # 사각형 (636, 460)-(1284, 1080) 이 정답을 다 담고 사진의 약 19% 를 덮는다
    assert (v2c["crop_covered"], v2c["crop_area"]) == (1.0, round(648 * 620 / (1920 * 1080), 3))


def test_evaluate_v2c_keeps_box_answer_when_hand_points_at_candidate():
    """손이 후보를 가리키면 v2+C 도 v2 와 똑같이 그 박스를 고른다(손 크롭 없음)."""
    sample = _sample(gt=RIGHT, has_hand=True)
    candidates = [_det(LEFT), _det(RIGHT)]

    v2 = _evaluate_one("v2", sample, candidates, TO_RIGHT)
    v2c = _evaluate_one("v2+C", sample, candidates, TO_RIGHT)

    assert (v2c["outcome"], v2c["hand_crop"]) == ("정답", False)
    del v2["resolve_ms"], v2c["resolve_ms"]
    assert v2c == v2


def test_evaluate_v2c_without_hand_still_asks():
    """손을 못 찾은 사진에서는 손 크롭이 없다. 후보가 0개면 되묻는다(손 없이는 자를 곳이 없다)."""
    row = _evaluate_one("v2+C", _sample(has_hand=True), [], None)
    assert (row["outcome"], row["hand_crop"], row["clarified"]) == ("후보 없어 되물음", False, True)


# --- 말 · 이어 묻기 임시 버전 (v2+lang · v2+lang+ctx, #166) ------------------------------------


def test_language_variants_are_ready_and_stack_on_v2():
    """시선(v3) 없이도 돈다. 단서는 v2 에 말 · 이어 묻기를 하나씩 더한 것이고, 손 크롭은 안 켠다."""
    names = ["v2+lang", "v2+lang+ctx"]
    assert run_ablation.implemented_variants(names) == (names, [])
    v2 = run_ablation.VARIANTS["v2"]
    assert run_ablation.VARIANTS["v2+lang"] == v2 | {CueKind.LANG}
    assert run_ablation.VARIANTS["v2+lang+ctx"] == v2 | {CueKind.LANG, CueKind.CTX}
    assert not set(names) & run_ablation.HAND_CROP_VARIANTS


def _write_photo(path: Path, *paints: tuple[BBox, tuple[int, int, int]]) -> None:
    """검은 1080p 사진에 박스마다 색을 칠해 저장한다(run_ablation.read_image 가 읽는 방식)."""
    image = np.zeros((1080, 1920, 3), np.uint8)
    for box, color in paints:
        image[int(box.y1) : int(box.y2), int(box.x1) : int(box.x2)] = color
    path.parent.mkdir(parents=True)
    cv2.imencode(".png", image)[1].tofile(path)


def test_evaluate_v2_lang_reads_color_from_raw_photo(tmp_path):
    """ "노란 거"는 raw 폴더의 사진에서 색을 봐서 대칭인 두 컵 중 노란 쪽을 고른다.

    사진을 안 주면(예전 동작) 색이 기권해 v2 처럼 되묻는다.
    """
    _write_photo(tmp_path / "s04" / "d.png", (LEFT, (0, 255, 255)), (RIGHT, (255, 0, 0)))
    sample = {**_sample(gt=LEFT, file="s04/d.png"), "utterance": "노란 거 뭐야?"}
    candidates = [_det(LEFT), _det(RIGHT)]

    v2 = _evaluate_one("v2", sample, candidates, None, raw=tmp_path)
    no_photo = _evaluate_one("v2+lang", sample, candidates, None)
    with_photo = _evaluate_one("v2+lang", sample, candidates, None, raw=tmp_path)

    assert (v2["clarified"], no_photo["clarified"]) == (True, True)
    assert (with_photo["outcome"], with_photo["correct"]) == ("정답", True)


def test_evaluate_v2_lang_ctx_reads_previous_target():
    """ "그거"는 정답표의 앞 대상(prev_label)으로 읽는다.

    말 단서만으로는 못 가르는 컵 · 병에서 병을 고른다.
    """
    sample = {**_sample(gt=RIGHT), "utterance": "그거 얼마나 남았어?", "prev_label": "bottle"}
    candidates = [_det(LEFT, "cup"), _det(RIGHT, "bottle")]

    lang = _evaluate_one("v2+lang", sample, candidates, None)
    ctx = _evaluate_one("v2+lang+ctx", sample, candidates, None)

    assert lang["clarified"]
    assert (ctx["outcome"], ctx["pred_label"]) == ("정답", "bottle")


def test_summarize_counts_hand_crops_apart_from_accuracy():
    """손 크롭 장수 · 그중 담긴 장수를 따로 세고, 정답률 · 확신 오답률에는 넣지 않는다."""
    hand = {"outcome": "손 방향 크롭", "correct": False, "answered": False, "pred_label": None}
    rows = [
        _row(),
        _row(**hand, hand_crop=True),  # 정답을 담음
        _row(**hand, hand_crop=True, crop_contained=False),  # 빗나감
    ]

    m = run_ablation.summarize(rows)

    assert (m["hand_crop_count"], m["hand_crop_contained"]) == (2, 1)
    assert m["outcomes"]["손 방향 크롭"] == 2
    assert m["accuracy"] == round(1 / 3, 3)
    assert m["answered_accuracy"] == 1.0  # 박스로 답한 1장만
    assert m["confident_error_rate"] == 0.0
    assert m["crop_contained_rate"] == round(2 / 3, 3)  # 크롭 담김에는 손 크롭도 들어간다
    assert "| 2(1) |" in run_ablation.format_row("w", "v2+C", m, 3, None)


# --- 지표 ---------------------------------------------------------------------------


def test_summarize_rates_and_outcome_counts():
    """정답률 · 답한 것 정답률 · 되묻기율 · 손 있음/없음 · 분류 개수를 낸다."""
    rows = [
        _row(),
        _row(has_hand=True, correct=False, outcome="다른 후보 1위", pred_label="bottle"),
        _row(answered=False, clarified=True, correct=False, outcome="불필요한 되묻기"),
        _row(candidate_hit=False, correct=False, outcome="후보 없는데 답함"),
    ]

    m = run_ablation.summarize(rows)

    assert m["accuracy"] == 0.25
    assert m["answered_accuracy"] == round(1 / 3, 3)  # 고른 3장 중 1장
    assert m["reask_rate"] == 0.25
    assert m["candidate_recall"] == 0.75
    assert (m["accuracy_with_hand"], m["accuracy_without_hand"]) == (0.0, round(1 / 3, 3))
    assert m["outcomes"]["후보 없는데 답함"] == 1
    assert set(m["outcomes"]) == set(run_ablation.OUTCOMES)


def test_summarize_without_hand_photos_is_none():
    """손 없는 사진이 0장이면 그 정답률은 0 이 아니라 None 이다."""
    m = run_ablation.summarize([_row(has_hand=True)])
    assert m["accuracy_without_hand"] is None


def test_summarize_confident_error_and_crop_contained_use_all_photos():
    """확신 오답률 · 크롭 담김률의 분모는 전체 장수다. 되물은 사진도 분모에 들어간다."""
    rows = [
        _row(),
        _row(correct=False, confident_error=True, crop_contained=False, outcome="다른 후보 1위"),
        _row(
            answered=False,
            clarified=True,
            correct=False,
            crop_covered=None,
            crop_area=None,
            crop_contained=False,
            outcome="불필요한 되묻기",
        ),
        _row(candidate_hit=False, correct=False, confident_error=True, outcome="후보 없는데 답함"),
    ]

    m = run_ablation.summarize(rows)

    assert m["confident_error_rate"] == 0.5  # 4장 중 2장(되물은 1장도 분모)
    assert m["crop_contained_rate"] == 0.5  # 맞힌 1장 + 후보 없는데 담은 1장
    assert m["high_conf_error_rate"] == 0.0  # 옛 지표는 그대로 남는다


# --- DB 기록 · 비교 -----------------------------------------------------------------


def test_save_run_writes_run_and_samples(store):
    """eval_run 한 줄과 사진 수만큼의 eval_sample 이 남는다."""
    rows = [_row(), _row(correct=False, pred_label=None, outcome="후보 없어 되물음")]
    metrics = run_ablation.summarize(rows)

    run_id = run_ablation.save_run(store, "v1", rows, metrics, "기준선")

    with store.session() as db:
        run = db.get(EvalRun, run_id)
        samples = db.query(EvalSample).filter_by(run_id=run_id).all()
        assert (run.variant, run.sample_count, run.note) == ("v1", 2, "기준선")
        assert run.metrics["accuracy"] == 0.5
        assert sorted(s.is_correct for s in samples) == [False, True]


def _conditions(sha: str = "aaaaaaaaaaaa", sessions=(1, 2), commit: str = "c1") -> dict:
    return {
        "commit": commit,
        "weights": {"name": "yolov8n.pt", "sha256": sha},
        "sessions": list(sessions),
        "conf": 0.25,
    }


def _save(store, variant: str, accuracy: float, conditions: dict | None) -> int:
    metrics = {"accuracy": accuracy}
    if conditions is not None:
        metrics["conditions"] = conditions
    return run_ablation.save_run(store, variant, [_row()], metrics, "")


def test_previous_metrics_is_same_variant_before(store):
    """같은 버전 · 같은 조건의 바로 이전 실행 지표를 찾는다. 다른 버전 · 첫 실행이면 None."""
    first = _save(store, "v1", 0.6, _conditions())
    _save(store, "v2", 0.9, _conditions())
    second = _save(store, "v1", 0.7, _conditions())

    assert run_ablation.previous_metrics(store, "v1", second, _conditions())["accuracy"] == 0.6
    assert run_ablation.previous_metrics(store, "v1", first, _conditions()) is None


def test_previous_metrics_skips_other_weights(store):
    """가중치가 다른 실행과는 비교하지 않는다. 그 차이는 코드 몫인지 가중치 몫인지 모른다."""
    _save(store, "v1", 0.6, _conditions(sha="r0r0r0r0r0r0"))
    _save(store, "v1", 0.8, _conditions(sha="r1r1r1r1r1r1"))
    latest = _save(store, "v1", 0.65, _conditions(sha="r0r0r0r0r0r0"))

    previous = run_ablation.previous_metrics(store, "v1", latest, _conditions(sha="r0r0r0r0r0r0"))

    assert previous["accuracy"] == 0.6  # 바로 앞(R1, 0.8)을 건너뛰고 같은 R0 실행


def test_previous_metrics_skips_other_sessions_and_old_runs(store):
    """세션이 다르거나, 조건 기록이 없는 옛 실행(run 1~3)과는 비교하지 않는다."""
    _save(store, "v1", 0.5, None)  # 옛 실행
    _save(store, "v1", 0.9, _conditions(sessions=(4,)))  # 말 · 맥락 세션
    latest = _save(store, "v1", 0.6, _conditions())

    assert run_ablation.previous_metrics(store, "v1", latest, _conditions()) is None


def test_previous_metrics_ignores_commit(store):
    """커밋이 달라도 같은 가중치 · 세션이면 비교한다. 코드를 고친 효과를 보려는 비교다."""
    _save(store, "v1", 0.6, _conditions(commit="old"))
    latest = _save(store, "v1", 0.7, _conditions(commit="new+dirty"))

    previous = run_ablation.previous_metrics(store, "v1", latest, _conditions(commit="new"))
    assert previous["accuracy"] == 0.6


def _two_rows_metrics() -> dict:
    """맞힌 1장(크롭 담김) + 골랐는데 틀린 1장(크롭 못 담음).

    정답률 · 확신 오답률 · 크롭 담김이 모두 50% 다.
    """
    wrong = _row(correct=False, outcome="다른 후보 1위", confident_error=True, crop_contained=False)
    return run_ablation.summarize([_row(), wrong])


def test_format_row_shows_delta_for_accuracy_and_confident_error():
    """맨 앞에 가중치, 정답률 · 확신 오답률을 붙여 두고 각각 이전 실행 대비 차이(%p),
    후보 재현율 뒤에 크롭 담김 · 손 크롭(담김), 끝에 대상 결정 p50/p95."""
    previous = {"accuracy": 0.4, "confident_error_rate": 0.6}

    line = run_ablation.format_row("yolov8n.pt(abc123)", "v1", _two_rows_metrics(), 2, previous)

    assert line.startswith("yolov8n.pt(abc123) | v1 | 2 | 50.0 (+10.0) | 50.0 (-10.0) |")
    # 후보 재현율 · 크롭 담김 · 손 크롭(담김) · p50/p95
    assert line.endswith("| 100.0 | 50.0 | 0(0) | 0.20/0.20")


def test_format_row_skips_delta_missing_in_old_run():
    """확신 오답률이 없던 옛 실행(run 4~7)과는 그 칸의 차이를 찍지 않는다.

    없는 값을 0 으로 빼면 엉뚱한 +50.0 이 된다.
    """
    line = run_ablation.format_row("w", "v1", _two_rows_metrics(), 2, {"accuracy": 0.4})
    assert "| 50.0 (+10.0) | 50.0 | " in line


def test_format_row_without_previous_has_no_delta():
    line = run_ablation.format_row("w", "v1", _two_rows_metrics(), 2, None)
    assert line.startswith("w | v1 | 2 | 50.0 | 50.0 |")


# --- 실행 조건 · 가중치 ----------------------------------------------------------------


def test_weights_fingerprint_is_name_and_content_hash(tmp_path):
    """이름과 내용 sha256 앞 12자리. 이름이 같아도(best.pt) 내용이 다르면 지문이 다르다."""
    r1, r2 = tmp_path / "r1" / "best.pt", tmp_path / "r2" / "best.pt"
    for path, content in ((r1, b"round one"), (r2, b"round two")):
        path.parent.mkdir()
        path.write_bytes(content)

    fp1, fp2 = run_ablation.weights_fingerprint(r1), run_ablation.weights_fingerprint(r2)

    assert fp1 == {"name": "best.pt", "sha256": hashlib.sha256(b"round one").hexdigest()[:12]}
    assert fp1["sha256"] != fp2["sha256"]


def test_weights_label_shows_name_and_short_hash():
    """표에는 이름 + 지문 앞 6자리를 찍는다."""
    label = run_ablation.weights_label({"name": "detector_r1.pt", "sha256": "3fa2c1d4e5f6"})
    assert label == "detector_r1.pt(3fa2c1)"


def test_missing_weights_only_checks_paths(tmp_path):
    """폴더까지 적었는데 없는 파일만 잡는다. 이름만 적은 yolov8n.pt 는 ultralytics 가 받아 온다."""
    there = tmp_path / "detector_r1.pt"
    there.write_bytes(b"x")
    absent = str(tmp_path / "detector_r2.pt")

    assert run_ablation.missing_weights(["yolov8n.pt", str(there), absent]) == [absent]


def test_run_conditions_records_what_was_measured(monkeypatch):
    """커밋 · 가중치 · 정렬한 세션 · conf · 장비 · ultralytics 버전을 남긴다."""
    monkeypatch.setattr(run_ablation, "git_commit", lambda: "abc1234")
    weights = {"name": "yolov8n.pt", "sha256": "aaaaaaaaaaaa"}

    c = run_ablation.run_conditions(weights, [2, 1], 0.25)

    assert (c["commit"], c["weights"], c["conf"]) == ("abc1234", weights, 0.25)
    assert c["sessions"] == [1, 2]
    assert c["device"].strip()
    assert c["ultralytics"]


def _fake_git(monkeypatch, *, head: str = "abc1234\n", status: str = "", code: int = 0) -> None:
    def run(args, **kwargs):
        out = head if "rev-parse" in args else status
        return subprocess.CompletedProcess(args, code, stdout=out, stderr="")

    monkeypatch.setattr(run_ablation.subprocess, "run", run)


def test_git_commit_marks_uncommitted_changes(monkeypatch):
    """커밋 안 한 수정이 있으면 +dirty. 같은 번호라도 코드가 달랐다는 표시다."""
    _fake_git(monkeypatch, status=" M scripts/run_ablation.py\n")
    assert run_ablation.git_commit() == "abc1234+dirty"


def test_git_commit_clean(monkeypatch):
    """수정이 없으면 커밋 번호만."""
    _fake_git(monkeypatch)
    assert run_ablation.git_commit() == "abc1234"


def test_git_commit_unknown_without_git(monkeypatch):
    """git 이 없거나(PATH) 저장소가 아니면 멈추지 않고 unknown 으로 남긴다."""

    def no_git(args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(run_ablation.subprocess, "run", no_git)
    assert run_ablation.git_commit() == "unknown"

    _fake_git(monkeypatch, code=128)
    assert run_ablation.git_commit() == "unknown"


def test_package_version_missing_is_none():
    """설치 안 된 패키지는 멈추지 않고 none. CI 에는 ultralytics 가 없다."""
    assert run_ablation.package_version("jarviseo-no-such-package") == "none"


@pytest.mark.parametrize(
    ("old", "expected"),
    [
        (_conditions(), True),
        (_conditions(commit="other"), True),
        (_conditions(sha="bbbbbbbbbbbb"), False),
        (_conditions(sessions=(4,)), False),
        (None, False),
    ],
)
def test_same_conditions_is_weights_and_sessions(old, expected):
    """같은 조건 = 가중치 지문 · 세션이 같음. 커밋은 안 보고, 조건 기록이 없으면 다른 조건."""
    assert run_ablation.same_conditions(old, _conditions()) is expected


def test_hit_changes_lists_gained_and_lost_photos():
    """기준 대비 새로 잡힌 사진과 새로 놓친 사진을 따로 센다.

    재현율 숫자만으론 둘이 상쇄돼 안 보인다.
    """
    base = {"s01/a.jpg": True, "s01/b.jpg": False, "s01/c.jpg": True, "s02/d.jpg": False}
    other = {"s01/a.jpg": True, "s01/b.jpg": True, "s01/c.jpg": False, "s02/d.jpg": True}

    gained, lost = run_ablation.hit_changes(base, other)

    assert (gained, lost) == (["s01/b.jpg", "s02/d.jpg"], ["s01/c.jpg"])


def test_write_csv_names_file_by_run(tmp_path):
    """사진별 CSV 는 run_id_버전.csv 이고 한 줄에 한 사진이다."""
    path = run_ablation.write_csv(tmp_path / "runs", 7, "v1", [_row(), _row(file="s01/b.jpg")])

    assert path.name == "7_v1.csv"
    with path.open(encoding="utf-8", newline="") as f:
        assert [r["file"] for r in csv.DictReader(f)] == ["s01/a.jpg", "s01/b.jpg"]
