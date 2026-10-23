"""대상 결정 — 순위·margin·되묻기 문장, 되묻기 대답으로 후보 좁히기, 단서 스위치, 손 단서 기권."""

import pytest

from jarviseo.pointing.hands import HandPose
from jarviseo.pointing.resolver import resolve_target
from jarviseo.types import BBox, CueKind, Detection

FRAME = (1920, 1080)

# 화면 가운데(960, 540)를 두고 좌우 대칭이라 가운데 단서 점수가 같다.
LEFT = BBox(400, 440, 600, 640)
RIGHT = BBox(1320, 440, 1520, 640)
MIDDLE = BBox(860, 440, 1060, 640)
CORNER = BBox(0, 0, 200, 200)


def _det(label: str, bbox: BBox) -> Detection:
    return Detection(label=label, confidence=0.9, bbox=bbox)


def _hand(base: tuple[float, float], tip: tuple[float, float]) -> HandPose:
    """검지 뿌리(5) → 끝(8)만 의미 있는 손. 방향만 쓰므로 3D 관절은 0 으로 둔다."""
    landmarks = [(0.0, 0.0)] * 21
    landmarks[5], landmarks[8] = base, tip
    return HandPose(tuple(landmarks), ((0.0, 0.0, 0.0),) * 21, "Right", 0.9)


# 화면 아래에서 RIGHT 중심(1420, 540)을 똑바로 가리키는 손. LEFT(500, 540)는 100° 넘게 벗어난다.
TO_RIGHT = _hand(base=(1068.0, 936.0), tip=(1100.0, 900.0))


def _resolve(candidates, replies=(), hand=None, **kwargs):
    return resolve_target(
        candidates=candidates,
        hand=hand,
        gaze_vector=None,
        text="저거 뭐야?",
        frame_size=FRAME,
        replies=replies,
        **kwargs,
    )


# --- 순위 · margin --------------------------------------------------------------


def test_no_candidates_asks_without_question():
    """후보가 없으면 고르지 않고, 되물을 문장도 없다(그래프가 다시 가리켜 달라고 한다)."""
    result = _resolve([])

    assert result.chosen is None
    assert result.needs_clarify
    assert result.margin == 0.0
    assert result.clarify_question is None


def test_single_candidate_is_chosen_with_full_margin():
    """후보가 하나면 비교 상대가 없으므로 margin 1.0 으로 바로 고른다."""
    cup = _det("cup", CORNER)

    result = _resolve([cup])

    assert result.chosen.detection is cup
    assert result.margin == 1.0
    assert not result.needs_clarify


def test_clear_winner_is_chosen_and_ranked_first():
    """가운데 물건이 구석 물건을 넉넉히 이기면 되묻지 않고 고른다."""
    corner, middle = _det("bottle", CORNER), _det("cup", MIDDLE)

    result = _resolve([corner, middle])

    assert result.chosen.detection is middle
    assert [c.detection for c in result.candidates] == [middle, corner]
    assert result.margin >= 0.15
    assert not result.needs_clarify


def test_score_is_normalized_by_enabled_weights():
    """가운데 단서 하나만 켜도 점수가 0~0.1 이 아니라 0~1 이다(가중치 0.10 으로 나눔)."""
    full = _det("cup", BBox(0, 0, *FRAME))

    [candidate] = _resolve([full]).candidates

    assert candidate.total_score == pytest.approx(1.0)


def test_only_enabled_cues_are_recorded():
    """DB cue_scores 에는 켠 단서만 남는다. v1 은 center 하나."""
    [candidate] = _resolve([_det("cup", MIDDLE)]).candidates
    assert [s.kind for s in candidate.cue_scores] == [CueKind.CENTER]


def test_margin_threshold_is_respected():
    """기준을 높이면 넉넉한 차이도 되묻는다."""
    result = _resolve([_det("bottle", CORNER), _det("cup", MIDDLE)], margin_threshold=0.99)
    assert result.needs_clarify
    assert result.chosen is None


# --- 되묻기 문장 ------------------------------------------------------------------


def test_same_kind_tie_asks_by_position():
    """같은 컵 둘이 비슷하면 위치로 묻는다."""
    result = _resolve([_det("cup", LEFT), _det("cup", RIGHT)])

    assert result.needs_clarify
    assert result.chosen is None
    assert result.clarify_question == "왼쪽 컵이요, 오른쪽 컵이요?"


def test_different_kind_tie_asks_by_name_with_particle():
    """종류가 다르면 이름으로 묻는다. 받침이 있으면 '이요', 없으면 '요'."""
    result = _resolve([_det("cup", LEFT), _det("apple", RIGHT)])
    assert result.clarify_question == "컵이요, 사과요?"


def test_unknown_label_question_falls_back_to_english():
    """한글 이름이 없는 라벨도 문장이 빈칸 없이 나온다."""
    result = _resolve([_det("glasses", LEFT), _det("cup", RIGHT)])
    assert result.clarify_question == "glasses요, 컵이요?"


def test_product_question_reads_food():
    """학습한 가중치의 product 는 영어가 아니라 "식품"으로 되묻는다."""
    result = _resolve([_det("product", LEFT), _det("cup", RIGHT)])
    assert result.clarify_question == "식품이요, 컵이요?"


# --- 되묻기 대답으로 좁히기 ---------------------------------------------------------


@pytest.mark.parametrize(("reply", "expected"), [("왼쪽", LEFT), ("오른쪽 거", RIGHT)])
def test_position_reply_picks_that_side(reply, expected):
    """'왼쪽'·'오른쪽 거'라고 대답하면 그쪽 컵을 margin 1.0 으로 고른다."""
    result = _resolve([_det("cup", LEFT), _det("cup", RIGHT)], replies=[reply])

    assert result.chosen.detection.bbox == expected
    assert result.margin == 1.0
    assert len(result.candidates) == 1


def test_name_reply_picks_that_kind():
    """'사과'라고 대답하면 사과를 고른다."""
    result = _resolve([_det("cup", LEFT), _det("apple", RIGHT)], replies=["사과요"])
    assert result.chosen.detection.label == "apple"


def test_name_reply_ignores_spaces_and_prefers_longest_name():
    """'곰인형'은 곰이 아니라 곰 인형이다. 띄어 말해도, 붙여 말해도 같다."""
    candidates = [_det("bear", LEFT), _det("teddy bear", RIGHT)]

    result = _resolve(candidates, replies=["곰인형"])

    assert result.chosen.detection.label == "teddy bear"


def test_unmatched_reply_keeps_asking():
    """대답이 어느 후보에도 안 맞으면(받아쓰기 실패 등) 그대로 다시 묻는다."""
    result = _resolve([_det("cup", LEFT), _det("cup", RIGHT)], replies=["글쎄"])

    assert result.needs_clarify
    assert len(result.candidates) == 2


def test_position_reply_chooses_among_asked_pair():
    """컵 셋 중 가운데 쪽 두 개를 물었으면, '왼쪽'은 맨 왼쪽 컵이 아니라 물은 둘 중 왼쪽이다."""
    far_left = _det("cup", BBox(0, 440, 200, 640))  # 구석이라 점수가 낮아 질문에 안 들어간다
    near_left = _det("cup", BBox(700, 440, 900, 640))
    near_right = _det("cup", BBox(1020, 440, 1220, 640))
    candidates = [far_left, near_left, near_right]
    assert _resolve(candidates).clarify_question == "왼쪽 컵이요, 오른쪽 컵이요?"

    result = _resolve(candidates, replies=["왼쪽"])

    assert result.chosen.detection is near_left


def test_replies_narrow_step_by_step():
    """두 번째 대답은 첫 번째 대답으로 좁힌 후보에서 다시 좁힌다."""
    candidates = [_det("cup", LEFT), _det("cup", RIGHT), _det("apple", CORNER)]

    first = _resolve(candidates, replies=["컵"])
    second = _resolve(candidates, replies=["컵", "오른쪽"])

    assert first.needs_clarify  # 컵 둘이 남아 아직 비슷하다
    assert [c.detection.label for c in first.candidates] == ["cup", "cup"]
    assert second.chosen.detection.bbox == RIGHT


# --- 단서 스위치 -------------------------------------------------------------------


def test_enabled_cues_center_matches_default():
    """center 만 켠 결과가 기본값(구현된 단서 전부)과 같다."""
    candidates = [_det("bottle", CORNER), _det("cup", MIDDLE)]

    default = _resolve(candidates)
    center_only = _resolve(candidates, enabled_cues={CueKind.CENTER})

    assert center_only.margin == pytest.approx(default.margin)
    assert center_only.chosen.detection is default.chosen.detection


@pytest.mark.parametrize("cues", [{CueKind.CENTER, CueKind.GAZE}, set()])
def test_unimplemented_or_empty_cues_raise(cues):
    """구현 안 된 단서나 빈 스위치는 조용히 넘기지 않고 멈춘다(ablation 숫자가 거짓이 된다)."""
    with pytest.raises(ValueError):
        _resolve([_det("cup", MIDDLE)], enabled_cues=cues)


# --- 손 단서 (v2) -------------------------------------------------------------------


def test_hand_breaks_symmetric_tie():
    """가운데 단서로는 비기는 좌우 대칭 두 컵도, 손이 가리킨 쪽을 되묻지 않고 고른다."""
    left, right = _det("cup", LEFT), _det("cup", RIGHT)

    result = _resolve([left, right], hand=TO_RIGHT)

    assert result.chosen.detection is right
    assert not result.needs_clarify
    assert [s.kind for s in result.chosen.cue_scores] == [CueKind.CENTER, CueKind.POINT]


def test_no_hand_abstains_like_v1():
    """손이 없으면 손 단서를 켜도 0점이 아니라 기권이다. 결과가 가운데 단서만 켠 것과 같다."""
    candidates = [_det("bottle", CORNER), _det("cup", MIDDLE)]

    v2 = _resolve(candidates, enabled_cues={CueKind.CENTER, CueKind.POINT})
    v1 = _resolve(candidates, enabled_cues={CueKind.CENTER})

    assert v2.margin == pytest.approx(v1.margin)
    assert v2.chosen.detection is v1.chosen.detection
    assert [s.kind for s in v2.chosen.cue_scores] == [CueKind.CENTER]  # point 칸 없음


def test_hand_without_direction_abstains():
    """검지 뿌리와 끝이 겹쳐 방향이 없는 손은 손이 없는 것과 같다."""
    left, right = _det("cup", LEFT), _det("cup", RIGHT)
    flat = _hand(base=(1100.0, 900.0), tip=(1100.0, 900.0))

    result = _resolve([left, right], hand=flat)

    assert result.needs_clarify  # 가운데 단서만 남아 다시 비긴다
    assert [s.kind for s in result.candidates[0].cue_scores] == [CueKind.CENTER]


def test_hand_ignored_when_point_switched_off():
    """손이 있어도 v1(가운데만)이면 손 단서를 쓰지 않는다(ablation 의 v1 줄이 그대로)."""
    left, right = _det("cup", LEFT), _det("cup", RIGHT)

    result = _resolve([left, right], hand=TO_RIGHT, enabled_cues={CueKind.CENTER})

    assert result.needs_clarify
    assert [s.kind for s in result.candidates[0].cue_scores] == [CueKind.CENTER]


def test_point_only_without_hand_gives_up():
    """손 단서만 켰는데 손이 없으면 고를 근거가 없다. 후보 0개처럼 다시 가리켜 달라고 한다."""
    result = _resolve([_det("cup", MIDDLE)], enabled_cues={CueKind.POINT})

    assert result.candidates == []
    assert result.chosen is None
    assert result.needs_clarify
    assert result.clarify_question is None


def test_reply_narrowing_keeps_hand():
    """되묻기 대답으로 다시 고를 때도 손 단서를 쓴다. 손이 가리킨 쪽 이름을 대면 그걸 고른다."""
    left, right, corner = _det("cup", LEFT), _det("bottle", RIGHT), _det("book", CORNER)

    result = _resolve([left, right, corner], replies=["병"], hand=TO_RIGHT)

    assert result.chosen.detection is right
    assert result.chosen.score_of(CueKind.POINT) > 0.9
