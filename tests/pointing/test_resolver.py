"""대상 결정 — 순위·margin·되묻기 문장, 후보 하나의 총점 하한, 되묻기 대답으로 후보 좁히기,
단서 스위치, 손 · 말 · 이어 묻기 단서 기권, 말 · 이어 묻기 단서 플래그."""

import numpy as np
import pytest

from jarviseo import config
from jarviseo.pointing.hands import HandPose
from jarviseo.pointing.resolver import MIN_SINGLE_SCORE, resolve_target
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
    """후보가 하나면 비교 상대가 없으므로 margin 1.0 으로 바로 고른다(총점이 하한 이상일 때)."""
    cup = _det("cup", MIDDLE)

    result = _resolve([cup])

    assert result.chosen.detection is cup
    assert result.margin == 1.0
    assert not result.needs_clarify


def test_lone_corner_candidate_is_not_asserted():
    """후보가 구석 컵 하나뿐이면(총점 0.14 < 0.3) 단정하지 않는다. 되물을 문장도 없다(#215).

    천장을 가리켰는데 구석 모니터를 답한 턴 23 과 같은 장면. 후보 순위는 기록용으로 남는다.
    """
    cup = _det("cup", CORNER)

    result = _resolve([cup])

    assert result.chosen is None
    assert result.needs_clarify
    assert result.clarify_question is None
    assert [c.detection for c in result.candidates] == [cup]
    assert result.candidates[0].total_score < MIN_SINGLE_SCORE


def test_lone_corner_candidate_pointed_at_is_chosen():
    """구석 컵 하나라도 손이 그 컵을 가리키면 손 점수가 총점을 올려 고른다.

    하한은 손 없는 사진에서만 일한다.
    """
    cup = _det("cup", CORNER)  # 중심 (100, 100)
    to_corner = _hand(base=(340.0, 340.0), tip=(300.0, 300.0))  # 왼쪽 위로 똑바로

    result = _resolve([cup], hand=to_corner)

    assert result.chosen.detection is cup
    assert result.chosen.total_score >= MIN_SINGLE_SCORE


def test_reply_narrowed_to_one_skips_the_floor():
    """대답으로 구석 컵 하나로 좁혀졌으면 사용자가 고른 것이라 하한을 보지 않고 고른다."""
    cup, apple = _det("cup", CORNER), _det("apple", MIDDLE)

    result = _resolve([cup, apple], replies=["컵"])

    assert result.chosen.detection is cup
    assert result.chosen.total_score < MIN_SINGLE_SCORE  # 처음부터 하나였다면 단정 안 했을 점수


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


# --- 종류 낱말 대답 (labels.KIND_WORDS) -------------------------------------------------


def test_alias_reply_picks_that_kind():
    """ "텀블러" 대답이 병 후보에 걸린다(모의 루프 턴 35 에서 안 걸렸다)."""
    candidates = [_det("bottle", LEFT), _det("cup", RIGHT)]
    result = _resolve(candidates, replies=["텀블러"])
    assert result.chosen.detection.label == "bottle"


def test_reply_prefers_longest_kind_word():
    """ "컵라면"은 "컵"을 품고 있지만 긴 낱말이 이겨 식품을 고른다."""
    candidates = [_det("product", LEFT), _det("cup", RIGHT)]
    result = _resolve(candidates, replies=["컵라면"])
    assert result.chosen.detection.label == "product"


# --- 말 단서 (v4) -------------------------------------------------------------------


def _ask(candidates, text, **kwargs):
    return resolve_target(
        candidates=candidates, hand=None, gaze_vector=None, text=text, frame_size=FRAME, **kwargs
    )


def _kinds(candidate) -> list[str]:
    return [s.kind.value for s in candidate.cue_scores]


def test_language_cue_is_off_by_default(monkeypatch):
    """플래그가 꺼져 있으면(기본) "왼쪽 컵"이어도 말 단서를 안 쓴다 — 지금 실제 비서 동작 그대로."""
    monkeypatch.setattr(config, "ENABLE_LANGUAGE_CUE", False)
    result = _ask([_det("cup", LEFT), _det("cup", RIGHT)], "왼쪽 컵 뭐야?")

    assert result.needs_clarify  # 대칭 컵 둘이라 v1 처럼 되묻는다
    assert all(_kinds(c) == ["center"] for c in result.candidates)


def test_language_cue_on_picks_left_cup_without_asking(monkeypatch):
    """플래그를 켜면 "왼쪽 컵"이 대칭 컵 둘의 되묻기를 없애고 왼쪽을 고른다."""
    monkeypatch.setattr(config, "ENABLE_LANGUAGE_CUE", True)
    result = _ask([_det("cup", LEFT), _det("cup", RIGHT)], "왼쪽 컵 뭐야?")

    assert result.chosen.detection.bbox == LEFT
    assert _kinds(result.chosen) == ["center", "lang"]


def test_ablation_switch_turns_language_cue_on_regardless_of_flag(monkeypatch):
    """ablation 은 단서를 직접 골라 켠다. 플래그는 실제 비서의 기본값만 정한다."""
    monkeypatch.setattr(config, "ENABLE_LANGUAGE_CUE", False)
    cups = [_det("cup", LEFT), _det("cup", RIGHT)]
    result = _ask(cups, "오른쪽 컵 뭐야?", enabled_cues={CueKind.CENTER, CueKind.LANG})

    assert result.chosen.detection.bbox == RIGHT


def test_language_cue_abstains_without_words():
    """쓸 말이 없으면 기권해서 v1(가운데만)과 점수 · margin 이 같다."""
    candidates = [_det("bottle", CORNER), _det("cup", MIDDLE)]
    center = _ask(candidates, "저거 뭐야?", enabled_cues={CueKind.CENTER})
    with_lang = _ask(candidates, "저거 뭐야?", enabled_cues={CueKind.CENTER, CueKind.LANG})

    assert with_lang.margin == pytest.approx(center.margin)
    assert _kinds(with_lang.chosen) == ["center"]


def test_color_word_needs_the_photo():
    """색 말은 사진이 있어야 쓴다. 사진이 없으면 기권(되묻기 그대로), 있으면 노란 쪽을 고른다."""
    cans = [_det("product", LEFT), _det("product", RIGHT)]
    lang_on = {CueKind.CENTER, CueKind.LANG}
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[440:640, 1320:1520] = (0, 255, 255)  # 오른쪽 캔만 노랗게(BGR)

    without_photo = _ask(cans, "노란 캔 뭐야?", enabled_cues=lang_on)
    with_photo = _ask(cans, "노란 캔 뭐야?", enabled_cues=lang_on, image=image)

    assert without_photo.needs_clarify
    assert with_photo.chosen.detection.bbox == RIGHT


def test_reply_narrowing_keeps_language_scores_from_question():
    """말 점수는 되묻기로 좁히기 전에 매긴다. "왼쪽 거"의 위치는 처음 장면 전체 기준으로 남는다."""
    candidates = [_det("bottle", LEFT), _det("cup", MIDDLE), _det("cup", RIGHT)]
    lang_on = {CueKind.CENTER, CueKind.LANG}

    result = _ask(candidates, "왼쪽 거 뭐야?", enabled_cues=lang_on, replies=["컵"])

    assert result.chosen.detection.bbox == MIDDLE  # 남은 컵 중 왼쪽
    lang = next(s for s in result.chosen.cue_scores if s.kind is CueKind.LANG)
    assert lang.score == pytest.approx(0.5)  # 좁힌 뒤 다시 매겼다면 1.0


# --- 이어 묻기 단서 (v5) ---------------------------------------------------------------


def test_context_cue_is_off_by_default(monkeypatch):
    """플래그가 꺼져 있으면(기본) 앞 대상이 있고 "그거"라고 해도 이어 묻기 단서를 안 쓴다."""
    monkeypatch.setattr(config, "ENABLE_CONTEXT_CUE", False)
    candidates = [_det("bottle", LEFT), _det("cup", RIGHT)]

    result = _ask(candidates, "그거 씻어야 돼?", prev_target_label="cup")

    assert result.needs_clarify  # 대칭이라 v1 처럼 되묻는다
    assert all(_kinds(c) == ["center"] for c in result.candidates)


def test_context_cue_on_picks_previous_target_without_asking(monkeypatch):
    """플래그를 켜면 "그거"가 앞에서 고른 컵을 가리킨다.

    대칭인 병 · 컵에서 되묻지 않고 컵을 고른다.
    """
    monkeypatch.setattr(config, "ENABLE_CONTEXT_CUE", True)
    candidates = [_det("bottle", LEFT), _det("cup", RIGHT)]

    result = _ask(candidates, "그거 씻어야 돼?", prev_target_label="cup")

    assert result.chosen.detection.label == "cup"
    assert _kinds(result.chosen) == ["center", "ctx"]


def test_ablation_switch_turns_context_cue_on_regardless_of_flag(monkeypatch):
    """ablation 은 단서를 직접 골라 켠다(정답표의 prev_label 로 잰다)."""
    monkeypatch.setattr(config, "ENABLE_CONTEXT_CUE", False)
    candidates = [_det("bottle", LEFT), _det("cup", RIGHT)]
    ctx_on = {CueKind.CENTER, CueKind.CTX}

    result = _ask(candidates, "그거 뭐야?", prev_target_label="bottle", enabled_cues=ctx_on)

    assert result.chosen.detection.label == "bottle"


def test_context_cue_abstains_without_back_reference():
    """되가리키는 말이 없으면 앞 대상이 있어도 기권해서 v1 과 margin 이 같다."""
    candidates = [_det("bottle", CORNER), _det("cup", MIDDLE)]
    ctx_on = {CueKind.CENTER, CueKind.CTX}
    center = _ask(candidates, "이 병 뭐야?", enabled_cues={CueKind.CENTER})
    with_ctx = _ask(candidates, "이 병 뭐야?", prev_target_label="cup", enabled_cues=ctx_on)

    assert with_ctx.margin == pytest.approx(center.margin)
    assert _kinds(with_ctx.chosen) == ["center"]
