"""성분 질문을 한 턴 안에서 이어 가기(api/ingredient_turn.py) — 가짜 부품으로 본다."""

import time

import numpy as np

from jarviseo.api.ingredient_turn import IngredientFollowUp
from jarviseo.capture import FrameBuffer
from jarviseo.nutrition.response import (
    RETAKE_AGAIN,
    RETAKE_GAVE_UP,
    RETAKE_SHOW_INGREDIENTS,
    RETAKE_SHOW_PANEL,
    RETAKE_SHOW_PANEL_AGAIN,
)
from jarviseo.nutrition.service import IngredientCheck
from jarviseo.types import (
    AllergenJudgement,
    AllergenVerdict,
    BBox,
    Detection,
    Frame,
    TargetCandidate,
    TargetResolution,
)
from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech

CROP = np.zeros((40, 40, 3), np.uint8)
PANEL = np.full((120, 160, 3), 200, np.uint8)  # "여기"라고 할 때 보여 준 장면
BOX = BBox(40, 20, 100, 80)  # 그 장면에서 가리킨 성분표(60×60)
SHOWN = (Detection("box", 0.9, BOX),)


def miss() -> IngredientCheck:
    j = AllergenJudgement(verdict=AllergenVerdict.UNDETERMINED, reason="no_ingredients")
    return IngredientCheck(
        judgement=j, message=RETAKE_SHOW_INGREDIENTS, retake=RETAKE_SHOW_INGREDIENTS
    )


def found() -> IngredientCheck:
    j = AllergenJudgement(verdict=AllergenVerdict.CONTAINS)
    return IngredientCheck(judgement=j, message="해당 제품에는 밀이 들어 있어요.")


class Scripted:
    """정해 둔 결과를 차례로 돌려주는 판정기. 받은 사진의 크기를 남긴다."""

    def __init__(self, *results):
        self.results, self.seen = list(results), []

    def check(self, image, allergens, *args, **kwargs):
        self.seen.append(image.shape)
        return self.results.pop(0)


class FakeDetector:
    def __init__(self, found):
        self.found = found

    def detect(self, image):
        return list(self.found)


def first_candidate(candidates, **kwargs):
    """resolve_target 대신. 첫 후보를 고른다."""
    ranked = [TargetCandidate(detection=d, total_score=1.0) for d in candidates]
    return TargetResolution(candidates=ranked, chosen=ranked[0], margin=1.0, needs_clarify=False)


def follow(checker, replies, active=None, found=SHOWN):
    frames = FrameBuffer(4)
    frames.push(Frame(image=PANEL, timestamp=time.monotonic(), source_id="t"))
    tts, chimes = FakeTextToSpeech(), []
    fu = IngredientFollowUp(
        checker, tts, FakeSpeechToText(replies), frames, start_chime=lambda: chimes.append(1),
        active=active, detector=FakeDetector(found), resolve=first_candidate,
    )  # fmt: skip
    return fu, tts, chimes


def test_reads_crop_without_asking():
    fu, tts, chimes = follow(Scripted(found()), [])
    assert fu.check(CROP, []).message == "해당 제품에는 밀이 들어 있어요."
    assert tts.spoken == [] and chimes == []


def test_show_panel_then_reads_it():
    """크롭에서 못 읽으면 보여 달라고 하고 시작음 뒤 '여기'를 듣고 그 장면을 잘라 다시 읽는다."""
    checker = Scripted(miss(), found())
    fu, tts, chimes = follow(checker, ["여기"])
    r = fu.check(CROP, [])
    assert r.message == "해당 제품에는 밀이 들어 있어요." and not r.retake
    assert tts.spoken == [RETAKE_SHOW_PANEL] and chimes == [1]
    # 두 번째는 보여 준 장면 전체(120×160)가 아니라 가리킨 박스를 여백 15% 넣어 자른 것(78×78)
    assert checker.seen == [CROP.shape, (78, 78, 3)]


def test_shown_scene_without_target_is_not_read_whole():
    """보여 준 장면에 후보가 없으면 장면 전체로 읽지 않고 못 읽은 것으로 보아 다음 단계로 간다."""
    checker = Scripted(miss())
    fu, tts, _ = follow(checker, ["여기", "여기"], found=())
    r = fu.check(CROP, [])
    assert r.message == RETAKE_GAVE_UP and checker.seen == [CROP.shape]
    assert tts.spoken == [RETAKE_SHOW_PANEL, RETAKE_SHOW_PANEL_AGAIN]


def test_asks_again_then_gives_up():
    """다시 보여 달라고 한 번 더 하고, 또 못 읽으면 '읽지 못했습니다'로 끝낸다."""
    checker = Scripted(miss(), miss(), miss())
    fu, tts, chimes = follow(checker, ["여기", "여기"])
    r = fu.check(CROP, [])
    assert tts.spoken == [RETAKE_SHOW_PANEL, RETAKE_SHOW_PANEL_AGAIN] and chimes == [1, 1]
    assert r.message == RETAKE_GAVE_UP and r.retake == RETAKE_GAVE_UP
    assert len(checker.seen) == 3


def test_no_reply_or_stop_ends_turn():
    fu, tts, _ = follow(Scripted(miss()), [])  # 대답 없음
    assert fu.check(CROP, []).message == RETAKE_GAVE_UP and tts.spoken == [RETAKE_SHOW_PANEL]
    fu, _, _ = follow(Scripted(miss()), ["그만할래"])
    assert fu.check(CROP, []).message == RETAKE_GAVE_UP


def test_typed_question_does_not_listen():
    """입력창 질문이면 듣지 않고, 잇달아 못 읽으면 두 번째부터 다시 찍어 달라고 한다."""
    fu, tts, chimes = follow(Scripted(miss(), miss()), ["여기"], active=lambda: False)
    assert fu.check(CROP, []).message == RETAKE_SHOW_INGREDIENTS
    assert fu.check(CROP, []).message == RETAKE_AGAIN
    assert tts.spoken == [] and chimes == []
