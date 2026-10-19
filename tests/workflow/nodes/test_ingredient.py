"""성분 질문이 VLM 을 거치지 않고 성분 판정 문장으로 끝나는지.

판정 자체(바코드·OCR·동의어)는 nutrition 쪽 테스트가 본다. 여기서는 그래프가 크롭과
사용자 알레르기를 넘기고, 돌아온 문장을 그대로 읽고 기록하는지만 본다.
CI 에는 EasyOCR 가 없어서 판정기는 이 파일 안의 작은 가짜로 바꿔 끼운다.
"""

import time

import numpy as np

from jarviseo.capture import FrameBuffer
from jarviseo.memory import TurnIngredient
from jarviseo.nutrition.service import IngredientCheck
from jarviseo.types import (
    AllergenJudgement,
    AllergenVerdict,
    BBox,
    Detection,
    Frame,
    Intent,
    TargetCandidate,
    TargetResolution,
)
from jarviseo.vlm import FakeVLMClient
from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech
from jarviseo.workflow import GraphDeps, build_graph
from jarviseo.workflow.messages import INGREDIENT_UNAVAILABLE

FULL = (1080, 1920, 3)
# 아래 과자 박스 (400,500)-(700,900) 300×400 에 사방 15% 여백. 성분 판정은 크기를 줄이지 않는다
CROP = (520, 390, 3)
MESSAGE = "땅콩이 들어 있어요. 땅콩 알레르기가 있으면 드시지 마세요."
SNACK = Detection("bottle", 0.9, BBox(400, 500, 700, 900))


class _Detector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        return [SNACK]


def _resolve(candidates, **kwargs) -> TargetResolution:
    chosen = TargetCandidate(candidates[0], total_score=1.0)
    return TargetResolution([chosen], chosen, margin=1.0, needs_clarify=False)


class _Checker:
    """IngredientChecker.check(crop, user_allergens) 와 모양이 같다. 받은 것을 모아 둔다."""

    def __init__(self, message: str = MESSAGE) -> None:
        self.message = message
        self.calls: list[tuple[tuple, list[str]]] = []

    def check(self, crop: np.ndarray, user_allergens: list[str]) -> IngredientCheck:
        self.calls.append((crop.shape, list(user_allergens)))
        judgement = AllergenJudgement(
            verdict=AllergenVerdict.CONTAINS, matched_terms=["땅콩"], matched_allergens=["땅콩"]
        )
        return IngredientCheck(judgement=judgement, message=self.message)


def _buffer() -> FrameBuffer:
    frames = FrameBuffer()
    image = np.zeros(FULL, np.uint8)
    frames.push(Frame(image=image, timestamp=time.monotonic(), source_id="test"))
    return frames


def _deps(**kwargs) -> GraphDeps:
    parts = {
        "stt": FakeSpeechToText(["자비서, 이거 먹어도 돼?"]),
        "tts": FakeTextToSpeech(),
        "vlm": FakeVLMClient(["VLM 이 지어낸 답"]),
        "frames": _buffer(),
        "detector": _Detector(),
        "resolve": _resolve,
        "checker": _Checker(),
    }
    return GraphDeps(**(parts | kwargs))


def test_ingredient_question_judges_crop_without_vlm(store):
    """성분 질문은 크롭과 알레르기를 넘겨 판정하고 VLM 은 부르지 않는다."""
    user_id = store.ensure_user()
    store.set_user_allergens(user_id, ["땅콩"])
    turn = {"session_id": store.start_session(user_id), "user_id": user_id}
    deps = _deps(store=store)

    result = build_graph(deps).invoke(turn)

    assert result["intent"] is Intent.INGREDIENT
    assert deps.checker.calls == [(CROP, ["땅콩"])]  # 전체 프레임이 아니라 고른 물건의 크롭
    assert deps.vlm.asked == []  # 알레르기 판정을 VLM 이 지어내면 안 된다
    assert deps.tts.spoken == [MESSAGE]
    assert result["response"].allergen.verdict is AllergenVerdict.CONTAINS
    assert "ingredient" in result["latency_ms"]
    with store.session() as db:
        row = db.get(TurnIngredient, result["turn_id"])
        assert (row.verdict, row.response_text) == ("contains", MESSAGE)


def test_judges_whole_photo_without_pointing_parts():
    """가리킬 부품이 없으면 사진 전체로 판정한다."""
    deps = _deps(detector=None, resolve=None)

    result = build_graph(deps).invoke({})

    assert result["intent"] is Intent.INGREDIENT
    assert deps.checker.calls == [(FULL, [])]  # 기록 저장소가 없으면 알레르기 목록도 비어 있다
    assert deps.vlm.asked == []


def test_no_checker_says_unavailable_without_vlm():
    """판정기가 없으면 확인할 수 없다고 하고 VLM 은 부르지 않는다."""
    deps = _deps(checker=None)

    build_graph(deps).invoke({})

    assert deps.tts.spoken == [INGREDIENT_UNAVAILABLE]
    assert deps.vlm.asked == []


def test_no_photo_says_unavailable():
    """사진이 없으면 확인할 수 없다고 한다."""
    deps = _deps(frames=None)

    build_graph(deps).invoke({})

    assert deps.tts.spoken == [INGREDIENT_UNAVAILABLE]
    assert deps.checker.calls == []


def test_retake_notice_is_read_and_ends_turn():
    """재촬영 안내가 오면 그 문장을 읽고 턴을 끝낸다."""
    retake = "성분표가 잘 안 보여요. 조금 더 가까이 비춰 주세요."
    deps = _deps(checker=_Checker(retake))

    result = build_graph(deps).invoke({})

    assert deps.tts.spoken == [retake]
    assert len(deps.checker.calls) == 1
    assert "recapture_count" not in result
