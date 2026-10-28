"""되물은 뒤 호출어 없이 이어 듣기(workflow/follow_up.py)와 그 턴을 도는 그래프.

듣기 루프(assistant.py · api/)는 follow_up_turn(state) 가 턴 정보를 주면 그 정보로 그래프를 한 번 더
부른다. 여기서는 그 두 턴을 차례로 부르는 것까지 본다.
"""

import time

import numpy as np
import pytest

from jarviseo.capture import FrameBuffer
from jarviseo.memory import SessionTurn
from jarviseo.nutrition.service import IngredientCheck
from jarviseo.types import AllergenJudgement, AllergenVerdict, Frame, Intent
from jarviseo.vlm import FakeVLMClient
from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech
from jarviseo.workflow import GraphDeps, build_graph
from jarviseo.workflow.follow_up import (
    FOLLOW_UP_TRIGGER,
    MAX_FOLLOW_UP,
    follow_up_question,
    follow_up_turn,
)
from jarviseo.workflow.messages import CLARIFY_GIVE_UP

RETAKE = "원재료명 부분을 보여 주세요."
FOUND = "땅콩이 들어 있어요. 땅콩 알레르기가 있으면 드시지 마세요."


class _Checker:
    """처음에는 성분표를 못 읽고(retake), 그다음에는 읽는다. 받은 사진을 모아 둔다."""

    def __init__(self, misses: int = 1) -> None:
        self.misses = misses
        self.images: list[np.ndarray] = []

    def check(self, image: np.ndarray, user_allergens: list[str]) -> IngredientCheck:
        self.images.append(image)
        if len(self.images) <= self.misses:
            judgement = AllergenJudgement(verdict=AllergenVerdict.UNDETERMINED)
            return IngredientCheck(judgement=judgement, message=RETAKE, retake=RETAKE)
        judgement = AllergenJudgement(
            verdict=AllergenVerdict.CONTAINS, matched_terms=["땅콩"], matched_allergens=["땅콩"]
        )
        return IngredientCheck(judgement=judgement, message=FOUND)


class _Camera(FrameBuffer):
    """pick_best 를 부를 때마다 새로 찍은 장면을 준다(값 = 몇 번째 장면인지)."""

    def __init__(self) -> None:
        super().__init__()
        self.shots = 0

    def pick_best(self, around=None, window=0.5):
        self.shots += 1
        image = np.full((48, 64, 3), self.shots, np.uint8)
        return Frame(image=image, timestamp=time.monotonic(), source_id="test")


def _turn(store) -> dict:
    user_id = store.ensure_user()
    return {"session_id": store.start_session(user_id), "user_id": user_id}


def test_retake_then_short_reply_judges_new_scene(store):
    """성분표를 다시 보여 달라고 한 뒤 "다시 확인해줘"만 말해도 앞 질문으로 새 장면을 판정한다."""
    chimed: list[str] = []
    deps = GraphDeps(
        stt=FakeSpeechToText(["자비서, 이거 먹어도 돼?", "다시 확인해줘"]),
        tts=FakeTextToSpeech(),
        vlm=FakeVLMClient(["VLM 이 지어낸 답"]),
        frames=_Camera(),
        checker=_Checker(),
        store=store,
        start_chime=lambda: chimed.append("start"),
    )
    graph = build_graph(deps)

    first = graph.invoke(_turn(store))
    follow = follow_up_turn(first)
    assert follow is not None
    second = graph.invoke(follow)

    assert deps.tts.spoken == [RETAKE, FOUND]
    assert chimed == ["start"]  # 이어 듣기 전에 이제 말하라는 시작음
    assert second["question"] == "이거 먹어도 돼? 다시 확인해줘"
    assert second["intent"] is Intent.INGREDIENT
    assert [int(img[0, 0, 0]) for img in deps.checker.images] == [1, 2]  # 다시 보여 준 장면
    assert deps.vlm.asked == []
    with store.session() as db:
        row = db.get(SessionTurn, second["turn_id"])
        # 기록 · 화면에는 사용자가 실제로 한 말이 남는다
        assert (row.trigger_type, row.question_text) == (FOLLOW_UP_TRIGGER, "다시 확인해줘")
        assert row.session_id == first["session_id"]
    assert follow_up_turn(second) is None  # 읽었으면 더 이어 듣지 않는다


def test_new_question_while_following_up_is_asked_as_is():
    """이어 듣는 중에 새로 물으면 앞 질문을 붙이지 않는다."""
    deps = GraphDeps(
        stt=FakeSpeechToText(["오늘 날씨 어때?"]),
        tts=FakeTextToSpeech(),
        vlm=FakeVLMClient(["맑아요."]),
    )

    result = build_graph(deps).invoke({"follow_up_of": "이거 먹어도 돼?"})

    assert result["question"] == "오늘 날씨 어때?"
    assert deps.vlm.asked[0][1] == "오늘 날씨 어때?"


def test_vlm_ask_back_reply_goes_with_previous_question():
    """VLM 이 되물은 말에 "빨간 컵"이라고 답하면 앞 질문과 같이 VLM 에 간다."""
    deps = GraphDeps(
        stt=FakeSpeechToText(["자비서, 이거 뭐야?", "빨간 컵"]),
        tts=FakeTextToSpeech(),
        vlm=FakeVLMClient(["어떤 물건을 말씀하시는지 알려 주시면 답해 드릴게요.", "머그컵이에요."]),
    )
    graph = build_graph(deps)

    second = graph.invoke(follow_up_turn(graph.invoke({})))

    assert deps.vlm.asked[1][1] == "이거 뭐야? 빨간 컵"
    assert second["response"].text == "머그컵이에요."


def test_silence_while_following_up_ends_without_log(store):
    """이어 듣는 동안 아무 말이 없으면 대답 · 기록 없이 끝난다."""
    tts = FakeTextToSpeech()
    deps = GraphDeps(stt=FakeSpeechToText([]), tts=tts, vlm=FakeVLMClient(), store=store)

    result = build_graph(deps).invoke({**_turn(store), "follow_up_of": "이거 먹어도 돼?"})

    assert "utterance" not in result and "turn_id" not in result
    assert tts.spoken == []


def test_call_while_following_up_is_next_turn():
    """이어 듣는 동안 "자비서"로 부르면 그 말은 새 질문이다. 듣기를 멈추고 끼어들기로 넘긴다."""
    deps = GraphDeps(
        stt=FakeSpeechToText(["자비서, 저건 뭐야?"]),
        tts=FakeTextToSpeech(),
        vlm=FakeVLMClient(),
        called_since=lambda t: True,
    )

    result = build_graph(deps).invoke({"follow_up_of": "이거 먹어도 돼?"})

    assert result["interrupted"] is True
    assert "utterance" not in result
    assert follow_up_turn(result) is None


def test_follow_up_stops_after_max_count():
    """성분표를 계속 못 읽어도 MAX_FOLLOW_UP 번 이어 들은 뒤에는 호출어 대기로 돌아간다."""
    deps = GraphDeps(
        stt=FakeSpeechToText(["이거 먹어도 돼?", "여기", "여기", "여기"]),
        tts=FakeTextToSpeech(),
        vlm=FakeVLMClient(),
        frames=_Camera(),
        checker=_Checker(misses=10),
    )
    graph = build_graph(deps)

    state, follows = graph.invoke({}), 0
    while (info := follow_up_turn(state)) is not None:
        state, follows = graph.invoke(info), follows + 1

    assert follows == MAX_FOLLOW_UP
    assert deps.tts.spoken == [RETAKE] * (MAX_FOLLOW_UP + 1)


@pytest.mark.parametrize(
    "answer",
    [
        CLARIFY_GIVE_UP,
        "무슨 말씀이신지 잘 모르겠어요. 다시 말씀해 주시겠어요?",
        "어떤 물건을 말씀하시는지 알려 주시면 답해 드릴게요.",
        "책상 위에 컵이 두 개 있어요. 어느 쪽을 물어보시는 건가요?",
    ],
)
def test_asking_back_answers_open_follow_up(answer):
    """사용자에게 다시 말하거나 가리켜 달라는 대답 뒤에는 이어 듣는다."""
    deps = GraphDeps(
        stt=FakeSpeechToText(["이거 뭐야?"]), tts=FakeTextToSpeech(), vlm=FakeVLMClient([answer])
    )

    follow = follow_up_turn(build_graph(deps).invoke({"session_id": 3, "user_id": 1}))

    assert follow == {
        "session_id": 3,
        "user_id": 1,
        "trigger_type": FOLLOW_UP_TRIGGER,
        "follow_up_of": "이거 뭐야?",
        "follow_up_count": 1,
    }


def test_plain_answer_does_not_follow_up():
    """그냥 대답이면 예전처럼 호출어 대기로 돌아간다."""
    deps = GraphDeps(
        stt=FakeSpeechToText(["이거 뭐야?"]), tts=FakeTextToSpeech(), vlm=FakeVLMClient()
    )

    assert follow_up_turn(build_graph(deps).invoke({})) is None


@pytest.mark.parametrize(
    ("reply", "question"),
    [
        ("다시 확인해줘", "이거 먹어도 돼? 다시 확인해줘"),
        ("여기", "이거 먹어도 돼? 여기"),
        ("이제 보여?", "이거 먹어도 돼? 이제 보여?"),
        ("왼쪽 거", "이거 먹어도 돼? 왼쪽 거"),
        ("저건 뭐야?", "저건 뭐야?"),
        ("아까 본 책 뭐야?", "아까 본 책 뭐야?"),
        ("이 과자 성분 알려줘", "이 과자 성분 알려줘"),
    ],
)
def test_follow_up_question_joins_short_replies(reply, question):
    """짧은 대답은 앞 질문 뒤에 붙이고, 물음말 · 성분 · 기억 질문은 새 질문으로 둔다."""
    assert follow_up_question("이거 먹어도 돼?", reply) == question
