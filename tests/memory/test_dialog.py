"""질문·대답이 뒤에서 observation 에 들어가고, 나중에 기억 검색으로 찾히는지 확인한다.

임베딩 API 는 부르지 않는다.
"""

import time

import pytest

from jarviseo.memory import DialogMemory, FakeEmbedder, MemoryStore, Observation, VectorMemory
from jarviseo.memory.dialog import dialog_text, says_nothing
from jarviseo.types import AssistantResponse, Intent, Utterance


@pytest.fixture
def file_store(tmp_path):
    """저장이 다른 스레드에서 돈다. SQLite 인메모리는 스레드마다 다른 DB 라서 파일 DB 를 쓴다."""
    s = MemoryStore(database_url=f"sqlite:///{(tmp_path / 'dialog.db').as_posix()}")
    s.init_schema()
    yield s
    s.engine.dispose()


def a_turn(store):
    """기록된 턴 하나와 사용자를 만든다. observation 은 턴에 붙는다."""
    user_id = store.ensure_user()
    session_id = store.start_session(user_id)
    now = time.monotonic()
    utterance = Utterance(text="이거 뭐야?", started_at=now - 1.0, ended_at=now)
    response = AssistantResponse(text="백팩이에요.", intent=Intent.GENERAL)
    return user_id, store.log_turn(session_id, utterance, response)


def test_saves_in_background_and_finds_later(file_store):
    """질문과 대답을 뒤에서 저장하고 나중에 찾는다."""
    memory = VectorMemory(file_store, embedder=FakeEmbedder(), min_score=0.0)
    dialog = DialogMemory(memory)
    user_id, turn_id = a_turn(file_store)

    observation_id = dialog.remember(user_id, turn_id, "이거 뭐야?", "검은 백팩이에요.")
    assert observation_id.result(timeout=5) is not None
    dialog.close()

    (hit,) = memory.search(user_id, "아까 본 검은 백팩", top_k=1)
    assert hit.text == dialog_text("이거 뭐야?", "검은 백팩이에요.")
    assert hit.text == "질문: 이거 뭐야?\n대답: 검은 백팩이에요."
    assert hit.metadata["turn_id"] == turn_id


@pytest.mark.parametrize(
    ("question", "answer"), [("이거 뭐야?", ""), ("", "컵이에요."), (" ", " ")]
)
def test_skips_empty_question_or_answer(file_store, question, answer):
    """질문이나 대답이 비었으면 넣지 않는다."""
    dialog = DialogMemory(VectorMemory(file_store, embedder=FakeEmbedder()))
    user_id, turn_id = a_turn(file_store)

    assert dialog.remember(user_id, turn_id, question, answer).result(timeout=5) is None
    with file_store.session() as db:
        assert db.query(Observation).count() == 0


def test_save_failure_does_not_raise(file_store, caplog):
    """저장이 실패해도 예외를 올리지 않는다."""

    def broken(*args, **kwargs):
        raise RuntimeError("DB 연결이 끊겼다")

    memory = VectorMemory(file_store, embedder=FakeEmbedder())
    memory.remember_observation = broken
    dialog = DialogMemory(memory)
    user_id, turn_id = a_turn(file_store)

    assert dialog.remember(user_id, turn_id, "이거 뭐야?", "컵이에요.").result(timeout=5) is None
    assert "대화 기억 저장 실패" in caplog.text


def test_close_flushes_pending_saves(file_store):
    """`close()` 는 밀린 저장을 마저 한다."""

    class Slow(FakeEmbedder):
        def embed(self, text):
            time.sleep(0.2)
            return super().embed(text)

    dialog = DialogMemory(VectorMemory(file_store, embedder=Slow()))
    user_id, turn_id = a_turn(file_store)

    dialog.remember(user_id, turn_id, "이거 뭐야?", "우산이에요.")
    dialog.remember(user_id, turn_id, "저건 뭐야?", "신발장이에요.")
    dialog.close()  # 프로그램을 끝내도 마지막 턴의 대화가 사라지지 않는다

    with file_store.session() as db:
        assert db.query(Observation).count() == 2


# 실측 기록(data/assistant.db, 10/15·10/19)에서 가져온 대답
NOTHING = [
    ("이건 뭐야?", "사진에는 안경을 쓴 사람이 보이고, 뒤에는 사물함이 있습니다."),
    (
        "이거 뭐야?",
        "안경을 쓴 분이 보이고, 뒤에는 번호가 붙은 사물함이 있습니다. "
        "어떤 것을 물으시는지 알려주시면 더 자세히 말씀드릴게요.",
    ),
    ("이거 뭐지?", "카메라 앞에 손바닥을 펼친 사람이 보입니다. 뒤에는 사물함이 있고요."),
    ("지금 몇 시?", "사진에는 시계가 보이지 않아 지금 몇 시인지 알 수 없습니다."),
    ("방금 제작한 게 뭐지?", "사진에는 제작한 물건이 보이지 않아 무엇을 만드셨는지 모르겠습니다."),
]
SOMETHING = [
    ("이건 뭐지?", "손에 들고 계신 건 스마트폰으로 보입니다."),
    ("이거 뭐야?", "손가락으로 가리키신 건 검은색 뿔테 안경입니다."),
    (
        "이건 뭐야?",
        "노트북 옆에 다이얼이 달린 전자기기가 보입니다. "
        "사진만으로는 정확히 어떤 기기인지 알기 어렵습니다.",
    ),
    (
        "이거 뭐야?",
        "손에 쥐고 계신 검은색 물건은 펜처럼 보이는데, 사진만으로는 정확히 무엇인지 모르겠습니다.",
    ),
    ("열쇠 어디 있어?", "열쇠는 보이지 않지만 지갑은 책상 위에 있습니다."),
    ("이거 뭐야?", "부분이 보이는 건 초록 물병입니다."),
    # 사람·뒤를 직접 물었으면 그 대답이 답이다
    ("사람 뒤에 뭐가 있지?", "사람 뒤에는 흰색 사물함이 여러 칸 있습니다."),
    ("사람은 안 보여?", "네, 안경을 쓴 사람 한 분이 바로 앞에 보입니다."),
]


@pytest.mark.parametrize(("question", "answer"), NOTHING)
def test_says_nothing_for_people_background_or_unknown(question, answer):
    """사람·뒤만 말하거나 모른다고만 한 대답은 알아본 것이 없다."""
    assert says_nothing(question, answer)


@pytest.mark.parametrize(("question", "answer"), SOMETHING)
def test_says_something_when_anything_was_named(question, answer):
    """하나라도 알아본 것이 있거나, 사람·뒤를 직접 물었으면 기억할 것이 있다."""
    assert not says_nothing(question, answer)


def test_skips_answer_that_says_nothing(file_store):
    """알아본 것이 없는 대답은 기억에 넣지 않는다. 그래서 다른 질문에 찾히지도 않는다."""
    memory = VectorMemory(file_store, embedder=FakeEmbedder(), min_score=0.0)
    dialog = DialogMemory(memory)
    user_id, turn_id = a_turn(file_store)

    question, answer = NOTHING[0]
    assert dialog.remember(user_id, turn_id, question, answer).result(timeout=5) is None
    dialog.close()

    with file_store.session() as db:
        assert db.query(Observation).count() == 0
    assert memory.search(user_id, "아까 본 지갑 기억나?") == []
