"""질문·대답이 뒤에서 observation 에 들어가고, 나중에 기억 검색으로 찾히는지 확인한다.

임베딩 API 는 부르지 않는다.
"""

import time

import pytest

from jarviseo.memory import DialogMemory, FakeEmbedder, MemoryStore, Observation, VectorMemory
from jarviseo.memory.dialog import dialog_text
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
