"""장면 설명이 뒤에서 observation 에 들어가고, 나중에 기억 검색으로 찾히는지 확인한다.

VLM 과 임베딩 API 는 부르지 않는다.
"""

import time

import numpy as np
import pytest

from jarviseo.memory import FakeEmbedder, MemoryStore, Observation, SceneMemory, VectorMemory
from jarviseo.types import AssistantResponse, Frame, Intent, Utterance


@pytest.fixture
def file_store(tmp_path):
    """저장이 다른 스레드에서 돈다. SQLite 인메모리는 스레드마다 다른 DB 라서 파일 DB 를 쓴다."""
    s = MemoryStore(database_url=f"sqlite:///{(tmp_path / 'scene.db').as_posix()}")
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


def frame():
    return Frame(image=np.zeros((10, 10, 3), np.uint8), timestamp=0.0, source_id="test")


def test_장면_설명을_뒤에서_저장하고_나중에_찾는다(file_store):
    memory = VectorMemory(file_store, embedder=FakeEmbedder(), min_score=0.0)
    scenes = SceneMemory(memory, describe=lambda f: "책상 위에 검은 백팩이 있다")
    user_id, turn_id = a_turn(file_store)

    observation_id = scenes.remember(user_id, turn_id, frame()).result(timeout=5)
    scenes.close()

    assert observation_id is not None
    (hit,) = memory.search(user_id, "검은 백팩 어디 있었지", top_k=1)
    assert hit.text == "책상 위에 검은 백팩이 있다"
    assert hit.metadata["turn_id"] == turn_id


def test_설명을_못_받으면_아무것도_넣지_않는다(file_store):
    scenes = SceneMemory(VectorMemory(file_store, embedder=FakeEmbedder()), describe=lambda f: None)
    user_id, turn_id = a_turn(file_store)

    assert scenes.remember(user_id, turn_id, frame()).result(timeout=5) is None
    with file_store.session() as db:
        assert db.query(Observation).count() == 0


def test_저장이_실패해도_예외를_올리지_않는다(file_store, caplog):
    def broken(f):
        raise RuntimeError("VLM 이 이상한 값을 줬다")

    scenes = SceneMemory(VectorMemory(file_store, embedder=FakeEmbedder()), describe=broken)
    user_id, turn_id = a_turn(file_store)

    assert scenes.remember(user_id, turn_id, frame()).result(timeout=5) is None
    assert "장면 기억 저장 실패" in caplog.text


def test_close_는_밀린_저장을_마저_한다(file_store):
    def slow(f):
        time.sleep(0.2)
        return "현관에 우산이 있다"

    memory = VectorMemory(file_store, embedder=FakeEmbedder())
    scenes = SceneMemory(memory, describe=slow)
    user_id, turn_id = a_turn(file_store)

    scenes.remember(user_id, turn_id, frame())
    scenes.remember(user_id, turn_id, frame())
    scenes.close()  # 프로그램을 끝내도 마지막 턴의 장면이 사라지지 않는다

    with file_store.session() as db:
        assert db.query(Observation).count() == 2
