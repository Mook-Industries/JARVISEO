"""기억 검색 — 장면 설명을 넣고 비슷한 질문으로 다시 찾는다.

임베딩은 FakeEmbedder(글자가 겹칠수록 가까움)로 API 없이 돈다.
``store`` fixture 가 SQLite 와 실제 Postgres(pgvector 코사인 거리)에서 한 번씩 돌린다.
"""

import time

import pytest
from openai import OpenAIError

from jarviseo.memory import FakeEmbedder, MemoryStore, VectorMemory
from jarviseo.memory.models import Observation
from jarviseo.types import AssistantResponse, Intent, Utterance

SCENES = ["책상 위 검은 백팩", "냉장고 옆 초록 물병", "현관 신발장 우산"]


class _BrokenEmbedder(FakeEmbedder):
    def embed(self, text: str) -> list[float]:
        raise OpenAIError("연결 끊김")


def _turn(store: MemoryStore, user_id: int, image_path: str = "data/frames/1.jpg") -> int:
    session_id = store.start_session(user_id)
    utterance = Utterance(text="이거 기억해", started_at=0.0, ended_at=1.0)
    response = AssistantResponse(text="네", intent=Intent.GENERAL)
    return store.log_turn(session_id, utterance, response, image_path=image_path)


@pytest.fixture
def memory(store: MemoryStore) -> VectorMemory:
    return VectorMemory(store, FakeEmbedder(), min_score=0.2)


def test_finds_scene_by_similar_question(store: MemoryStore, memory: VectorMemory):
    """비슷한 질문으로 본 장면을 찾는다."""
    user_id = store.ensure_user()
    turn_id = _turn(store, user_id)
    ids = [memory.remember_observation(user_id, turn_id, s, place="거실") for s in SCENES]

    hits = memory.search(user_id, "검은 백팩 어디 있었지")

    assert [h.memory_id for h in hits] == [str(ids[0])]
    hit = hits[0]
    assert hit.kind == "observation"
    assert hit.text == "책상 위 검은 백팩"
    assert 0.2 <= hit.score <= 1.0
    assert hit.metadata == {"turn_id": turn_id, "place": "거실"}
    assert str(hit.image_path).replace("\\", "/") == "data/frames/1.jpg"
    assert abs(hit.observed_at - time.time()) < 60
    assert memory.last_latency_ms is not None


def test_returns_top_k_nearest_first(store: MemoryStore):
    """가까운 순으로 `top_k` 개까지만 준다."""
    memory = VectorMemory(store, FakeEmbedder(), min_score=0.0)
    user_id = store.ensure_user()
    turn_id = _turn(store, user_id)
    for scene in ["검은 백팩", "검은 백팩 지퍼", "검은 우산", "초록 물병"]:
        memory.remember_observation(user_id, turn_id, scene)

    hits = memory.search(user_id, "검은 백팩", top_k=2)

    assert [h.text for h in hits] == ["검은 백팩", "검은 백팩 지퍼"]
    assert hits[0].score >= hits[1].score


def test_drops_hits_below_threshold(store: MemoryStore, memory: VectorMemory):
    """점수가 문턱보다 낮으면 버린다."""
    user_id = store.ensure_user()
    turn_id = _turn(store, user_id)
    for scene in SCENES:
        memory.remember_observation(user_id, turn_id, scene)

    assert memory.search(user_id, "오늘 날씨 어때") == []


def test_excludes_other_users(store: MemoryStore, memory: VectorMemory):
    """다른 사용자의 기록은 안 나온다."""
    me, other = store.ensure_user(), store.ensure_user("other@jarviseo.local")
    memory.remember_observation(other, _turn(store, other), "책상 위 검은 백팩")

    assert memory.search(me, "검은 백팩") == []


def test_embedding_failure_keeps_text_but_hides_from_search(store: MemoryStore):
    """임베딩이 실패해도 설명은 남고 검색에는 안 나온다."""
    broken = VectorMemory(store, _BrokenEmbedder(), min_score=0.0)
    user_id = store.ensure_user()
    turn_id = _turn(store, user_id)

    observation_id = broken.remember_observation(user_id, turn_id, "책상 위 검은 백팩")

    with store.session() as db:
        saved = db.get(Observation, observation_id)
        assert saved.description == "책상 위 검은 백팩"
        assert saved.embedding is None
        assert saved.embed_model is None
    assert VectorMemory(store, FakeEmbedder(), min_score=0.0).search(user_id, "검은 백팩") == []
    assert broken.search(user_id, "검은 백팩") == []
    assert broken.last_latency_ms is None


def test_empty_query_is_not_searched(store: MemoryStore, memory: VectorMemory):
    """빈 질문은 찾지 않는다."""
    user_id = store.ensure_user()
    memory.remember_observation(user_id, _turn(store, user_id), "책상 위 검은 백팩")

    assert memory.search(user_id, "  ") == []


def test_embedding_model_name_is_saved(store: MemoryStore, memory: VectorMemory):
    """임베딩 모델 이름이 남는다."""
    user_id = store.ensure_user()
    observation_id = memory.remember_observation(user_id, _turn(store, user_id), "검은 백팩")

    with store.session() as db:
        saved = db.get(Observation, observation_id)
        assert saved.embed_model == "fake-bigram"
        assert len(saved.embedding) == 1536
