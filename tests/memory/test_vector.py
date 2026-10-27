"""기억 검색 — 장면 설명을 넣고 비슷한 질문으로 다시 찾는다.

임베딩은 FakeEmbedder(글자가 겹칠수록 가까움)로 API 없이 돈다.
``store`` fixture 가 SQLite 와 실제 Postgres(pgvector 코사인 거리)에서 한 번씩 돌린다.
"""

import time
from datetime import UTC, datetime, timedelta

import pytest
from openai import OpenAIError

from jarviseo.memory import FakeEmbedder, MemoryStore, VectorMemory
from jarviseo.memory.models import Observation, SessionTurn
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


def _seen_days_ago(store: MemoryStore, observation_id: int, days: float) -> None:
    """기록을 며칠 전에 본 것으로 고친다."""
    with store.session() as db:
        db.get(Observation, observation_id).observed_at = datetime.now(UTC) - timedelta(days=days)


def test_recent_question_skips_old_records(store: MemoryStore, memory: VectorMemory):
    """아까 본 것을 물으면 며칠 전 기록은 찾지 않는다. 때 낱말이 없으면 찾는다."""
    user_id = store.ensure_user()
    turn_id = _turn(store, user_id)
    old = memory.remember_observation(user_id, turn_id, "검은 카드 지갑")
    _seen_days_ago(store, old, 6)

    assert memory.search(user_id, "아까 본 검은 카드 지갑") == []
    assert [h.memory_id for h in memory.search(user_id, "검은 카드 지갑")] == [str(old)]


def test_recent_question_finds_todays_record(store: MemoryStore, memory: VectorMemory):
    """아까 본 것을 물으면 오늘 기록 중에서 찾는다."""
    user_id = store.ensure_user()
    turn_id = _turn(store, user_id)
    _seen_days_ago(store, memory.remember_observation(user_id, turn_id, "검은 카드 지갑"), 6)
    today = memory.remember_observation(user_id, turn_id, "갈색 카드 지갑")

    assert [h.memory_id for h in memory.search(user_id, "아까 본 카드 지갑")] == [str(today)]


def test_yesterday_question_finds_only_yesterday(store: MemoryStore, memory: VectorMemory):
    """어제 본 것을 물으면 어제 기록만 찾는다. 오늘 것도 엿새 전 것도 뺀다."""
    user_id = store.ensure_user()
    turn_id = _turn(store, user_id)
    now = datetime.now().astimezone()
    noon_yesterday = now.replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=1)
    ids = [memory.remember_observation(user_id, turn_id, "검은 카드 지갑") for _ in range(3)]
    _seen_days_ago(store, ids[1], 6)
    with store.session() as db:
        db.get(Observation, ids[2]).observed_at = noon_yesterday.astimezone(UTC)

    hits = memory.search(user_id, "어제 본 검은 카드 지갑", now=now)

    assert [h.memory_id for h in hits] == [str(ids[2])]


def _asked(
    store: MemoryStore, session_id: int, question: str, answer: str, minutes_ago: float
) -> int:
    """질문 한 번을 minutes_ago 분 전에 한 것으로 기록한다."""
    utterance = Utterance(text=question, started_at=0.0, ended_at=1.0)
    turn_id = store.log_turn(
        session_id, utterance, AssistantResponse(text=answer, intent=Intent.GENERAL)
    )
    with store.session() as db:
        db.get(SessionTurn, turn_id).asked_at = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    return turn_id


def test_recent_dialog_question_adds_last_turns_in_time_order(
    store: MemoryStore, memory: VectorMemory
):
    """방금 물어본 것을 물으면 마지막 3턴을 시간순으로 붙인다. 기억에 안 넣은 턴(사과)도 넣는다."""
    user_id = store.ensure_user()
    session_id = store.start_session(user_id)
    _asked(store, session_id, "이 컵 뭐야?", "빨간 머그컵이에요.", 30)
    _asked(store, session_id, "이거 뭐야?", "검은 백팩이에요.", 20)
    _asked(store, session_id, "이거 먹어도 돼?", "죄송해요, 성분표를 읽지 못했어요.", 10)
    last = _asked(store, session_id, "마우스 어디 있어?", "책상 왼쪽에 있어요.", 1)

    hits = memory.search(user_id, "방금 물어봤던거 뭐지?")

    assert [h.kind for h in hits] == ["turn"] * 3
    assert [h.text.splitlines()[0] for h in hits] == [
        "질문: 이거 뭐야?",
        "질문: 이거 먹어도 돼?",
        "질문: 마우스 어디 있어?",
    ]
    assert hits[-1].text == "질문: 마우스 어디 있어?\n대답: 책상 왼쪽에 있어요."
    assert hits[-1].metadata == {"turn_id": last}
    assert hits[0].observed_at < hits[1].observed_at < hits[2].observed_at
    assert memory.last_latency_ms is not None


def test_recent_turns_skip_other_users_and_yesterday(store: MemoryStore, memory: VectorMemory):
    """다른 사용자의 턴과 아까 범위 밖(어제) 턴은 붙이지 않는다."""
    me, other = store.ensure_user(), store.ensure_user("other@jarviseo.local", "other")
    mine = store.start_session(me)
    _asked(store, mine, "어제 물은 것", "어제 대답", 60 * 30)
    _asked(store, store.start_session(other), "남의 질문", "남의 대답", 1)
    _asked(store, mine, "오늘 물은 것", "오늘 대답", 5)

    hits = memory.search(me, "아까 내가 뭐라고 물어봤지?")

    assert [h.text.splitlines()[0] for h in hits] == ["질문: 오늘 물은 것"]


def test_recent_turn_replaces_same_turn_found_by_meaning(store: MemoryStore):
    """뜻으로도 찾힌 턴은 한 번만 넣는다(최근 턴 쪽)."""
    memory = VectorMemory(store, FakeEmbedder(), min_score=0.0)
    user_id = store.ensure_user()
    turn_id = _asked(store, store.start_session(user_id), "그거 뭐야?", "검은 백팩이에요.", 1)
    memory.remember_observation(user_id, turn_id, "질문: 그거 뭐야?\n대답: 검은 백팩이에요.")

    hits = memory.search(user_id, "방금 그거 뭐였지?")

    assert [(h.kind, h.metadata["turn_id"]) for h in hits] == [("turn", turn_id)]


def test_recent_turns_come_even_if_embedding_fails(store: MemoryStore):
    """임베딩이 실패해도 최근 턴은 붙인다. 최근 턴은 임베딩 없이 읽는다."""
    memory = VectorMemory(store, _BrokenEmbedder())
    user_id = store.ensure_user()
    _asked(store, store.start_session(user_id), "이거 뭐야?", "검은 백팩이에요.", 1)

    hits = memory.search(user_id, "방금 뭐 물어봤지?")

    assert [h.text for h in hits] == ["질문: 이거 뭐야?\n대답: 검은 백팩이에요."]
