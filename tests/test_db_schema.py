"""DB 스키마가 실제로 만들어지고 돌아가는지 확인한다.

Postgres 를 띄우지 않고 SQLite 인메모리로 돌린다.
테이블·관계·외래키가 맞는지 보는 데는 그걸로 충분하고,
CI 에서 컨테이너를 띄우지 않아도 되기 때문이다.

Postgres 에서만 터지는 문제(타입 불일치 등)는 여기서 안 잡힌다.
그건 실제로 붙여보고 확인해야 한다.

실행: pytest tests/test_db_schema.py
"""

import pytest

from jarviseo.memory.models import Session, Turn, TurnLatency, User, UserAllergen
from jarviseo.memory.store import MemoryStore
from jarviseo.types import AssistantResponse, Frame, Intent, Utterance


@pytest.fixture
def store() -> MemoryStore:
    """매 테스트마다 빈 인메모리 DB 를 준다."""
    s = MemoryStore(database_url="sqlite://")
    s.init_schema()
    return s


def test_테이블이_전부_만들어진다(store: MemoryStore):
    from sqlalchemy import inspect

    tables = set(inspect(store.engine).get_table_names())
    expected = {
        "users",
        "user_allergens",
        "sessions",
        "frames",
        "turns",
        "turn_latencies",
        "detections",
        "target_resolutions",
        "allergen_judgements",
        "belongings",
        "observations",
    }
    missing = expected - tables
    assert not missing, f"만들어지지 않은 테이블: {missing}"


def test_사용자와_알레르기(store: MemoryStore):
    user_id = store.ensure_user("owner")

    # 같은 이름으로 다시 불러도 새로 만들지 않는다.
    assert store.ensure_user("owner") == user_id

    store.set_user_allergens(user_id, ["땅콩", "우유"])
    assert set(store.get_user_allergens(user_id)) == {"땅콩", "우유"}

    # 통째로 교체된다. 예전 값이 남으면 안 된다.
    store.set_user_allergens(user_id, ["땅콩"])
    assert store.get_user_allergens(user_id) == ["땅콩"]


def test_대화_한_건을_기록한다(store: MemoryStore):
    import numpy as np

    user_id = store.ensure_user()
    session_id = store.start_session(user_id, device="macbook-m3", frame_source="folder")

    utterance = Utterance(text="저거 뭐야?", started_at=10.0, ended_at=11.2, confidence=0.9)
    frame = Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        timestamp=10.1,
        source_id="folder/img_001.jpg",
        sharpness=142.5,
    )
    response = AssistantResponse(
        text="저 가방은 검은색 백팩입니다.",
        intent=Intent.POINTING,
        used_frame=frame,
        latency_ms={"stt": 120.0, "detect": 45.0, "vlm": 830.0, "tts": 210.0},
    )

    turn_id = store.log_turn(session_id, utterance, response, frame_path="data/x.jpg")

    with store.session() as db:
        turn = db.get(Turn, turn_id)
        assert turn.utterance == "저거 뭐야?"
        assert turn.intent == "pointing"
        assert turn.utterance_started_monotonic == 10.0
        # 프레임이 같이 저장되고 크기가 (width, height) 순서로 들어갔는지
        assert turn.frame is not None
        assert (turn.frame.width, turn.frame.height) == (640, 480)
        # 지연은 단계별로 행이 나뉘어야 한다
        assert len(turn.latencies) == 4


def test_지연_백분위(store: MemoryStore):
    user_id = store.ensure_user()
    session_id = store.start_session(user_id)

    with store.session() as db:
        turn = Turn(
            session_id=session_id,
            utterance="x",
            utterance_started_monotonic=0.0,
            intent="general",
        )
        db.add(turn)
        db.flush()
        # 1~100ms 를 넣으면 p50 은 50 언저리, p95 는 95 언저리여야 한다.
        db.add_all(
            TurnLatency(turn_id=turn.id, stage="vlm", elapsed_ms=float(i)) for i in range(1, 101)
        )

    stats = store.latency_percentiles("vlm")
    assert stats["count"] == 100
    assert 49 <= stats["p50"] <= 52
    assert 94 <= stats["p95"] <= 97

    # 기록이 없는 단계는 0 을 돌려준다. 예외를 던지면 대시보드가 깨진다.
    assert store.latency_percentiles("없는단계") == {"p50": 0.0, "p95": 0.0, "count": 0}


def test_사용자를_지우면_딸린_기록도_지워진다(store: MemoryStore):
    """cascade 설정이 실제로 걸려 있는지 확인한다."""
    user_id = store.ensure_user()
    store.set_user_allergens(user_id, ["땅콩"])
    store.start_session(user_id)

    with store.session() as db:
        db.delete(db.get(User, user_id))

    with store.session() as db:
        assert db.query(UserAllergen).count() == 0
        assert db.query(Session).count() == 0
