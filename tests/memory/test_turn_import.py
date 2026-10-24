"""노트북 턴을 서버 DB 에 옮겨 남기기(store.turn_payload → import_turns) — #144.

받는 쪽 ``store`` 는 SQLite 와 Postgres 에서 한 번씩 돈다(conftest). 서버는 Postgres 라
시각 비교(같은 턴 다시 받기)가 Postgres 에서도 맞아야 한다.
"""

import json
import time

import pytest

from jarviseo.memory import MemoryStore, SessionTurn
from jarviseo.memory.store import turn_payload
from jarviseo.types import AssistantResponse, Intent, Utterance


@pytest.fixture
def laptop(tmp_path) -> MemoryStore:
    s = MemoryStore(database_url=f"sqlite:///{tmp_path / 'laptop.db'}")
    s.init_schema()
    return s


def sent(laptop: MemoryStore, *questions: str) -> list[dict]:
    """노트북에 턴을 남기고, 보낸 JSON 을 서버가 읽은 모양(글자 → dict)으로 준다."""
    session = laptop.start_session(title="assistant")
    for q in questions:
        now = time.monotonic()
        utterance = Utterance(text=q, started_at=now - 1.2, ended_at=now - 0.2)
        latency = {"stt": 700.0, "total": 2000.0}
        response = AssistantResponse(text=f"{q} 대답", intent=Intent.GENERAL, latency_ms=latency)
        laptop.log_turn(session, utterance, response, filler_sent=True)
    return json.loads(json.dumps([turn_payload(t) for t in laptop.turns_after(0)]))


def test_import_keeps_turn_voice_and_times(store, laptop):
    """받은 턴의 질문 · 대답 · 지연 · 말 시작 시각이 그대로 남는다."""
    turns = sent(laptop, "이거 뭐야?")

    assert store.import_turns("ab12cd34", turns) == {"saved": 1, "skipped": 0, "failed": 0}
    with store.session() as db:
        row = db.query(SessionTurn).one()
        assert row.question_text == "이거 뭐야?" and row.total_ms == 2000
        assert row.voice.stt_ms == 700 and row.voice.is_filler_sent
        local = laptop.turns_after(0)[0]
        assert row.voice.speech_started_at.replace(tzinfo=None) == (
            local.voice.speech_started_at.replace(tzinfo=None)
        )


def test_import_twice_skips_same_turns(store, laptop):
    """같은 턴을 두 번 받아도 한 번만 남는다. 같은 노트북 세션은 서버에서도 한 세션이다."""
    turns = sent(laptop, "첫 질문", "둘째 질문")
    store.import_turns("ab12cd34", turns[:1])

    assert store.import_turns("ab12cd34", turns) == {"saved": 1, "skipped": 1, "failed": 0}
    with store.session() as db:
        rows = db.query(SessionTurn).all()
        assert len(rows) == 2 and len({r.session_id for r in rows}) == 1


def test_broken_turn_does_not_block_others(store, laptop):
    """깨진 턴 하나는 failed 로 세고, 나머지는 남긴다(노트북이 그 턴에 막히지 않는다)."""
    turns = sent(laptop, "첫 질문", "둘째 질문")
    turns[0]["turn"]["asked_at"] = "시각이 아님"

    assert store.import_turns("ab12cd34", turns) == {"saved": 1, "skipped": 0, "failed": 1}
