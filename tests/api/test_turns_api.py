"""노트북 턴 받기(POST /api/turns)와 대시보드 채팅 이력(GET /api/turns) — #144.

노트북 쪽 업로더(memory/uploader.py)가 실제로 만드는 본문을 그대로 보낸다.
CI 에는 fastapi 를 설치하지 않으므로(requirements-dev.txt) 로컬에서만 돈다.
"""

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from jarviseo.api.main import create_app  # noqa: E402
from jarviseo.memory import MemoryStore, SessionTurn  # noqa: E402
from jarviseo.memory.store import turn_payload  # noqa: E402
from jarviseo.types import AssistantResponse, Intent, Utterance  # noqa: E402

TOKEN = "secret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class NoChecker:
    def close(self):
        pass


class NoAssistant:
    def close(self):
        pass


@pytest.fixture
def server(tmp_path):
    store = MemoryStore(database_url=f"sqlite:///{tmp_path / 'cloud.db'}")
    app = create_app(store=store, checker=NoChecker(), assistant=NoAssistant(), cloud_token=TOKEN)
    with TestClient(app) as client:
        yield client, store


@pytest.fixture
def laptop(tmp_path) -> MemoryStore:
    store = MemoryStore(database_url=f"sqlite:///{tmp_path / 'laptop.db'}")
    store.init_schema()
    return store


def laptop_turns(store: MemoryStore, *questions: str) -> list[dict]:
    """노트북 DB 에 턴을 남기고, 업로더가 보내는 모양으로 읽어 온다."""
    session = store.start_session(title="assistant")
    for q in questions:
        now = time.monotonic()
        utterance = Utterance(text=q, started_at=now - 1.5, ended_at=now - 0.5)
        latency = {"stt": 800.0, "vlm": 1500.0, "total": 2400.0}
        response = AssistantResponse(text=f"{q} 대답", intent=Intent.GENERAL, latency_ms=latency)
        store.log_turn(session, utterance, response, image_path="data/captures/x.jpg")
    return [turn_payload(t) for t in store.turns_after(0)]


def test_receives_turns_with_token(server, laptop):
    """토큰이 맞으면 받은 턴을 서버 DB 에 남긴다. 지연 · 음성 기록도 같이 들어간다."""
    client, store = server
    body = {"source": "ab12cd34", "turns": laptop_turns(laptop, "이거 뭐야?", "아까 그거 뭐였지?")}

    r = client.post("/api/turns", json=body, headers=AUTH)

    assert r.status_code == 200
    assert r.json() == {"saved": 2, "skipped": 0, "failed": 0}
    with store.session() as db:
        rows = db.query(SessionTurn).order_by(SessionTurn.turn_id).all()
        assert [t.question_text for t in rows] == ["이거 뭐야?", "아까 그거 뭐였지?"]
        assert rows[0].total_ms == 2400 and rows[0].voice.stt_ms == 800
        assert rows[0].inference.llm_ms == 1500
        assert rows[0].image_path is None  # 사진 경로는 안 온다
        assert rows[0].session.title == "노트북 ab12cd34 #1"


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}]
)
def test_rejects_wrong_token(server, laptop, headers):
    """토큰이 없거나 틀리면 401 이고 아무것도 남기지 않는다."""
    client, store = server
    body = {"source": "ab12cd34", "turns": laptop_turns(laptop, "이거 뭐야?")}

    r = client.post("/api/turns", json=body, headers=headers)

    assert r.status_code == 401
    assert store.last_turn_id() == 0


def test_server_without_token_accepts_nothing(tmp_path, laptop):
    """서버에 토큰을 정하지 않았으면 어떤 요청도 받지 않는다(빈 토큰으로 열리지 않는다)."""
    store = MemoryStore(database_url=f"sqlite:///{tmp_path / 'cloud.db'}")
    app = create_app(store=store, checker=NoChecker(), assistant=NoAssistant(), cloud_token="")
    body = {"source": "ab12cd34", "turns": laptop_turns(laptop, "이거 뭐야?")}
    with TestClient(app) as client:
        r = client.post("/api/turns", json=body, headers={"Authorization": "Bearer "})
    assert r.status_code == 503


def test_resent_turns_are_skipped(server, laptop):
    """서버가 받았다는 답을 노트북이 못 듣고 같은 턴을 다시 보내도 두 번 남지 않는다."""
    client, store = server
    body = {"source": "ab12cd34", "turns": laptop_turns(laptop, "이거 뭐야?")}
    client.post("/api/turns", json=body, headers=AUTH)

    r = client.post("/api/turns", json=body, headers=AUTH)

    assert r.json() == {"saved": 0, "skipped": 1, "failed": 0}
    assert store.last_turn_id() == 1


def test_feed_lists_laptop_turns_then_new_ones(server, laptop):
    """채팅 이력은 최근 턴을 새것부터 주고, after 를 주면 그 뒤에 들어온 턴만 준다."""
    client, _ = server
    turns = laptop_turns(laptop, "첫 질문", "둘째 질문", "셋째 질문")
    client.post("/api/turns", json={"source": "ab12cd34", "turns": turns[:2]}, headers=AUTH)

    first = client.get("/api/turns").json()["turns"]
    assert [t["question"] for t in first] == ["둘째 질문", "첫 질문"]
    assert first[0]["where"] == "노트북" and first[0]["answer"] == "둘째 질문 대답"
    assert first[0]["asked_at"].endswith("+00:00")  # 시각대가 있어야 브라우저가 바르게 바꾼다

    client.post("/api/turns", json={"source": "ab12cd34", "turns": turns[2:]}, headers=AUTH)
    later = client.get(f"/api/turns?after={first[0]['id']}").json()["turns"]
    assert [t["question"] for t in later] == ["셋째 질문"]
