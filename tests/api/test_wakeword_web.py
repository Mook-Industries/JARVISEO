"""브라우저 호출어(#218) — 서버가 모델 3개를 내려주고, 끼어들어 물은 턴을 BARGE_IN 으로 남긴다.

CI 에는 fastapi 를 설치하지 않으므로(requirements-dev.txt) 로컬에서만 돈다.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")
from fastapi.testclient import TestClient  # noqa: E402

from jarviseo import config  # noqa: E402
from jarviseo.api import main  # noqa: E402
from jarviseo.api.assistant_web import WebAssistant  # noqa: E402
from jarviseo.memory import MemoryStore, SessionTurn  # noqa: E402
from jarviseo.vlm.client import FakeVLMClient  # noqa: E402


class NoObjects:
    """아무것도 못 찾는 검출기. 넣지 않으면 진짜 YOLO · 손 모델을 읽어 pytest 가 끝나지 않는다."""

    def detect(self, image):
        return []


class _NoChecker:
    synonyms: dict = {}

    def close(self):
        pass


@pytest.fixture
def models(tmp_path, monkeypatch):
    """모델 3개를 가짜 파일로 둔 폴더. 웹 이미지처럼 data/models/ 에 다 있는 상태."""
    for name in (*main.WAKEWORD_FEATURES, main.WAKEWORD_FILE):
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "WAKEWORD_MODEL", str(tmp_path / main.WAKEWORD_FILE))
    return tmp_path


@pytest.fixture
def client(tmp_path):
    store = MemoryStore(database_url=f"sqlite:///{tmp_path / 'web.db'}")
    store.init_schema()
    user = store.ensure_user()
    web = WebAssistant(
        store, None, user, vlm=FakeVLMClient(["네."]), detector=NoObjects(), photo_dir=None
    )
    with TestClient(main.create_app(store=store, checker=_NoChecker(), assistant=web)) as c:
        yield c, store


def test_wakeword_models_are_served(client, models):
    """모델 3개가 다 있으면 켤 수 있다고 알리고, 이름대로 파일을 내려준다."""
    c, _ = client
    info = c.get("/api/wakeword").json()
    assert info == {
        "available": True,
        "threshold": config.WAKEWORD_THRESHOLD,
        "playback_threshold": config.WAKEWORD_PLAYBACK_THRESHOLD,
    }
    for name in (*main.WAKEWORD_FEATURES, main.WAKEWORD_FILE):
        r = c.get(f"/models/wakeword/{name}")
        assert r.status_code == 200 and r.content == name.encode()
    assert c.get("/wakeword.js").text.startswith("/* 브라우저 호출어")


def test_only_wakeword_models_are_served(client, models):
    """다른 파일 이름이나 폴더 밖 경로는 내려주지 않는다."""
    c, _ = client
    (models / "hand_landmarker.task").write_bytes(b"x")
    for name in ("hand_landmarker.task", "..%2F..%2Fconfig.py", "jarviseo.pt"):
        assert c.get(f"/models/wakeword/{name}").status_code == 404


def test_wakeword_is_off_without_team_model(client, models, monkeypatch):
    """팀 모델이 없으면(또는 .onnx 경로가 아니면) 켤 수 없다고 알린다. 대시보드는 버튼만 쓴다."""
    c, _ = client
    monkeypatch.setattr(config, "WAKEWORD_MODEL", "hey_jarvis")
    assert c.get("/api/wakeword").json()["available"] is False
    assert c.get(f"/models/wakeword/{main.WAKEWORD_FILE}").status_code == 404


def test_barge_in_question_is_recorded(client):
    """대답을 읽는 중 "자비서"로 끼어들어 물은 턴은 BARGE_IN, 아니면 WAKEWORD 로 남는다."""
    c, store = client
    first = c.post("/api/ask", json={"question": "오늘 날씨 어때?"}).json()["turn_id"]
    barged = c.post("/api/ask", json={"question": "그럼 내일은?", "trigger": "BARGE_IN"})
    with store._make_session() as db:
        assert db.get(SessionTurn, first).trigger_type == "WAKEWORD"
        assert db.get(SessionTurn, barged.json()["turn_id"]).trigger_type == "BARGE_IN"
    assert c.post("/api/ask", json={"question": "음", "trigger": "CLICK"}).status_code == 422
