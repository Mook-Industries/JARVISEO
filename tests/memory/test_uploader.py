"""끝난 턴을 클라우드로 보내기(memory/uploader.py) — 느리거나 꺼진 서버, 다시 보내기, 보내는 내용.

서버는 httpx.MockTransport 로 흉내 낸다(네트워크 없이 돈다). DB 는 임시 SQLite 파일이다.
업로더는 다른 스레드에서 DB 를 읽는데, 인메모리 SQLite 는 스레드마다 다른 DB 가 된다.
"""

import json
import threading
import time

import httpx
import pytest

from jarviseo.memory import MemoryStore
from jarviseo.memory.uploader import TurnUploader
from jarviseo.types import AssistantResponse, Intent, Utterance

URL = "https://jarviseo.example"


class FakeServer:
    """POST /api/turns 를 받는 가짜 서버. up=False 면 연결이 안 되고, delay 만큼 늦게 답한다."""

    def __init__(self, token: str = "secret") -> None:
        self.token = token
        self.up = True
        self.delay = 0.0
        self.bodies: list[dict] = []
        self.release = threading.Event()  # delay 동안 잡혀 있는 요청을 먼저 풀어 줄 때
        self.release.set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        if not self.up:
            raise httpx.ConnectError("꺼짐", request=request)
        self.release.wait(self.delay or None)
        if request.headers.get("authorization") != f"Bearer {self.token}":
            return httpx.Response(401, json={"detail": "토큰이 틀립니다"})
        body = json.loads(request.content)
        self.bodies.append(body)
        return httpx.Response(200, json={"saved": len(body["turns"]), "skipped": 0, "failed": 0})

    @property
    def questions(self) -> list[str]:
        return [t["turn"]["question_text"] for b in self.bodies for t in b["turns"]]


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(database_url=f"sqlite:///{tmp_path / 'local.db'}")
    s.init_schema()
    return s


@pytest.fixture
def server():
    return FakeServer()


def make_uploader(store, server, tmp_path, token="secret", **kw) -> TurnUploader:
    kw.setdefault("retry_sec", (0.05, 0.1))
    return TurnUploader(
        store,
        URL,
        token,
        state_path=tmp_path / "cloud_upload.json",
        transport=httpx.MockTransport(server.handler),
        **kw,
    )


def log(store: MemoryStore, session_id: int, question: str) -> int:
    now = time.monotonic()
    utterance = Utterance(text=question, started_at=now - 1.5, ended_at=now - 0.5)
    latency = {"stt": 900.0, "total": 2100.0}
    answer = f"{question} 의 대답"
    response = AssistantResponse(text=answer, intent=Intent.GENERAL, latency_ms=latency)
    return store.log_turn(session_id, utterance, response, image_path="data/captures/1.jpg")


def wait_until(check, timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.01)
    return check()


def test_sends_new_turns_without_photo_path(store, server, tmp_path):
    """새 턴을 보낸다. 질문 · 대답 · 지연은 가고 사진 경로는 안 간다."""
    session = store.start_session(title="assistant")
    uploader = make_uploader(store, server, tmp_path)
    turn_id = log(store, session, "이거 뭐야?")

    assert uploader.flush()
    [sent] = server.bodies[0]["turns"]
    assert sent["local_turn_id"] == turn_id and sent["local_session_id"] == session
    assert sent["turn"]["question_text"] == "이거 뭐야?"
    assert sent["turn"]["answer_text"] == "이거 뭐야? 의 대답"
    assert sent["turn"]["total_ms"] == 2100
    assert sent["voice"]["stt_ms"] == 900
    assert sent["turn"]["asked_at"].endswith("+00:00")  # UTC 로 보낸다
    assert "image_path" not in sent["turn"] and "audio_path" not in sent["voice"]
    assert uploader.last_sent_id == turn_id


def test_starts_after_existing_turns(store, server, tmp_path):
    """처음 켜면 그때까지 있던 턴(예전 측정 턴)은 보내지 않고, 그 뒤 턴부터 보낸다."""
    session = store.start_session()
    log(store, session, "예전 턴")
    uploader = make_uploader(store, server, tmp_path)
    log(store, session, "새 턴")

    assert uploader.flush()
    assert server.questions == ["새 턴"]


def test_resumes_from_saved_position(store, server, tmp_path):
    """끄고 다시 켜면 보낸 데까지는 건너뛰고, 못 보낸 턴부터 보낸다."""
    session = store.start_session()
    first = make_uploader(store, server, tmp_path)
    log(store, session, "보낸 턴")
    assert first.flush()
    server.up = False
    log(store, session, "못 보낸 턴")
    assert not first.flush()
    first.close()

    server.up = True
    again = make_uploader(store, server, tmp_path)
    assert again.flush()
    assert server.questions == ["보낸 턴", "못 보낸 턴"]


def test_slow_server_does_not_block_logging(store, server, tmp_path):
    """서버가 느려도 턴 기록과 notify 는 기다리지 않는다(대답이 늦어지지 않는다)."""
    session = store.start_session()
    uploader = make_uploader(store, server, tmp_path)
    server.delay = 2.0
    server.release.clear()
    uploader.start()
    try:
        t0 = time.perf_counter()
        for i in range(3):
            log(store, session, f"질문 {i}")
            uploader.notify()
        spent = time.perf_counter() - t0
        assert spent < 0.5  # 서버는 한 요청에 2초를 잡고 있다
        server.release.set()
        assert wait_until(lambda: len(server.questions) == 3)
    finally:
        uploader.close()
    assert server.questions == ["질문 0", "질문 1", "질문 2"]


def test_server_down_then_back_sends_missed_turns(store, server, tmp_path):
    """서버가 꺼진 동안 쌓인 턴은, 다시 켜지면 순서대로 한 번씩 올라간다."""
    session = store.start_session()
    uploader = make_uploader(store, server, tmp_path)
    server.up = False
    uploader.start()
    try:
        for i in range(3):
            log(store, session, f"끊긴 동안 {i}")
            uploader.notify()
        assert wait_until(lambda: uploader.last_error is not None)
        assert uploader.pending() == 3
        server.up = True  # 다시 시도(0.05~0.1초 뒤)가 보낸다
        assert wait_until(lambda: uploader.pending() == 0)
    finally:
        uploader.close()
    assert server.questions == ["끊긴 동안 0", "끊긴 동안 1", "끊긴 동안 2"]
    assert uploader.last_error is None


def test_wrong_token_keeps_turns_pending(store, server, tmp_path):
    """토큰이 틀려 서버가 안 받으면 보낸 것으로 치지 않는다."""
    session = store.start_session()
    uploader = make_uploader(store, server, tmp_path, token="wrong")
    log(store, session, "이거 뭐야?")

    assert not uploader.flush()
    assert uploader.last_error.startswith("401")
    assert uploader.pending() == 1 and server.bodies == []


def test_close_sends_once_more(store, server, tmp_path):
    """끝낼 때 notify 를 못 받은 턴도 한 번 더 보내 본다."""
    session = store.start_session()
    uploader = make_uploader(store, server, tmp_path, idle_sec=60.0)
    uploader.start()
    log(store, session, "마지막 질문")
    uploader.close()
    assert server.questions == ["마지막 질문"]
