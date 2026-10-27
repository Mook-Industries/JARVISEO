"""브라우저 음성(/api/voice/ws, browser_voice.py) — 브라우저 소리로 노트북과 같은 부품이 돈다.

녹음 파일("자비서 … 질문")을 브라우저처럼 보내고, 서버가 2단계 확인 · 받아쓰기(OpenAI) · 읽기
(OpenAI 목소리)를 거쳐 대답 소리와 결과를 돌려주는지 본다. OpenAI 는 가짜 클라이언트로 바꾼다.
VAD(webrtcvad)는 진짜로 돈다.

CI 에는 fastapi 를 설치하지 않으므로(requirements-dev.txt) 로컬에서만 돈다.
"""

import wave
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")
pytest.importorskip("webrtcvad")
from fastapi.testclient import TestClient  # noqa: E402

from jarviseo.api import main  # noqa: E402
from jarviseo.api.assistant_web import WebAssistant  # noqa: E402
from jarviseo.api.browser_voice import AUDIO, HEADER, TTS_CHANNEL  # noqa: E402
from jarviseo.memory import MemoryStore, SessionTurn  # noqa: E402
from jarviseo.vlm.client import FakeVLMClient  # noqa: E402

DATA = Path(__file__).parents[1] / "data"
ANSWER = "책상 위의 컵이에요."


class NoObjects:
    def detect(self, image):
        return []


class _NoChecker:
    synonyms: dict = {}

    def close(self):
        pass


class FakeOpenAI:
    """받아쓰기: 호출어 확인 구간(wake.wav)은 wake, 질문(speech.wav)은 question.
    읽기: 요청마다 0.1초짜리 PCM 두 조각."""

    def __init__(self, wake="자비서", question="자비서, 이거 뭐야?"):
        self.wake, self.question = wake, question
        self.heard: list[str] = []
        self.spoken: list[str] = []
        speech = SimpleNamespace(with_streaming_response=SimpleNamespace(create=self._speech))
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self._transcribe), speech=speech
        )

    def _transcribe(self, **kwargs):
        name = kwargs["file"][0]
        self.heard.append(name)
        if name == "wake.wav":
            return SimpleNamespace(text=self.wake)
        if isinstance(self.question, list):  # 차례로
            return SimpleNamespace(text=self.question.pop(0))
        return SimpleNamespace(text=self.question)

    @contextmanager
    def _speech(self, **kwargs):
        self.spoken.append(kwargs["input"])
        chunk = b"\x10\x00" * 2400
        yield SimpleNamespace(iter_bytes=lambda size: iter([chunk, chunk]))


def recording() -> np.ndarray:
    """ "자비서"(0~0.6초) · 쉼 · 질문(1.2~2.06초) 뒤에 1.5초 조용함."""
    with wave.open(str(DATA / "jarviseo_question.wav"), "rb") as f:
        audio = np.frombuffer(f.readframes(f.getnframes()), np.int16)
    return np.concatenate([audio, np.zeros(24000, np.int16)])


@pytest.fixture
def ws_app(tmp_path):
    store = MemoryStore(database_url=f"sqlite:///{tmp_path / 'web.db'}")
    store.init_schema()
    user = store.ensure_user()
    web = WebAssistant(
        store, None, user, vlm=FakeVLMClient([ANSWER]), detector=NoObjects(), photo_dir=None
    )
    fake = FakeOpenAI()

    def make():
        app = main.create_app(store=store, checker=_NoChecker(), assistant=web, openai_client=fake)
        return TestClient(app)

    return make, store, fake


def until(ws, kind, sounds):
    """kind 알림이 올 때까지 받는다. 그사이 온 소리(채널, PCM)는 sounds 에, 알림 종류는 돌려준다."""
    seen = []
    while True:
        m = ws.receive()
        if m.get("bytes"):
            sounds.append((m["bytes"][0], m["bytes"][1:]))
            continue
        import json

        msg = json.loads(m["text"])
        seen.append(msg)
        if msg["type"] == kind:
            return seen


def test_browser_turn_runs_wake_check_stt_and_server_voice(ws_app):
    """호출 → 2단계 확인 → 받아쓰기 → 대답을 OpenAI 목소리로 읽어 브라우저로 보낸다."""
    make, store, fake = ws_app
    sounds: list = []
    with make() as c, c.websocket_connect("/api/voice/ws") as ws:
        assert until(ws, "ready", sounds)[-1]["type"] == "ready"
        audio = recording()
        ws.send_bytes(HEADER.pack(AUDIO, 0) + audio.tobytes())  # 직전 소리 묶음
        ws.send_json({"type": "wake", "id": 1, "pos": 11200, "score": 0.99, "camera": False})
        seen = until(ws, "idle", sounds)
    kinds = [m["type"] for m in seen]
    assert kinds[0] == "wake" and "thinking" in kinds and "result" in kinds
    result = next(m for m in seen if m["type"] == "result")["data"]
    assert result["question"] == "이거 뭐야?"  # 호출어는 지우고(transcript.clean)
    assert result["voice"] is True
    assert seen[-1] == {"type": "idle", "upto": 1}
    assert fake.heard[0] == "wake.wav" and "speech.wav" in fake.heard  # 확인 · 질문 둘 다
    assert ANSWER in fake.spoken  # 브라우저 목소리가 아니라 서버 TTS 가 읽었다
    voice = b"".join(pcm for ch, pcm in sounds if ch == TTS_CHANNEL)
    assert len(voice) >= 2 * 4800
    assert any(ch != TTS_CHANNEL for ch, _ in sounds)  # 시작음 · 종료음
    with store.session() as db:
        turns = db.query(SessionTurn).all()
        assert [t.question_text for t in turns] == ["이거 뭐야?"]


def test_wake_rejected_goes_back_quietly(ws_app):
    """확인 구간에 "자비서"가 없으면(TV 소리) 대답 · 되묻기 없이 대기로 돌아간다."""
    make, store, fake = ws_app
    fake.wake = "오늘 날씨는 맑겠습니다"
    sounds: list = []
    with make() as c, c.websocket_connect("/api/voice/ws") as ws:
        until(ws, "ready", sounds)
        ws.send_bytes(HEADER.pack(AUDIO, 0) + recording().tobytes())
        ws.send_json({"type": "wake", "id": 3, "pos": 11200, "score": 0.98, "camera": False})
        seen = until(ws, "idle", sounds)
    # 말끝 판정이 확인 결과보다 먼저 나면 thinking 이 끼기도 한다(노트북도 종료음이 난다)
    assert [m["type"] for m in seen if m["type"] != "thinking"] == ["wake", "rejected", "idle"]
    assert ANSWER not in fake.spoken


def test_button_call_skips_wake_check(ws_app):
    """호출 버튼으로 부르면 "자비서"를 말하지 않았으니 2단계 확인을 하지 않는다."""
    make, store, fake = ws_app
    fake.wake = "아무 말"  # 확인했다면 거절됐을 것
    sounds: list = []
    with make() as c, c.websocket_connect("/api/voice/ws") as ws:
        until(ws, "ready", sounds)
        ws.send_bytes(HEADER.pack(AUDIO, 0) + recording().tobytes())
        wake = {"type": "wake", "id": 1, "pos": 11200, "score": 1.0, "button": True}
        ws.send_json(wake | {"camera": False})
        seen = until(ws, "idle", sounds)
    assert "result" in [m["type"] for m in seen]
    assert "wake.wav" not in fake.heard


def test_typed_question_is_read_by_server_voice(ws_app):
    """입력칸 질문도 서버 목소리로 읽는다(브라우저 목소리를 쓰지 않는다)."""
    make, store, fake = ws_app
    sounds: list = []
    with make() as c, c.websocket_connect("/api/voice/ws") as ws:
        until(ws, "ready", sounds)
        ws.send_json({"type": "ask", "question": "오늘 뭐 했지?"})
        seen = until(ws, "idle", sounds)
    result = next(m for m in seen if m["type"] == "result")["data"]
    assert result["question"] == "오늘 뭐 했지?"
    assert ANSWER in fake.spoken
    assert fake.heard == []  # 받아쓰기는 안 했다


def test_without_key_says_off(tmp_path, monkeypatch):
    """OpenAI 키가 없으면 켤 수 없다고 알리고 닫는다(페이지는 입력칸만 쓴다)."""
    from jarviseo import config

    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    store = MemoryStore(database_url=f"sqlite:///{tmp_path / 'web.db'}")
    store.init_schema()
    user = store.ensure_user()
    web = WebAssistant(store, None, user, vlm=FakeVLMClient(["네."]), detector=NoObjects())
    app = main.create_app(store=store, checker=_NoChecker(), assistant=web)
    with TestClient(app) as c, c.websocket_connect("/api/voice/ws") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "off"


def test_barge_in_while_reading_typed_answer_opens_next_turn(ws_app):
    """대답을 읽는 중 "자비서"로 부르면 읽기를 끊고(flush) 그 호출로 새 질문을 받는다(BARGE_IN)."""
    make, store, fake = ws_app
    chunk = b"\x10\x00" * 2400

    @contextmanager
    def long_speech(**kwargs):  # 첫 대답은 3초 동안 읽는다
        fake.spoken.append(kwargs["input"])
        yield SimpleNamespace(iter_bytes=lambda size: iter([chunk] * 30))

    fake.audio.speech.with_streaming_response.create = long_speech
    sounds: list = []
    with make() as c, c.websocket_connect("/api/voice/ws") as ws:
        until(ws, "ready", sounds)
        ws.send_json({"type": "ask", "question": "길게 말해 줘"})
        while not any(ch == TTS_CHANNEL for ch, _ in sounds):  # 대답을 읽기 시작했다
            until_sound = ws.receive()
            if until_sound.get("bytes"):
                sounds.append((until_sound["bytes"][0], until_sound["bytes"][1:]))
        ws.send_bytes(HEADER.pack(AUDIO, 0) + recording().tobytes())
        ws.send_json({"type": "wake", "id": 1, "pos": 11200, "score": 0.99, "camera": False})
        seen = until(ws, "idle", sounds)
    kinds = [m["type"] for m in seen]
    assert "flush" in kinds  # 브라우저에 남은 대답 소리를 버리라고 했다
    results = [m["data"] for m in seen if m["type"] == "result"]
    assert results[0]["interrupted"] is True
    assert results[-1]["question"] == "이거 뭐야?"
    assert {"type": "wake", "barge_in": True} in seen
    with store.session() as db:
        turns = db.query(SessionTurn).order_by(SessionTurn.turn_id).all()
        assert [t.trigger_type for t in turns][-1] == "BARGE_IN"


def test_wake_word_only_reprompts_then_hears_question(ws_app):
    """ "자비서." 하고 쉬면 서버가 "네, 말씀하세요"를 읽고 시작음을 낸 뒤 질문을 다시 듣는다."""
    import time

    make, store, fake = ws_app
    fake.question = ["자비서.", "이거 뭐야?"]
    audio = recording()
    # "자비서" + 4초 쉼. 서버 VAD 는 받은 소리로 시간을 세서, 브라우저처럼 쉬는 소리도 보내야 한다
    wake_only = np.concatenate([audio[:11200], np.zeros(64000, np.int16)])
    question = np.concatenate([audio[19200:], np.zeros(16000, np.int16)])
    sounds: list = []
    with make() as c, c.websocket_connect("/api/voice/ws") as ws:
        until(ws, "ready", sounds)
        ws.send_bytes(HEADER.pack(AUDIO, 0) + wake_only.tobytes())
        ws.send_json({"type": "wake", "id": 1, "pos": 11200, "score": 0.99, "camera": False})
        while not (  # "네, 말씀하세요"(대답 채널) 다음 시작음(채널 1)이 나면 말한다
            any(ch == TTS_CHANNEL for ch, _ in sounds) and sounds[-1][0] != TTS_CHANNEL
        ):
            m = ws.receive()
            if m.get("bytes"):
                sounds.append((m["bytes"][0], m["bytes"][1:]))
        pos = len(wake_only)
        for i in range(0, len(question), 1280):  # 브라우저처럼 80ms 마다 실시간으로
            ws.send_bytes(HEADER.pack(AUDIO, pos + i) + question[i : i + 1280].tobytes())
            time.sleep(0.08)
        seen = until(ws, "idle", sounds)
    result = next(m for m in seen if m["type"] == "result")["data"]
    assert result["question"] == "이거 뭐야?"
    assert "네, 말씀하세요." in fake.spoken


class NeverReads:
    """늘 성분표를 못 읽는 판정기."""

    synonyms: dict = {}

    def check(self, image, allergens, *args, **kwargs):
        from jarviseo.nutrition.response import RETAKE_SHOW_INGREDIENTS
        from jarviseo.nutrition.service import IngredientCheck
        from jarviseo.types import AllergenJudgement, AllergenVerdict

        j = AllergenJudgement(verdict=AllergenVerdict.UNDETERMINED, reason="no_ingredients")
        msg = RETAKE_SHOW_INGREDIENTS
        return IngredientCheck(judgement=j, message=msg, retake=msg)

    def close(self):
        pass


def test_ingredient_retake_ladder_like_web_question(tmp_path):
    """성분표를 잇달아 못 읽으면 두 번째부터 다시 찍어 달라고 한다(웹 질문 · 음성 모드와 같다)."""
    import base64

    import cv2

    from jarviseo.nutrition.response import RETAKE_AGAIN, RETAKE_SHOW_INGREDIENTS

    store = MemoryStore(database_url=f"sqlite:///{tmp_path / 'web.db'}")
    store.init_schema()
    user = store.ensure_user()
    web = WebAssistant(store, None, user, vlm=FakeVLMClient(["네."]), detector=NoObjects())
    fake = FakeOpenAI()
    app = main.create_app(store=store, checker=NeverReads(), assistant=web, openai_client=fake)
    image = np.full((240, 320, 3), 200, np.uint8)
    url = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", image)[1]).decode()
    sounds: list = []
    with TestClient(app) as c, c.websocket_connect("/api/voice/ws") as ws:
        until(ws, "ready", sounds)
        for _ in range(2):
            ws.send_json({"type": "ask", "question": "이거 먹어도 돼?", "image": url})
            until(ws, "idle", sounds)
    assert RETAKE_SHOW_INGREDIENTS in fake.spoken and RETAKE_AGAIN in fake.spoken
    assert fake.spoken.index(RETAKE_SHOW_INGREDIENTS) < fake.spoken.index(RETAKE_AGAIN)
