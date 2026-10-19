"""STT 가 요청을 맞게 보내고, 받아쓴 글자를 맞게 정리하는지 확인한다.

OpenAI API 와 마이크는 부르지 않는다. 가짜 client 로 바꿔 끼우고, VAD 는
1초짜리 발화 하나를 잡은 것으로 친다(VAD 자체는 test_voice_vad.py 가 본다).
"""

import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from openai import OpenAIError

from jarviseo import config
from jarviseo.voice import stt, vad
from jarviseo.voice.mic import SAMPLE_RATE
from jarviseo.voice.transcript import clean, is_question
from jarviseo.voice.vad import Speech


class FakeClient:
    """openai 클라이언트 대신 요청을 받아 두고, 정해 둔 글자를 돌려준다."""

    def __init__(self, text="자비서, 저거 뭐야?", error=None):
        self.requests, self.text, self.error = [], text, error
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text)


@pytest.fixture
def heard(monkeypatch):
    """마이크를 열지 않고, VAD 가 100~101초에 발화 하나를 잡은 것으로 친다."""

    class Mic:
        def __init__(self, device):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def blocks(self):
            return iter(())

    def record(blocks, on_speech_start=None, timeout=5.0):
        if on_speech_start is not None:
            on_speech_start(100.0)
        return Speech(np.zeros(SAMPLE_RATE, np.int16), 100.0, 101.0)

    monkeypatch.setattr(stt, "Microphone", Mic)
    monkeypatch.setattr(vad, "record_speech", record)


def test_transcribe_sends_model_language_hints_and_wav():
    """`transcribe()` 는 모델 · 언어 · 호출어 힌트와 WAV 를 보낸다."""
    client = FakeClient()
    audio = np.arange(-800, 800, dtype=np.int16)

    utterance = stt.SpeechToText(client=client).transcribe(audio, 1.0, 2.0)

    (req,) = client.requests
    assert (req["model"], req["language"], req["prompt"]) == (
        config.STT_MODEL,
        "ko",
        config.WAKE_WORD,
    )
    name, data = req["file"]
    with wave.open(io.BytesIO(data), "rb") as w:
        assert name.endswith(".wav")
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SAMPLE_RATE, 1, 2)
        assert np.array_equal(np.frombuffer(w.readframes(w.getnframes()), "<i2"), audio)
    assert (utterance.text, utterance.started_at, utterance.ended_at) == (
        "자비서, 저거 뭐야?",
        1.0,
        2.0,
    )


def test_listen_strips_wake_word_and_keeps_raw_and_latency(heard):
    """`listen()` 은 호출어를 뺀 질문을 주고 원문과 지연을 남긴다."""
    starts = []
    s = stt.SpeechToText(client=FakeClient("아, 들려? 자비스 저거 뭐야?"))

    utterance = s.listen(on_speech_start=starts.append)

    assert (utterance.text, utterance.started_at, utterance.ended_at) == (
        "저거 뭐야?",
        100.0,
        101.0,
    )
    assert starts == [100.0]
    assert s.last_raw_text == "아, 들려? 자비스 저거 뭐야?"
    assert s.last_latency_ms >= 0


@pytest.mark.parametrize("raw", ["음...", "자비서", "어?"])
def test_listen_returns_none_for_filler_or_wake_word_only(heard, raw):
    """`listen()` 은 필러나 호출어뿐이면 None 을 준다."""
    s = stt.SpeechToText(client=FakeClient(raw))

    assert s.listen() is None
    assert s.last_raw_text == raw


@pytest.mark.parametrize(("raw", "want"), [("컵.", "컵."), ("자비서, 컵", "컵"), ("음...", None)])
def test_listen_accepts_one_letter_with_min_letters(heard, raw, want):
    """되묻기("컵이요, 병이요?")의 대답은 "컵" 한 글자일 수 있다. 필러뿐이면 그래도 None."""
    assert stt.SpeechToText(client=FakeClient(raw)).listen() is None

    utterance = stt.SpeechToText(client=FakeClient(raw)).listen(min_letters=1)

    assert (utterance and utterance.text) == want


def test_listen_uses_hints_instead_of_wake_word(heard):
    """되묻기 대답을 들을 때 후보 이름을 주면 "컵"이 "컷"으로 받아써지는 일이 준다."""
    client = FakeClient("컵.")

    utterance = stt.SpeechToText(client=client).listen(min_letters=1, hints=["컵", "병"])

    assert utterance.text == "컵."
    assert client.requests[0]["prompt"] == "컵, 병"


def test_listen_returns_none_on_transcription_failure(heard):
    """`listen()` 은 받아쓰기가 실패하면 None 을 준다."""
    s = stt.SpeechToText(client=FakeClient(error=OpenAIError("연결 실패")))

    assert s.listen() is None
    assert s.last_latency_ms is None


def test_listen_skips_api_without_speech(heard, monkeypatch):
    """`listen()` 은 말이 없으면 API 를 부르지 않는다."""
    monkeypatch.setattr(vad, "record_speech", lambda *args, **kwargs: None)
    client = FakeClient()

    assert stt.SpeechToText(client=client).listen() is None
    assert client.requests == []


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("자비서, 저거 뭐야?", "저거 뭐야?"),
        ("아, 들려? 자비스 저거 뭐야?", "저거 뭐야?"),
        ("저거 뭐야, 자비서?", "저거 뭐야"),
        ("자비서야, 아까 본 거 뭐였지?", "아까 본 거 뭐였지?"),
        ("음... 어 그 컵   뭐야", "그 컵 뭐야"),
        ("자비스럽게 말해 줘", "자비스럽게 말해 줘"),
        # 호출어 끝만 받아써진 토막 (listen 이 감지 시각보다 앞에서부터 읽는다)
        ("서, 오늘 날씨 어때?", "오늘 날씨 어때?"),
        ("비서 지금 몇 시야?", "지금 몇 시야?"),
        ("서울 날씨 어때?", "서울 날씨 어때?"),
    ],
)
def test_strips_wake_word_filler_and_spaces(raw, want):
    """호출어와 필러를 빼고 공백을 정리한다."""
    assert clean(raw) == want


def test_empty_or_one_letter_is_not_question():
    """비었거나 한 글자면 질문이 아니다."""
    assert not is_question("")
    assert not is_question("뭐?")
    assert is_question("뭐야?")
    assert is_question("컵", min_letters=1)
    assert not is_question("", min_letters=1)
