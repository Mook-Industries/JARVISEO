"""STT 가 요청을 맞게 보내고, 받아쓴 글자를 맞게 정리하는지 확인한다.

OpenAI API 와 마이크는 부르지 않는다. 가짜 client 로 바꿔 끼우고, VAD 는
1초짜리 발화 하나를 잡은 것으로 친다(VAD 자체는 test_voice_vad.py 가 본다).
"""

import io
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from openai import OpenAIError

from jarviseo import config
from jarviseo.voice import stt, vad
from jarviseo.voice.chime import START_WAV, load_sound
from jarviseo.voice.gate import SOUND_TAIL_SEC, MicGate
from jarviseo.voice.mic import BLOCK, SAMPLE_RATE, MicStream
from jarviseo.voice.speaker import SAMPLE_RATE as SPEAKER_RATE
from jarviseo.voice.transcript import clean, is_question
from jarviseo.voice.vad import Speech

DATA = Path(__file__).parents[1] / "data"


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

    def record(blocks, on_speech_start=None, timeout=5.0, ignore=None):
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


def pause_with_chime():
    """호출어·질문 대신 "자비서, 저거 뭐야?" 녹음 → 1초 쉼(그 사이 시작음) → 같은 녹음을 만든다.

    (소리, 감지 시각, 시작음 구간, 질문 시작 시각) 을 돌려준다. 시각은 첫 블록이 0초다.
    시작음은 감지 0.15초 뒤에 말소리와 같은 크기로 섞는다(lookback.py --chime 0 과 같다).
    """
    with wave.open(str(DATA / "jarviseo_question.wav"), "rb") as f:
        voice = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float64)
    pcm = np.frombuffer(load_sound(START_WAV), np.int16).astype(np.float64)
    n = round(len(pcm) * SAMPLE_RATE / SPEAKER_RATE)
    chime = np.interp(np.arange(n) * SPEAKER_RATE / SAMPLE_RATE, np.arange(len(pcm)), pcm)
    chime *= np.sqrt(np.mean(voice**2) / np.mean(chime**2))
    wake_end = 1.0 + len(voice) / SAMPLE_RATE
    detected, chime_at, question_at = wake_end + 0.12, wake_end + 0.27, wake_end + 1.0
    audio = np.random.default_rng(0).normal(0, 20, round((question_at + 3.0) * SAMPLE_RATE))
    for at, sound in [(1.0, voice), (chime_at, chime), (question_at, voice)]:
        i = round(at * SAMPLE_RATE)
        audio[i : i + len(sound)] += sound
    span = (chime_at, chime_at + len(chime) / SAMPLE_RATE + SOUND_TAIL_SEC)
    return audio.clip(-32768, 32767).astype(np.int16), detected, span, question_at


@pytest.mark.parametrize("marked", [False, True])
def test_listen_after_pause_skips_chime_between_wake_word_and_question(monkeypatch, marked):
    """호출어 뒤 쉬는 틈에 시작음이 나도, 표시된 구간은 말로 안 봐서 질문부터 잡는다(#104).

    표시하지 않으면 시작음이 쉼을 메워 호출어부터 한 발화가 되고, 말 시작이 호출어 쪽으로 잡혀
    사진도 그때 것을 고른다. 표시는 Speaker 가 시작음을 낼 때 한다(gate.sounding).
    """
    audio, detected, span, question_at = pause_with_chime()
    gate = MicGate()
    monkeypatch.setattr(stt, "MIC_GATE", gate)
    if marked:
        gate.mark_sound(*span)
    blocks = [(audio[i : i + BLOCK], i / SAMPLE_RATE) for i in range(0, len(audio) - BLOCK, BLOCK)]
    starts = []

    with MicStream(source=blocks, history_sec=len(audio) / SAMPLE_RATE + 1) as mic:
        s = stt.SpeechToText(mic=mic, client=FakeClient())
        utterance = s.listen(on_speech_start=starts.append, timeout=2.0, since=detected)

    assert utterance.text == "저거 뭐야?"
    if marked:
        assert utterance.started_at == pytest.approx(question_at, abs=0.1)
    else:
        assert utterance.started_at == detected - stt.WAKE_LAG_SEC  # 호출어 끝으로 잡혔다


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
