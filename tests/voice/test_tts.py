"""TTS 가 요청을 맞게 보내고, 읽을 글자를 맞게 다듬고, 받은 조각을 재생하는지 확인한다.

OpenAI API 와 스피커는 부르지 않는다.
"""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import httpx2
import pytest

from jarviseo import config
from jarviseo.voice.tts import CHUNK_BYTES, SAMPLE_RATE, TRIM_PAD_SEC, TextToSpeech
from jarviseo.voice.tts_text import normalize, split_sentences


def two_chunks(size):
    yield b"\x01" * size
    yield b"\x02" * size


class FakeClient:
    """openai 클라이언트 대신 요청을 받아 두고, chunks(size) 가 낸 PCM 조각을 흘려준다."""

    def __init__(self, chunks=two_chunks):
        self.requests = []
        self.chunks = chunks
        speech = SimpleNamespace(with_streaming_response=SimpleNamespace(create=self._create))
        self.audio = SimpleNamespace(speech=speech)

    @contextmanager
    def _create(self, **kwargs):
        self.requests.append(kwargs)
        self.closed = False
        try:
            yield SimpleNamespace(iter_bytes=self.chunks)
        finally:
            self.closed = True  # 응답 스트림을 닫았다


@pytest.fixture
def played(monkeypatch):
    """sounddevice 대신 스피커에 쓴 조각을 모은다. 리눅스 CI 에는 PortAudio 가 없다."""
    written = []

    class FakeOutputStream:
        def __init__(self, samplerate, device, channels, dtype):
            assert (samplerate, channels, dtype) == (24000, 1, "int16")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def write(self, data):
            written.append(data)

        def abort(self):
            pass

    sounddevice = SimpleNamespace(RawOutputStream=FakeOutputStream)
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    return written


def test_synthesize_는_다듬은_글자를_pcm_으로_요청하고_조각을_그대로_낸다():
    client = FakeClient()
    tts = TextToSpeech(voice="alloy", client=client)

    stream = tts.synthesize("나트륨 1,200mg 이에요.")
    assert tts.last_latency_ms is None  # 첫 조각을 받기 전
    chunks = list(stream)

    (req,) = client.requests
    assert (req["model"], req["voice"], req["response_format"]) == (
        config.TTS_MODEL,
        "alloy",
        "pcm",
    )
    assert req["input"] == "나트륨 1200밀리그램 이에요."
    assert chunks == [b"\x01" * CHUNK_BYTES, b"\x02" * CHUNK_BYTES]
    assert tts.last_latency_ms >= 0


def test_speak_는_받은_조각을_순서대로_재생하고_첫_조각_시간을_돌려준다(played):
    tts = TextToSpeech(client=FakeClient())

    took = tts.speak("안녕하세요.")

    # 스피커에는 20ms 씩 나눠 쓴다
    assert b"".join(played) == b"\x01" * CHUNK_BYTES + b"\x02" * CHUNK_BYTES
    assert took == tts.last_latency_ms >= 0
    assert tts.last_interrupted is False


def test_stop_하면_남은_조각을_받지_않고_응답_스트림을_닫는다(played):
    def chunks(size):
        yield b"\x01" * size
        tts.stop()  # 첫 조각을 쓰는 사이 호출어가 들렸다
        yield b"\x02" * size
        raise AssertionError("끊긴 뒤에는 더 받지 않는다")

    client = FakeClient(chunks)
    tts = TextToSpeech(client=client)

    took = tts.speak("안녕하세요.")

    assert b"".join(played) == b"\x01" * CHUNK_BYTES
    assert tts.last_interrupted is True
    assert client.closed
    assert took == tts.last_latency_ms >= 0

    # 다음 speak() 은 처음부터 끝까지 읽는다
    client.chunks = two_chunks
    tts.speak("다시")
    assert tts.last_interrupted is False


def test_받는_도중_끊기면_받은_만큼만_재생하고_예외를_올리지_않는다(played):
    def cut(size):
        yield b"\x01" * size
        raise httpx2.ReadTimeout("끊김")  # openai 가 감싸지 않고 그대로 올라온다

    took = TextToSpeech(client=FakeClient(cut)).speak("안녕하세요.")

    assert b"".join(played) == b"\x01" * CHUNK_BYTES
    assert took is not None


def test_첫_조각도_못_받으면_아무것도_재생하지_않고_None(played):
    def dead(size):
        raise httpx2.ReadTimeout("끊김")
        yield

    assert TextToSpeech(client=FakeClient(dead)).speak("안녕하세요.") is None
    assert played == []


@pytest.mark.parametrize(
    ("speed", "value"),
    [("SLOW", 0.85), ("FAST", 1.2), ("fast", 1.2), ("NORMAL", None), ("모르는값", None)],
)
def test_tts_speed_를_speed_값으로_바꿔_보낸다(speed, value):
    client = FakeClient()

    list(TextToSpeech(speed=speed, client=client).synthesize("안녕하세요."))

    assert client.requests[0].get("speed") == value  # NORMAL 은 아예 안 보낸다


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("열량 250kcal, 하루 권장량의 60%", "열량 250킬로칼로리, 하루 권장량의 60퍼센트"),
        ("탄수화물 3.5g이 들어 있어요.", "탄수화물 3.5그램이 들어 있어요."),
        ("3~5개, 1,234,567원", "3에서 5개, 1234567원"),
        ("**우유**와 `대두`\n- 땅콩", "우유와 대두\n땅콩"),
        ("5m 앞, 5min 뒤, 3,4번째", "5미터 앞, 5min 뒤, 3,4번째"),
    ],
)
def test_단위·쉼표·물결표를_읽는_말로_바꾸고_마크다운을_뺀다(raw, want):
    assert normalize(raw) == want


def test_문장_끝과_줄바꿈에서_자르고_소수점은_안_자른다():
    text = "우유가 들어 있어요. 나트륨은 3.5그램이에요! 괜찮을까요?\n네"

    assert split_sentences(text) == [
        "우유가 들어 있어요.",
        "나트륨은 3.5그램이에요!",
        "괜찮을까요?",
        "네",
    ]
    assert split_sentences("") == []


def test_prepare_는_끝까지_합성해_두고_play_가_그대로_재생한다(played):
    tts = TextToSpeech(client=FakeClient())

    pcm = tts.prepare("잠깐만요.")
    assert pcm == b"\x01" * CHUNK_BYTES + b"\x02" * CHUNK_BYTES
    assert played == []  # 만들어 두기만 하고 소리는 안 냈다

    tts.play(pcm)
    assert b"".join(played) == pcm
    assert tts.last_interrupted is False


def test_미리_합성이_실패하면_빈_바이트이고_play_는_소리를_안_낸다(played):
    def dead(size):
        raise httpx2.ReadTimeout("끊김")
        yield

    tts = TextToSpeech(client=FakeClient(dead))

    assert tts.prepare("잠깐만요.") == b""
    tts.play(b"")
    assert played == []


def test_prepare_는_앞뒤_무음을_잘라_둔다(played):
    quiet = bytes(SAMPLE_RATE * 2)  # 1초 무음
    voice = (3000).to_bytes(2, "little", signed=True) * (SAMPLE_RATE // 2)  # 0.5초 말소리

    def padded(size):
        yield quiet
        yield voice
        yield quiet

    def silent(size):
        yield quiet

    pcm = TextToSpeech(client=FakeClient(padded)).prepare("잠깐만요.")

    # 말소리와 앞뒤 여유만 남는다. 무음 2초를 다 읽으면 대답이 그만큼 늦는다
    assert len(pcm) // 2 == SAMPLE_RATE // 2 + 2 * round(TRIM_PAD_SEC * SAMPLE_RATE)
    assert TextToSpeech(client=FakeClient(silent)).prepare("잠깐만요.") == b""
