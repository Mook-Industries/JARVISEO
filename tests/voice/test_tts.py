"""TTS 가 요청을 맞게 보내고, 읽을 글자를 맞게 다듬고, 받은 조각을 재생하는지 확인한다.

OpenAI API 와 스피커는 부르지 않는다.
"""

import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import httpx2
import pytest

from jarviseo import config
from jarviseo.voice.tts import (
    CHUNK_BYTES,
    READ_TIMEOUT_SEC,
    RETRIES,
    SAMPLE_RATE,
    TRIM_PAD_SEC,
    TextToSpeech,
)
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
        def __init__(self, samplerate, device, channels, dtype, latency):
            assert (samplerate, channels, dtype, latency) == (24000, 1, "int16", "high")

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


def test_synthesize_requests_pcm_and_yields_chunks():
    """`synthesize()` 는 다듬은 글자를 PCM 으로 요청하고 조각을 그대로 낸다."""
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


def test_speak_plays_chunks_and_returns_first_chunk_time(played):
    """`speak()` 는 받은 조각을 순서대로 재생하고 첫 조각 시간을 돌려준다."""
    tts = TextToSpeech(client=FakeClient())

    took = tts.speak("안녕하세요.")

    # 스피커에는 20ms 씩 나눠 쓴다
    assert b"".join(played) == b"\x01" * CHUNK_BYTES + b"\x02" * CHUNK_BYTES
    assert took == tts.last_latency_ms >= 0
    assert tts.last_interrupted is False


def test_stop_closes_stream_without_rest(played):
    """`stop()` 하면 남은 조각을 받지 않고 응답 스트림을 닫는다."""

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


def test_broken_stream_plays_received_part_without_raising(played):
    """받는 도중 끊기면 받은 만큼만 재생하고 예외를 올리지 않는다."""

    def cut(size):
        yield b"\x01" * size
        raise httpx2.ReadTimeout("끊김")  # openai 가 감싸지 않고 그대로 올라온다

    took = TextToSpeech(client=FakeClient(cut)).speak("안녕하세요.")

    assert b"".join(played) == b"\x01" * CHUNK_BYTES
    assert took is not None


def test_no_first_chunk_plays_nothing_and_returns_none(played):
    """첫 조각도 못 받으면 아무것도 재생하지 않고 None 을 준다."""

    def dead(size):
        raise httpx2.ReadTimeout("끊김")
        yield

    client = FakeClient(dead)

    assert TextToSpeech(client=client).speak("안녕하세요.") is None
    assert played == []
    assert len(client.requests) == 2  # 한 번 더 요청해 보고 포기했다


def test_stalled_before_first_chunk_retries(played):
    """첫 조각 전에 멈췄으면 새로 요청해서 읽는다."""

    def stalled_once(size):
        client.chunks = two_chunks  # 다음 요청은 정상
        raise httpx2.ReadTimeout("응답을 받기 시작한 뒤 서버가 멈췄다")
        yield

    client = FakeClient(stalled_once)
    tts = TextToSpeech(client=client)

    took = tts.speak("안녕하세요.")

    assert len(client.requests) == 2
    assert b"".join(played) == b"" * CHUNK_BYTES + b"" * CHUNK_BYTES
    assert took == tts.last_latency_ms >= 0


def test_first_chunk_wait_is_short_with_one_retry(monkeypatch):
    """첫 조각을 기다리는 시간은 짧게 걸고 한 번만 다시 보낸다."""
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")  # 요청은 보내지 않는다

    client = TextToSpeech().client

    assert client.timeout.read == READ_TIMEOUT_SEC == 4.0
    assert client.max_retries == RETRIES == 1


def slow_chunks(delay):
    """첫 조각이 delay 초 뒤에 오는 합성. 실제 TTS 의 첫 조각 왕복(약 0.9초)을 흉내 낸다."""

    def chunks(size):
        time.sleep(delay)
        yield from two_chunks(size)

    return chunks


def test_speak_uses_prefetched_chunks(played):
    """`prefetch()` 해 두면 `speak()` 가 새로 요청하지 않고 받아 둔 조각을 바로 읽는다."""
    client = FakeClient(slow_chunks(0.2))
    tts = TextToSpeech(client=client)

    tts.prefetch("컵이에요.")
    time.sleep(0.4)  # "잠깐만요"를 읽는 동안 첫 조각이 왔다
    took = tts.speak("컵이에요.")

    assert len(client.requests) == 1
    assert b"".join(played) == b"" * CHUNK_BYTES + b"" * CHUNK_BYTES
    assert took == tts.last_latency_ms < 100  # 첫 조각 왕복(200ms)을 다시 기다리지 않았다
    assert client.closed


def test_speak_without_prefetch_waits_round_trip(played):
    """`prefetch()` 없이 `speak()` 하면 첫 조각 왕복을 그대로 기다린다."""
    took = TextToSpeech(client=FakeClient(slow_chunks(0.2))).speak("컵이에요.")

    assert took >= 150  # 윈도우의 time.monotonic() 은 15.6ms 단위라 200 보다 조금 작게 나온다


def test_speak_other_text_discards_prefetch(played):
    """다른 글자로 `speak()` 하면 미리 받던 것은 버리고 새로 합성한다."""
    client = FakeClient()
    tts = TextToSpeech(client=client)

    tts.prefetch("버린 답이에요.")
    tts.speak("새 답이에요.")

    assert sorted(r["input"] for r in client.requests) == ["버린 답이에요.", "새 답이에요."]
    assert b"".join(played) == b"" * CHUNK_BYTES + b"" * CHUNK_BYTES  # 한 번만 읽었다


def test_failed_prefetch_makes_speak_silent_none(played):
    """미리 합성이 실패하면 `speak()` 도 소리 없이 None 을 준다."""

    def dead(size):
        raise httpx2.ReadTimeout("끊김")
        yield

    tts = TextToSpeech(client=FakeClient(dead))
    tts.prefetch("컵이에요.")

    assert tts.speak("컵이에요.") is None
    assert played == []


@pytest.mark.parametrize(
    ("speed", "value"),
    [("SLOW", 0.85), ("FAST", 1.2), ("fast", 1.2), ("NORMAL", None), ("모르는값", None)],
)
def test_converts_tts_speed_to_speed(speed, value):
    """`tts_speed` 를 `speed` 값으로 바꿔 보낸다."""
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
def test_speaks_units_commas_tildes_and_strips_markdown(raw, want):
    """단위 · 쉼표 · 물결표를 읽는 말로 바꾸고 마크다운을 뺀다."""
    assert normalize(raw) == want


def test_splits_at_sentence_end_and_newline_not_decimal():
    """문장 끝과 줄바꿈에서 자르고 소수점은 안 자른다."""
    text = "우유가 들어 있어요. 나트륨은 3.5그램이에요! 괜찮을까요?\n네"

    assert split_sentences(text) == [
        "우유가 들어 있어요.",
        "나트륨은 3.5그램이에요!",
        "괜찮을까요?",
        "네",
    ]
    assert split_sentences("") == []


def test_prepare_synthesizes_all_and_play_replays(played):
    """`prepare()` 는 끝까지 합성해 두고 `play()` 가 그대로 재생한다."""
    tts = TextToSpeech(client=FakeClient())

    pcm = tts.prepare("잠깐만요.")
    assert pcm == b"\x01" * CHUNK_BYTES + b"\x02" * CHUNK_BYTES
    assert played == []  # 만들어 두기만 하고 소리는 안 냈다

    tts.play(pcm)
    assert b"".join(played) == pcm
    assert tts.last_interrupted is False


def test_failed_prepare_is_empty_and_play_is_silent(played):
    """미리 합성이 실패하면 빈 바이트이고 `play()` 는 소리를 안 낸다."""

    def dead(size):
        raise httpx2.ReadTimeout("끊김")
        yield

    tts = TextToSpeech(client=FakeClient(dead))

    assert tts.prepare("잠깐만요.") == b""
    tts.play(b"")
    assert played == []


def test_prepare_trims_silence(played):
    """`prepare()` 는 앞뒤 무음을 잘라 둔다."""
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
