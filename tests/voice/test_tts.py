"""TTS 가 요청을 맞게 보내고, 읽을 글자를 맞게 다듬고, 받은 조각을 재생하는지 확인한다.

OpenAI API 와 스피커는 부르지 않는다.
"""

import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import httpx2
import numpy as np
import openai
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


def wait_until(condition, timeout=2.0):
    """condition() 이 True 가 될 때까지 기다린다. 받는 스레드가 따로 돌아서 바로 안 보일 수 있다."""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "시간 안에 안 됐다"
        time.sleep(0.01)


def stop_when(tts, condition, until=None):
    """condition() 이 True 가 되면 다른 스레드에서 `tts.stop()` 한다(호출어 감지).

    until(Event) 을 주면 그것이 설정될 때까지 20ms 마다 다시 부른다. play() 가 시작하기 전에
    온 stop() 은 play() 가 지우므로, 첫 조각을 기다리는 중에 닿게 하려는 것이다.
    """

    def run():
        wait_until(condition)
        tts.stop()
        while until is not None and not until.wait(0.02):
            tts.stop()

    threading.Thread(target=run, daemon=True).start()


def test_stop_while_waiting_next_chunk_closes_stream_without_rest(played):
    """다음 조각을 기다리는 동안 `stop()` 하면 바로 돌아오고, 남은 조각은 받지 않고 닫는다."""
    stalled, received = threading.Event(), []

    def chunks(size):
        yield b"\x01" * size
        stalled.wait(5)  # 서버가 멈췄다
        yield b"\x02" * size
        received.append("끊긴 뒤에 더 받음")

    client = FakeClient(chunks)
    tts = TextToSpeech(client=client)
    stop_when(tts, lambda: len(b"".join(played)) == CHUNK_BYTES)  # 첫 조각을 다 쓴 뒤 불렀다

    started = time.monotonic()
    took = tts.speak("안녕하세요.")

    assert time.monotonic() - started < 1.0  # 멈춘 서버(5초)를 기다리지 않았다
    assert b"".join(played) == b"\x01" * CHUNK_BYTES
    assert tts.last_interrupted is True
    assert took == tts.last_latency_ms >= 0

    stalled.set()  # 서버가 다시 보내도
    wait_until(lambda: client.closed)
    assert received == []

    # 다음 speak() 은 처음부터 끝까지 읽는다
    client.chunks = two_chunks
    played.clear()
    tts.speak("다시")
    assert tts.last_interrupted is False
    assert b"".join(played) == b"\x01" * CHUNK_BYTES + b"\x02" * CHUNK_BYTES


def test_stop_while_waiting_first_chunk_returns_at_once(played):
    """첫 조각을 기다리는 동안 `stop()` 하면 서버를 기다리지 않고 소리 없이 돌아온다."""
    slow = threading.Event()

    def chunks(size):
        slow.wait(5)  # 첫 조각이 늦다(#105 실측 11~16초)
        yield from two_chunks(size)

    client = FakeClient(chunks)
    tts = TextToSpeech(client=client)
    # 요청을 보내고 첫 조각을 기다리는 중에 불렀다. 요청은 받는 스레드가 play() 보다 먼저 보낸다
    returned = threading.Event()
    stop_when(tts, lambda: client.requests, until=returned)

    started = time.monotonic()
    took = tts.speak("안녕하세요.")
    returned.set()

    assert time.monotonic() - started < 1.0
    assert took is None
    assert played == []
    assert tts.last_interrupted is True
    assert len(client.requests) == 1
    slow.set()


def test_stop_then_stall_does_not_request_again(played):
    """첫 조각 전에 끊긴 뒤 합성이 멈춰도 다시 요청하지 않는다.

    다시 요청하면 끼어든 사용자의 새 질문 위로 앞 대답을 처음부터 읽고, 끊긴 턴으로도 안 남았다.
    """

    def stalled(size):
        time.sleep(0.1)  # 첫 조각을 기다리는 중
        client.chunks = two_chunks  # 다시 요청하면 이번에는 정상으로 온다
        tts.stop()  # 끊은 바로 그때 읽기 타임아웃이 났다
        raise httpx2.ReadTimeout("첫 조각 전에 서버가 멈췄다")
        yield

    client = FakeClient(stalled)
    tts = TextToSpeech(client=client)

    took = tts.speak("안녕하세요.")
    time.sleep(0.2)  # 다시 요청했다면 이때까지 보냈다

    assert len(client.requests) == 1
    assert played == []
    assert took is None
    assert tts.last_interrupted is True


def test_stall_while_reading_requests_fallback_and_reads_from_start(played):
    """읽는 도중 합성이 멈추면 폴백 모델로 한 번 더 요청해 처음부터 끝까지 읽는다(#130)."""

    def stalled(size):
        yield b"\x01" * size
        client.chunks = two_chunks  # 다시 요청하면 이번에는 끝까지 온다
        raise httpx2.ReadTimeout("읽는 도중 서버가 멈췄다")  # openai 가 감싸지 않고 그대로 올라온다

    client = FakeClient(stalled)
    tts = TextToSpeech(voice="alloy", speed="FAST", client=client)

    took = tts.speak("컵이에요.")

    assert [r["model"] for r in client.requests] == [config.TTS_MODEL, config.TTS_FALLBACK_MODEL]
    assert client.requests[0] | {"model": None} == client.requests[1] | {"model": None}
    # 앞부분이 한 번 더 나오고 끝까지 읽었다
    assert b"".join(played) == b"\x01" * CHUNK_BYTES * 2 + b"\x02" * CHUNK_BYTES
    assert took == tts.last_latency_ms >= 0  # 처음 받은 첫 조각 시간
    assert tts.last_interrupted is False


def test_stall_again_after_fallback_plays_received_part_without_raising(played):
    """폴백 요청도 도중에 멈추면 받은 만큼만 재생하고 예외를 올리지 않는다. 세 번째 요청은 없다."""

    def cut(size):
        yield b"\x01" * size
        raise httpx2.ReadTimeout("끊김")

    client = FakeClient(cut)
    took = TextToSpeech(client=client).speak("안녕하세요.")

    assert len(client.requests) == 2
    assert b"".join(played) == b"\x01" * CHUNK_BYTES * 2
    assert took is not None


def test_stop_then_stall_while_reading_does_not_request_again(played):
    """읽는 도중 끊긴 뒤 합성이 멈춰도 다시 요청하지 않는다."""

    def stalled(size):
        yield b"\x01" * size
        wait_until(lambda: len(b"".join(played)) == size)  # 첫 조각을 다 읽었다
        client.chunks = two_chunks
        tts.stop()  # 끊은 바로 그때 읽기 타임아웃이 났다
        raise httpx2.ReadTimeout("읽는 도중 서버가 멈췄다")

    client = FakeClient(stalled)
    tts = TextToSpeech(client=client)

    tts.speak("안녕하세요.")
    time.sleep(0.2)  # 다시 요청했다면 이때까지 보냈다

    assert len(client.requests) == 1
    assert b"".join(played) == b"\x01" * CHUNK_BYTES
    assert tts.last_interrupted is True


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
    """첫 조각 전에 멈췄으면 폴백 모델로 새로 요청해서 읽는다."""

    def stalled_once(size):
        client.chunks = two_chunks  # 다음 요청은 정상
        raise httpx2.ReadTimeout("응답을 받기 시작한 뒤 서버가 멈췄다")
        yield

    client = FakeClient(stalled_once)
    tts = TextToSpeech(client=client)

    took = tts.speak("안녕하세요.")

    assert [r["model"] for r in client.requests] == [config.TTS_MODEL, config.TTS_FALLBACK_MODEL]
    assert b"".join(played) == b"" * CHUNK_BYTES + b"" * CHUNK_BYTES
    assert took == tts.last_latency_ms >= 0


def test_no_response_head_falls_back(played):
    """응답 머리도 못 받아 openai 가 `APITimeoutError` 를 올리면 폴백 모델로 새로 요청한다."""

    def no_head(size):
        client.chunks = two_chunks
        request = httpx2.Request("POST", "https://api.openai.com/v1/audio/speech")
        raise openai.APITimeoutError(request=request)  # openai 가 RETRIES 만큼 다시 보낸 뒤
        yield

    client = FakeClient(no_head)
    tts = TextToSpeech(client=client)

    took = tts.speak("안녕하세요.")

    assert [r["model"] for r in client.requests] == [config.TTS_MODEL, config.TTS_FALLBACK_MODEL]
    assert b"".join(played) == b"" * CHUNK_BYTES + b"" * CHUNK_BYTES
    assert took is not None


def test_other_api_error_does_not_fall_back(played):
    """멈춤이 아닌 API 오류(크레딧 소진 429 등)는 다시 요청하지 않고 소리 없이 None 을 준다."""

    def refused(size):
        raise openai.OpenAIError("크레딧이 없다")
        yield

    client = FakeClient(refused)

    assert TextToSpeech(client=client).speak("안녕하세요.") is None
    assert len(client.requests) == 1
    assert played == []


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


def test_prefetched_stall_while_reading_falls_back(played):
    """미리 받던 대답이 "잠깐만요" 뒤 읽는 도중 멈춰도 폴백 모델로 처음부터 끝까지 읽는다.

    #117 실측 88 · 90 턴: 필러 뒤 대답이 "네"까지만 나오고 조용히 끝났다.
    """

    def stalled(size):
        yield b"" * size
        client.chunks = two_chunks
        raise httpx2.ReadTimeout("읽는 도중 서버가 멈췄다")

    client = FakeClient(stalled)
    tts = TextToSpeech(client=client)

    tts.prefetch("컵이에요.")
    time.sleep(0.1)  # "잠깐만요"를 읽는 동안 받기 시작했다
    took = tts.speak("컵이에요.")

    assert [r["model"] for r in client.requests] == [config.TTS_MODEL, config.TTS_FALLBACK_MODEL]
    assert b"".join(played) == b"" * CHUNK_BYTES * 2 + b"" * CHUNK_BYTES
    assert took is not None
    assert tts.last_interrupted is False


@pytest.mark.parametrize(
    ("speed", "value"),
    [("SLOW", 0.85), ("FAST", 1.2), ("fast", 1.2), ("NORMAL", None), ("모르는값", None)],
)
def test_converts_tts_speed_to_speed(speed, value):
    """`tts_speed` 를 `speed` 값으로 바꿔 보낸다."""
    client = FakeClient()

    list(TextToSpeech(speed=speed, client=client).synthesize("안녕하세요."))

    assert client.requests[0].get("speed") == value  # NORMAL 은 아예 안 보낸다


@pytest.mark.parametrize(("speed", "value"), [(1.5, 1.5), (0.5, 0.5), (1.0, None), (9, 4.0)])
def test_sends_speed_multiplier_within_openai_range(speed, value):
    """환경설정 배수는 그대로 보내고, OpenAI 범위(0.25~4) 밖이면 자른다."""
    client = FakeClient()

    list(TextToSpeech(speed=speed, client=client).synthesize("안녕하세요."))

    assert client.requests[0].get("speed") == value


def test_configure_changes_voice_and_speed_for_next_request():
    """`configure()` 는 아는 목소리 · 속도만 바꾸고, 합성 소리가 달라질 때만 True 다."""
    client = FakeClient()
    tts = TextToSpeech(client=client)

    assert tts.configure(voice="onyx", speed=1.5) is True
    list(tts.synthesize("안녕하세요."))
    assert (client.requests[0]["voice"], client.requests[0]["speed"]) == ("onyx", 1.5)
    assert tts.configure(voice="onyx", speed=1.5) is False  # 그대로다
    assert tts.configure(voice="모르는목소리", speed="빠르게") is False
    assert (tts.voice, tts.speed) == ("onyx", 1.5)
    assert tts.configure(volume=0.4) is False  # 소리 크기는 합성과 상관없다
    assert tts.volume == 0.4
    tts.configure(volume=3)
    assert tts.volume == 1.0


def test_volume_scales_played_sound(played):
    """소리 크기만큼 조각에 곱해 재생한다. 1 이면 받은 그대로."""
    loud = np.full(CHUNK_BYTES // 2, 1000, "<i2").tobytes()
    tts = TextToSpeech(client=FakeClient(chunks=lambda size: iter([loud])))

    tts.configure(volume=0.5)
    tts.speak("안녕하세요.")
    tts.play(loud)

    pcm = np.frombuffer(b"".join(played), "<i2")
    assert len(pcm) == CHUNK_BYTES and set(pcm.tolist()) == {500}


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
