"""Response Collector 가 늦은 답 앞에 "잠깐만요"를 읽고, 다시 부르면 늦은 답을 버리는지 확인한다.

스피커·API 는 부르지 않는다. 답이 늦는 것은 time.sleep 으로 흉내 낸다.
"""

import threading
import time

from jarviseo.voice import FakeTextToSpeech, ResponseCollector
from jarviseo.voice.collector import FILLER


def answer_after(sec, text="컵이에요."):
    def ask():
        time.sleep(sec)
        return text

    return ask


def test_fast_answer_skips_filler():
    """답이 빨리 오면 필러 없이 답만 준다."""
    tts = FakeTextToSpeech()
    collector = ResponseCollector(tts, after_sec=0.5)

    assert collector.collect(answer_after(0.0)) == "컵이에요."
    assert tts.spoken == []
    assert (collector.filler_sent, collector.dropped) == (False, False)


def test_slow_answer_reads_filler_first():
    """답이 늦으면 "잠깐만요"를 먼저 읽는다."""
    tts = FakeTextToSpeech()
    collector = ResponseCollector(tts, after_sec=0.05)

    assert collector.collect(answer_after(0.3)) == "컵이에요."
    assert tts.spoken == [FILLER]  # 한 번만 읽는다
    assert collector.filler_sent is True


def test_call_while_waiting_drops_late_answer():
    """기다리는 동안 다시 부르면 늦게 온 답은 버린다."""
    called = threading.Event()
    collector = ResponseCollector(FakeTextToSpeech(), called_since=lambda t: called.is_set())
    threading.Timer(0.1, called.set).start()  # 0.1초 뒤 "자비서"

    started = time.monotonic()
    assert collector.collect(answer_after(1.0)) is None
    assert collector.dropped is True
    assert time.monotonic() - started < 0.5  # 1초짜리 답을 끝까지 기다리지 않았다


def test_call_before_start_keeps_answer():
    """시작 전에 들린 호출로는 답을 버리지 않는다."""
    # 질문을 받아쓰는 동안 다시 잡힌 호출어처럼, 기다리기 시작하기 전의 감지는 상관없다
    before = time.monotonic()
    collector = ResponseCollector(FakeTextToSpeech(), called_since=lambda t: before > t)

    assert collector.collect(answer_after(0.0)) == "컵이에요."
    assert collector.dropped is False


def test_waits_only_without_prepared_filler():
    """필러를 미리 못 만들었으면 기다리기만 한다."""

    class NoPrepare(FakeTextToSpeech):
        def prepare(self, text):
            return b""  # 합성 실패

    tts = NoPrepare()
    collector = ResponseCollector(tts, after_sec=0.05)

    assert collector.collect(answer_after(0.2)) == "컵이에요."
    assert tts.spoken == []
    assert collector.filler_sent is False


def test_starts_prefetch_on_answer():
    """답이 오면 바로 미리 합성을 시작한다."""
    tts = FakeTextToSpeech()
    collector = ResponseCollector(tts, after_sec=0.05)

    assert collector.collect(answer_after(0.2)) == "컵이에요."
    assert tts.prefetched == ["컵이에요."]  # "잠깐만요"를 읽는 동안 대답 합성을 시작했다


def test_no_prefetch_without_answer():
    """답이 없으면 미리 합성하지 않는다."""
    tts = FakeTextToSpeech()

    assert ResponseCollector(tts).collect(lambda: None) is None
    assert tts.prefetched == []


def test_dropped_answer_is_not_prefetched_later():
    """버린 답은 다음 질문 중에 와도 미리 합성하지 않는다."""
    called_at = []
    tts = FakeTextToSpeech()
    collector = ResponseCollector(tts, called_since=lambda t: any(c > t for c in called_at))
    threading.Timer(0.05, lambda: called_at.append(time.monotonic())).start()

    assert collector.collect(answer_after(0.3, "버린 답")) is None  # 0.05초에 다시 불렀다
    assert collector.collect(answer_after(0.5, "새 답")) == "새 답"  # 버린 답은 0.3초에 온다

    assert tts.prefetched == ["새 답"]


def test_refresh_filler_prepares_new_voice_in_background():
    """`refresh_filler()` 는 "잠깐만요"를 다른 스레드에서 다시 만들고, 다음 필러부터 읽는다."""

    class CountingTTS(FakeTextToSpeech):
        made = 0

        def prepare(self, text):
            self.made += 1
            return f"{text}#{self.made}".encode()

    tts = CountingTTS()
    collector = ResponseCollector(tts, after_sec=0.05)
    collector.refresh_filler()
    deadline = time.monotonic() + 2
    while tts.made < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    collector.collect(answer_after(0.3))
    assert tts.spoken == [f"{FILLER}#2"]
