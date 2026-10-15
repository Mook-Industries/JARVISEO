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


def test_답이_빨리_오면_필러_없이_답만_준다():
    tts = FakeTextToSpeech()
    collector = ResponseCollector(tts, after_sec=0.5)

    assert collector.collect(answer_after(0.0)) == "컵이에요."
    assert tts.spoken == []
    assert (collector.filler_sent, collector.dropped) == (False, False)


def test_답이_늦으면_잠깐만요를_먼저_읽는다():
    tts = FakeTextToSpeech()
    collector = ResponseCollector(tts, after_sec=0.05)

    assert collector.collect(answer_after(0.3)) == "컵이에요."
    assert tts.spoken == [FILLER]  # 한 번만 읽는다
    assert collector.filler_sent is True


def test_기다리는_동안_다시_부르면_늦게_온_답은_버린다():
    called = threading.Event()
    collector = ResponseCollector(FakeTextToSpeech(), called_since=lambda t: called.is_set())
    threading.Timer(0.1, called.set).start()  # 0.1초 뒤 "자비서"

    started = time.monotonic()
    assert collector.collect(answer_after(1.0)) is None
    assert collector.dropped is True
    assert time.monotonic() - started < 0.5  # 1초짜리 답을 끝까지 기다리지 않았다


def test_시작_전에_들린_호출로는_답을_버리지_않는다():
    # 질문을 받아쓰는 동안 다시 잡힌 호출어처럼, 기다리기 시작하기 전의 감지는 상관없다
    before = time.monotonic()
    collector = ResponseCollector(FakeTextToSpeech(), called_since=lambda t: before > t)

    assert collector.collect(answer_after(0.0)) == "컵이에요."
    assert collector.dropped is False


def test_필러를_미리_못_만들었으면_기다리기만_한다():
    class NoPrepare(FakeTextToSpeech):
        def prepare(self, text):
            return b""  # 합성 실패

    tts = NoPrepare()
    collector = ResponseCollector(tts, after_sec=0.05)

    assert collector.collect(answer_after(0.2)) == "컵이에요."
    assert tts.spoken == []
    assert collector.filler_sent is False


def test_답이_오면_바로_미리_합성을_시작한다():
    tts = FakeTextToSpeech()
    collector = ResponseCollector(tts, after_sec=0.05)

    assert collector.collect(answer_after(0.2)) == "컵이에요."
    assert tts.prefetched == ["컵이에요."]  # "잠깐만요"를 읽는 동안 대답 합성을 시작했다


def test_답이_없으면_미리_합성하지_않는다():
    tts = FakeTextToSpeech()

    assert ResponseCollector(tts).collect(lambda: None) is None
    assert tts.prefetched == []


def test_버린_답은_다음_질문_중에_와도_미리_합성하지_않는다():
    called_at = []
    tts = FakeTextToSpeech()
    collector = ResponseCollector(tts, called_since=lambda t: any(c > t for c in called_at))
    threading.Timer(0.05, lambda: called_at.append(time.monotonic())).start()

    assert collector.collect(answer_after(0.3, "버린 답")) is None  # 0.05초에 다시 불렀다
    assert collector.collect(answer_after(0.5, "새 답")) == "새 답"  # 버린 답은 0.3초에 온다

    assert tts.prefetched == ["새 답"]
