"""끼어들기(BargeIn)가 호출어를 받으면 읽던 답을 끊고, 그 호출을 다음 턴으로 넘기는지 확인한다.

마이크·스피커·API 는 부르지 않는다.
"""

from jarviseo.voice import BargeIn, Wake


class StopCounter:
    """tts 대신 stop() 이 불린 횟수만 센다."""

    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1


def test_호출어가_들리면_읽던_답을_끊고_그_호출을_넘긴다():
    tts = StopCounter()
    barge = BargeIn(tts)

    barge.on_detect(12.5, 0.995)

    assert tts.stops == 1
    assert barge.wait(timeout=0.1) == Wake(12.5, 0.995)
    assert barge.wait(timeout=0.01) is None  # 한 번 넘긴 호출은 다시 안 준다


def test_쌓인_호출이_여럿이면_가장_최근_것만_준다():
    barge = BargeIn(StopCounter())
    barge.on_detect(1.0, 0.98)
    barge.on_detect(4.0, 0.99)

    assert barge.wait(timeout=0.1).t == 4.0
    assert barge.wait(timeout=0.01) is None


def test_clear_는_쌓인_호출을_버린다():
    barge = BargeIn(StopCounter())
    barge.on_detect(1.0, 0.98)

    barge.clear()

    assert barge.wait(timeout=0.01) is None
