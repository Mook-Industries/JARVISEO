"""재생 중 끼어들기(barge-in).  담당: 문태현

호출어 감지기의 ``on_detect`` 로 넘긴다. "자비서"가 들리면 읽던 답을 바로 끊고(``tts.stop()``),
감지 시각을 모아 둔다. 음성 루프는 ``wait()`` 로 다음 호출을 받는다.

    barge = BargeIn(tts)
    detector = WakeWordDetector(mic, on_detect=barge.on_detect)
    detector.start()
    trigger = "WAKEWORD"
    while True:
        wake = barge.wait(timeout=0.5)
        if wake is None:
            continue
        utterance = stt.listen(since=wake.t)   # 호출어에 이어 말한 질문
        ...
        tts.speak(answer)                      # 도중에 "자비서"가 들리면 여기서 바로 돌아온다
        trigger = "BARGE_IN" if tts.last_interrupted else "WAKEWORD"  # 다음 턴의 trigger_type
        if not tts.last_interrupted:
            barge.clear()

재생 중에는 감지기가 임계값을 0.99 로 올려 자기 목소리에는 깨지 않게 한다(wakeword.py).
끊긴 뒤 게이트는 ``Speaker`` 가 유예 없이 바로 연다(gate.py ``reopen``).
"""

from __future__ import annotations

import logging
import queue
from typing import NamedTuple, Protocol

__all__ = ["BargeIn", "Wake"]

log = logging.getLogger(__name__)


class Wake(NamedTuple):
    """호출어 감지 한 번. t 는 호출어가 끝난 시각(monotonic), ``listen(since=t)`` 에 넘긴다."""

    t: float
    score: float


class _Stoppable(Protocol):
    def stop(self) -> None: ...


class BargeIn:
    """호출어가 들리면 읽던 답을 끊고, 그 호출을 다음 턴으로 넘긴다."""

    def __init__(self, tts: _Stoppable) -> None:
        """tts 는 ``TextToSpeech`` 나 ``FakeTextToSpeech``. stop() 만 쓴다."""
        self.tts = tts
        self._wakes: queue.Queue[Wake] = queue.Queue()

    def on_detect(self, t: float, score: float) -> None:
        """감지 스레드에서 불린다. 읽는 중이 아니면 tts.stop() 은 아무 일도 하지 않는다."""
        self.tts.stop()
        self._wakes.put(Wake(t, score))

    def wait(self, timeout: float | None = None) -> Wake | None:
        """다음 호출을 기다린다. 쌓인 게 여럿이면 가장 최근 것만 주고 나머지는 버린다.

        timeout 초 안에 없으면 None. 윈도우에서는 timeout 없이 기다리면 Ctrl+C 가 안 먹는다.
        """
        try:
            wake = self._wakes.get(timeout=timeout)
        except queue.Empty:
            return None
        while not self._wakes.empty():
            wake = self._wakes.get_nowait()
        return wake

    def clear(self) -> None:
        """쌓인 호출을 버린다. 답을 끝까지 읽은 뒤 부른다.

        질문을 받아쓰는 동안 다시 잡힌 호출어처럼, 끼어들기가 아닌 감지가 다음 턴을 열지 않게 한다.
        """
        while not self._wakes.empty():
            dropped = self._wakes.get_nowait()
            log.info("지난 호출어 감지를 버림 %.2f (점수 %.3f)", dropped.t, dropped.score)
