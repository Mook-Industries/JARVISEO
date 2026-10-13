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

재생 중에는 감지기가 임계값을 따로 쓴다(``config.WAKEWORD_PLAYBACK_THRESHOLD``, wakeword.py).
끊긴 뒤 게이트는 ``Speaker`` 가 유예 없이 끊은 시각에 열고, ``wait()`` 이 그 호출을 넘길 때
질문을 읽기 시작하는 곳(감지 ``config.WAKE_LOOKBACK_SEC`` 전)까지 거슬러 연다. 재생 중에
말한 "자비서"와 바로 이어 말한 질문 앞부분이 무음으로 바뀌지 않고 받아써진다(재생 소리가
같이 들어가도 ``clean()`` 이 호출어 앞을 지운다).
"""

from __future__ import annotations

import logging
import queue
from typing import NamedTuple, Protocol

from jarviseo import config
from jarviseo.voice.gate import MIC_GATE, MicGate

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

    def __init__(self, tts: _Stoppable, gate: MicGate | None = None) -> None:
        """tts 는 ``TextToSpeech`` 나 ``FakeTextToSpeech``. stop() 만 쓴다.

        gate 는 테스트에서만 넘긴다. 기본은 STT·TTS 와 같이 쓰는 ``MIC_GATE``.
        """
        self.tts = tts
        self.gate = gate or MIC_GATE
        self._wakes: queue.Queue[Wake] = queue.Queue()

    def on_detect(self, t: float, score: float) -> None:
        """감지 스레드에서 불린다. 읽는 중이 아니면 tts.stop() 은 아무 일도 하지 않는다."""
        self.tts.stop()
        self._wakes.put(Wake(t, score))

    def wait(self, timeout: float | None = None) -> Wake | None:
        """다음 호출을 기다린다. 쌓인 게 여럿이면 가장 최근 것만 주고 나머지는 버린다.

        재생 중에 들린 호출이면 게이트를 그 질문을 읽기 시작하는 곳까지 거슬러 연다.
        그래서 speak() 이 돌아온 뒤에 부른다(재생 중에 열면 재생이 끝날 때 다시 닫힌다).
        timeout 초 안에 없으면 None. 윈도우에서는 timeout 없이 기다리면 Ctrl+C 가 안 먹는다.
        """
        try:
            wake = self._wakes.get(timeout=timeout)
        except queue.Empty:
            return None
        while not self._wakes.empty():
            wake = self._wakes.get_nowait()
        if not self.gate.is_open(wake.t):
            self.gate.reopen(wake.t - config.WAKE_LOOKBACK_SEC)
        return wake

    def clear(self) -> None:
        """쌓인 호출을 버린다. 답을 끝까지 읽은 뒤 부른다.

        질문을 받아쓰는 동안 다시 잡힌 호출어처럼, 끼어들기가 아닌 감지가 다음 턴을 열지 않게 한다.
        """
        while not self._wakes.empty():
            dropped = self._wakes.get_nowait()
            log.info("지난 호출어 감지를 버림 %.2f (점수 %.3f)", dropped.t, dropped.score)
