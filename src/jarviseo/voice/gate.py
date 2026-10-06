"""재생 중 마이크 게이트.  담당: 문태현

스피커 소리를 마이크가 다시 들으면 자기 목소리를 질문으로 받아쓴다(되먹임 루프).
그래서 재생이 시작되면 STT 쪽 마이크를 닫고, 재생이 끝나고 ``config.TTS_MIC_GATE_SEC``
(0.3초) 뒤에 다시 연다. 0.3초는 이미 나간 소리와 방 울림이 잦아들 시간이다.

닫혀 있던 동안 들어온 블록은 버리지 않고 무음으로 바꿔 넘긴다. 버리면 VAD 가 그 시간을
세지 못해서, 재생이 길면 ``listen()`` 의 timeout 이 재생 시간만큼 늘어난다.

호출어 감지는 막지 않는다(소프트 게이트). 재생 중에도 "자비서"로 끼어들 수 있어야 해서,
재생이 시작·끝날 때 콜백으로 알려 주기만 하고 판단은 호출어 쪽에 맡긴다.

스피커와 마이크는 하나씩이라 게이트도 프로세스에 하나(``MIC_GATE``)를 같이 쓴다.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import numpy as np

from jarviseo import config

__all__ = ["MIC_GATE", "MicGate"]


class MicGate:
    """재생은 ``with gate.playing():`` 으로 감싸고, 마이크 블록은 ``gate.mute()`` 로 거른다."""

    def __init__(self, grace_sec: float = config.TTS_MIC_GATE_SEC) -> None:
        self.grace_sec = grace_sec
        # 마지막으로 닫혀 있던 구간 [closed_at, open_at). 재생 중이면 open_at 은 무한대다.
        self._closed_at = self._open_at = -math.inf
        self._hooks: list[Callable[[bool], None]] = []

    def on_playback(self, hook: Callable[[bool], None]) -> None:
        """재생이 시작되면 ``hook(True)``, 끝나면 ``hook(False)`` 를 부르게 등록한다.

        호출어 쪽 소프트 게이트용이다. 재생 중에 임계값을 올리거나 재생음과 비교하는 것은
        훅을 받는 쪽이 정한다. 끝날 때는 grace_sec 를 기다리지 않고 바로 부른다.
        """
        self._hooks.append(hook)

    @contextmanager
    def playing(self) -> Iterator[None]:
        """with 블록 동안 재생 중으로 보고 마이크를 닫는다. 빠져나가고 grace_sec 뒤에 연다."""
        self._open_at = math.inf
        self._closed_at = time.monotonic()
        try:
            for hook in self._hooks:
                hook(True)
            yield
        finally:
            self._open_at = time.monotonic() + self.grace_sec
            for hook in self._hooks:
                hook(False)

    def is_open(self, t: float) -> bool:
        """t 시각(monotonic)에 마이크에 들어온 소리를 STT 에 넘겨도 되면 True."""
        return not self._closed_at <= t < self._open_at

    def mute(
        self, blocks: Iterable[tuple[np.ndarray, float]]
    ) -> Iterator[tuple[np.ndarray, float]]:
        """``Microphone.blocks()`` 를 받아, 닫혀 있던 동안 들어온 블록만 무음으로 바꿔 낸다."""
        for block, t in blocks:
            yield (block if self.is_open(t) else np.zeros_like(block)), t


MIC_GATE = MicGate()
