"""마이크 입력.  담당: 문태현

16kHz 모노 int16 을 30ms 블록으로 읽고, 블록마다 첫 샘플이 마이크에 들어온 시각
(``time.monotonic()`` 기준)을 같이 낸다.

시각은 첫 블록에서 한 번 정하고, 그 뒤로는 읽은 샘플 수로 센다. 블록마다
``time.monotonic()`` 을 찍으면 윈도우(파이썬 3.12)에서는 15.6ms 단위로 뭉치고,
드라이버가 블록을 몰아서 줄 때마다 시각이 흔들린다.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterator

import numpy as np

__all__ = ["BLOCK", "SAMPLE_RATE", "Microphone"]

SAMPLE_RATE = 16000
BLOCK = 480  # 30ms. webrtcvad 가 받는 길이는 10·20·30ms 뿐이다


class Microphone:
    """``with Microphone() as mic: for block, t in mic.blocks(): ...``"""

    def __init__(self, device: int | None = None) -> None:
        self.device = device

    def __enter__(self) -> Microphone:
        # 리눅스 CI 에는 PortAudio 가 없어서, 맨 위에서 import 하면 이 모듈을 읽기만 해도 죽는다.
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK, device=self.device, channels=1, dtype="int16"
        )
        self._stream.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stream.close()

    def blocks(self) -> Iterator[tuple[np.ndarray, float]]:
        """(블록, 그 블록 첫 샘플의 시각) 을 계속 낸다. 그만 받으려면 for 문을 빠져나온다."""
        block, _ = self._stream.read(BLOCK)
        # 방금 읽은 블록 길이만큼, 그리고 장치가 알려 준 입력 지연만큼 거슬러 올라간다.
        t0 = time.monotonic() - BLOCK / SAMPLE_RATE - self._stream.latency
        for n in itertools.count():
            yield block[:, 0], t0 + n * BLOCK / SAMPLE_RATE
            block, overflowed = self._stream.read(BLOCK)
            if overflowed:  # 놓친 샘플이 있으면 샘플 수로 센 시각이 밀리므로 다시 맞춘다
                t0 = time.monotonic() - (n + 2) * BLOCK / SAMPLE_RATE - self._stream.latency
