"""마이크 입력.  담당: 문태현

16kHz 모노 int16 을 30ms 블록으로 읽고, 블록마다 첫 샘플이 마이크에 들어온 시각
(``time.monotonic()`` 기준)을 같이 낸다.

시각은 첫 블록에서 한 번 정하고, 그 뒤로는 읽은 샘플 수로 센다. 블록마다
``time.monotonic()`` 을 찍으면 윈도우(파이썬 3.12)에서는 15.6ms 단위로 뭉치고,
드라이버가 블록을 몰아서 줄 때마다 시각이 흔들린다.

``Microphone`` 은 한 곳에서만 읽는다. 호출어 감지와 STT 처럼 여럿이 같이 읽을 때는
``MicStream`` 을 쓴다.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator

import numpy as np

__all__ = ["BLOCK", "SAMPLE_RATE", "MicStream", "Microphone"]

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


class MicStream:
    """마이크 하나를 열어 두고 여러 곳에서 같이 읽는다.

    ``with MicStream() as mic:`` 안에서 ``mic.blocks()`` 를 부를 때마다 따로 읽는 줄이 하나 생긴다.
    호출어 감지는 계속 읽고, STT 는 호출어를 잡은 시각부터 읽는다. 호출어 바로 뒤에 이어 말한
    질문이 잘리지 않도록 최근 history_sec 초는 들고 있다.
    """

    def __init__(
        self,
        device: int | None = None,
        history_sec: float = 3.0,
        source: Iterable[tuple[np.ndarray, float]] | None = None,
    ) -> None:
        """source 는 테스트에서 마이크 대신 (블록, 시각) 을 흘릴 때만 넘긴다. 다 흘리면 닫힌다."""
        self.device = device
        self._source = source
        # (번호, 블록, 시각). 번호는 그때까지 받은 블록 수이고, 읽는 쪽은 이 번호로 위치를 센다.
        self._held: deque[tuple[int, np.ndarray, float]] = deque(
            maxlen=round(history_sec * SAMPLE_RATE / BLOCK)
        )
        self._count = 0
        self._closed = False
        self._cond = threading.Condition()

    def __enter__(self) -> MicStream:
        # 마이크는 여기서 연다. 장치가 없으면 읽는 스레드가 아니라 부른 쪽에서 바로 오류가 난다.
        self._mic = None
        if self._source is None:
            self._mic = Microphone(self.device).__enter__()
            source = self._mic.blocks()
        else:
            source = iter(self._source)
        self._thread = threading.Thread(target=self._read, args=(source,), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """마이크를 닫는다. 읽던 쪽은 남은 블록을 다 받고 끝난다."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._thread.join()
        if self._mic is not None:
            self._mic.__exit__()

    def _read(self, source: Iterator[tuple[np.ndarray, float]]) -> None:
        try:
            for block, t in source:
                with self._cond:
                    if self._closed:
                        return
                    self._held.append((self._count, block, t))
                    self._count += 1
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._closed = True
                self._cond.notify_all()

    def blocks(self, since: float | None = None) -> Iterator[tuple[np.ndarray, float]]:
        """(블록, 시각) 을 계속 낸다. since 가 없으면 부른 뒤에 들어온 블록부터, 있으면 그 시각
        이후 블록부터 낸다.

        들고 있는 것보다 이전 시각을 주면 가장 오래된 블록부터 낸다. 읽는 쪽이 history_sec 넘게
        밀리면 그 사이 블록은 건너뛴다. 마이크가 닫히면 남은 블록을 다 내고 끝난다.
        """
        # 어디서부터 읽을지는 첫 블록을 꺼낼 때가 아니라 부른 지금 정한다.
        with self._cond:
            after = (n for n, _, t in self._held if since is not None and t >= since)
            start = next(after, self._count)
        return self._follow(start)

    def _follow(self, nxt: int) -> Iterator[tuple[np.ndarray, float]]:
        while True:
            with self._cond:
                while nxt >= self._count and not self._closed:
                    self._cond.wait()
                if nxt >= self._count:
                    return
                oldest = self._held[0][0]
                nxt = max(nxt, oldest)
                _, block, t = self._held[nxt - oldest]
            yield block, t
            nxt += 1
