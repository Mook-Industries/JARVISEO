"""마이크 입력.  담당: 문태현

16kHz 모노 int16 을 30ms 블록으로 읽고, 블록마다 첫 샘플이 마이크에 들어온 시각
(``time.monotonic()`` 기준)을 같이 낸다.

시각은 읽은 샘플 수로 센다. 블록마다 ``time.monotonic()`` 을 찍으면 윈도우(파이썬 3.12)에서는
15.6ms 단위로 뭉치고, 드라이버가 블록을 몰아서 줄 때마다 시각이 흔들린다.
다만 장치는 16kHz 를 딱 맞춰 주지 않는다. 샘플 수로만 세면 실제 시각보다 점점 늦어져서
(10/15 실측 1.4%, 8분에 약 7초) 재생 구간·호출 시각·사진 시각과 비교가 틀어졌다(#69).
그래서 ``BlockClock`` 이 블록을 받은 시각과 견줘 늦어진 만큼 당긴다.

``Microphone`` 은 한 곳에서만 읽는다. 호출어 감지와 STT 처럼 여럿이 같이 읽을 때는
``MicStream`` 을 쓴다.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator

import numpy as np

__all__ = ["BLOCK", "SAMPLE_RATE", "BlockClock", "MicStream", "Microphone"]

SAMPLE_RATE = 16000
BLOCK = 480  # 30ms. webrtcvad 가 받는 길이는 10·20·30ms 뿐이다
BLOCK_SEC = BLOCK / SAMPLE_RATE
# 샘플 수로 센 시각이 이보다 더 늦으면 당긴다. 이보다 작은 차이는 읽기 흔들림과 못 가른다.
DRIFT_TOLERANCE_SEC = 0.01
# 늦었는지는 최근 이만큼 받은 블록들 중 가장 덜 늦은 것으로 본다. 읽는 쪽이 잠깐 밀려서
# 늦게 받은 블록만 보고 당기면 안 되기 때문이다.
DRIFT_WINDOW_SEC = 1.0
# MicStream 이 들고 있는 최근 소리(초). 호출어 바로 뒤 질문은 호출 1초 앞부터 다시 읽는다.
# 되묻기 대답을 기다리다 다시 부른 질문(#105)은 그 대답을 받아쓴 뒤(질문 끝 + 왕복 최대 2초쯤)에야
# 다시 읽어서, 호출보다 3~4초 늦게 읽어도 남아 있어야 한다. 6초는 192KB 다.
HISTORY_SEC = 6.0


class BlockClock:
    """블록 시각을 샘플 수로 세되, 실제 시각보다 늦어지거나 앞서면 맞춘다.

    블록을 받은 시각(read_at)에서 블록 길이와 장치 입력 지연을 빼면 "늦어도 이때는 들어왔다"는
    상한이 나온다. 소리는 들어오기 전에 받을 수 없으니 실제 시각은 이 상한보다 늦을 수 없다.

    - 샘플 수로 센 시각이 상한보다 앞서면(장치가 빠르거나 첫 블록을 늦게 받음) 바로 상한에 맞춘다.
    - 최근 ``DRIFT_WINDOW_SEC`` 동안 받은 블록이 모두 상한보다 ``DRIFT_TOLERANCE_SEC`` 넘게 늦으면
      (장치가 느림) 그중 가장 덜 늦은 만큼 당긴다. 가장 덜 늦은 블록은 제때 받은 블록이다.
    - 읽는 쪽이 밀렸다가 쌓인 블록을 몰아 받는 동안에는 블록이 실제보다 빨리 들어온다.
      그때는 당기지 않는다(쌓인 블록은 원래 늦게 받은 것이다). 밀린 동안 생긴 차이는
      다 받은 뒤 맞춘다.
    - overflow(놓친 샘플)가 나면 센 수가 틀렸으니 상한에 다시 맞춘다.

    낸 시각은 늘 앞 블록보다 크다.
    """

    def __init__(
        self,
        latency: float,
        tolerance: float = DRIFT_TOLERANCE_SEC,
        window: float = DRIFT_WINDOW_SEC,
    ) -> None:
        """latency 는 장치가 알려 준 입력 지연(초)이다(``InputStream.latency``)."""
        self.latency = latency
        self.tolerance = tolerance
        self.window = window
        self._t0: float | None = None  # 블록 0 의 시각. 맞출 때마다 옮긴다
        self._n = 0
        self._last = -math.inf
        self._lags: deque[tuple[float, float]] = deque()  # (받은 시각, 상한 − 센 시각)

    def stamp(self, read_at: float, overflowed: bool = False) -> float:
        """방금 받은 블록의 첫 샘플 시각. read_at 은 그 블록을 받은 monotonic 시각이다."""
        latest = read_at - BLOCK_SEC - self.latency
        if self._t0 is None or overflowed:
            self._t0 = latest - self._n * BLOCK_SEC
            self._lags.clear()
        counted = self._t0 + self._n * BLOCK_SEC
        if counted > latest:
            self._t0 -= counted - latest
            counted = latest
            self._lags.clear()
        self._lags.append((read_at, latest - counted))
        while read_at - self._lags[0][0] > self.window:
            self._lags.popleft()
        span = read_at - self._lags[0][0]
        # 창이 반 넘게 찼고, 블록을 실제보다 빨리 받고 있지 않을 때만 본다(몰아 받는 중이 아님)
        if span >= self.window / 2 and len(self._lags) * BLOCK_SEC <= 1.5 * span + BLOCK_SEC:
            lag = min(lag for _, lag in self._lags)
            if lag > self.tolerance:
                self._t0 += lag
                counted += lag
                self._lags.clear()
        self._n += 1
        self._last = max(counted, self._last + 1e-6)
        return self._last


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
        clock = BlockClock(self._stream.latency)
        while True:
            block, overflowed = self._stream.read(BLOCK)
            yield block[:, 0], clock.stamp(time.monotonic(), overflowed)


class MicStream:
    """마이크 하나를 열어 두고 여러 곳에서 같이 읽는다.

    ``with MicStream() as mic:`` 안에서 ``mic.blocks()`` 를 부를 때마다 따로 읽는 줄이 하나 생긴다.
    호출어 감지는 계속 읽고, STT 는 호출어를 잡은 시각부터 읽는다. 호출어 바로 뒤에 이어 말한
    질문이 잘리지 않도록 최근 history_sec 초는 들고 있다.
    """

    def __init__(
        self,
        device: int | None = None,
        history_sec: float = HISTORY_SEC,
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

    def clip(self, start: float, end: float) -> np.ndarray:
        """들고 있는 소리 중 start ~ end 시각(monotonic)에 들어온 블록을 이어 붙여 돌려준다.

        지나간 구간만 자른다. 아직 안 들어온 뒤쪽은 기다리지 않고 빈 채로 둔다.
        호출어 2단계 확인(``wakecheck.py``)이 감지 직전 구간을 받아쓸 때 쓴다.
        """
        with self._cond:
            parts = [block for _, block, t in self._held if start <= t < end]
        return np.concatenate(parts) if parts else np.empty(0, np.int16)

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
