"""브라우저 마이크 · 스피커.  담당: 문태현

서버에 마이크 · 스피커가 없을 때(클라우드) 대시보드 브라우저가 그 자리를 맡는다(#218).
브라우저가 보낸 마이크 소리를 ``MicStream`` 에 흘리고(``RemoteMic``), ``Speaker`` 가 쓰는 소리를
브라우저로 보낸다(``RemoteOutput``). 받아쓰기 · 읽기 · 게이트 · 끼어들기는 노트북과 같은 코드다.

시각
----
브라우저는 소리마다 "마이크를 연 뒤 몇 번째 샘플부터인지"(pos)를 붙여 보낸다. 서버는 받은 시각에서
그 길이를 빼서 샘플 0 의 시각(기준점)을 잡고(``StreamClock``), 블록 시각 = 기준점 + pos / 16000 으로
센다. 호출어 감지 · 카메라 장면도 pos 로 오니 같은 시계로 바뀐다. 호출되면 브라우저가 직전 몇 초를
한꺼번에 보내는데(1.5초 앞 읽기 · 2단계 확인 구간), 이 묶음도 pos 로 제자리에 놓인다.

소리 내기
---------
``RemoteOutput`` 은 sounddevice 출력 대신 쓴다. 쓴 소리는 바로 보내고, 브라우저가 실제로 내는 속도에
맞춰 write() 가 기다린다(``LEAD_SEC`` 만큼만 앞서 보낸다). 그래야 재생 중 게이트 · 끼어들기 · 필러
타이밍이 스피커를 쓸 때와 같다. 끊으면(abort) 브라우저에 남은 소리를 버리라고 알린다.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator

import numpy as np

from jarviseo.voice import speaker
from jarviseo.voice.mic import BLOCK, DRIFT_WINDOW_SEC, SAMPLE_RATE

__all__ = ["LEAD_SEC", "RemoteMic", "RemoteOutput", "StreamClock"]

# 브라우저에 이만큼(초)만 앞서 보낸다. 끊었을 때 이미 보낸 소리는 브라우저가 버린다.
LEAD_SEC = 0.3
# 보낸 소리가 브라우저 스피커에서 나기까지 걸린다고 보는 시간(초). 게이트를 이만큼 늦게 연다.
LATENCY_SEC = 0.15


class StreamClock:
    """샘플 번호(pos)를 서버 시각(monotonic)으로 바꾼다.

    소리는 들어오기 전에 받을 수 없으니 "받은 시각 − 끝 pos / 16000" 은 샘플 0 의 시각보다 늦다.
    최근 ``window`` 초 동안 받은 것 중 가장 이른 값을 기준점으로 쓴다(가장 덜 늦게 받은 소리).
    오래 켜 두면 브라우저 · 서버 시계가 조금씩 어긋나서, 창 밖의 값은 버린다.
    """

    def __init__(self, window: float = DRIFT_WINDOW_SEC) -> None:
        self.window = window
        self.anchor: float | None = None
        self._seen: deque[tuple[float, float]] = deque()  # (받은 시각, 그때 본 기준점)

    def update(self, end_pos: int, arrived: float) -> None:
        """end_pos 샘플까지 들어 있는 소리를 arrived 시각에 받았다."""
        self._seen.append((arrived, arrived - end_pos / SAMPLE_RATE))
        while arrived - self._seen[0][0] > self.window:
            self._seen.popleft()
        self.anchor = min(a for _, a in self._seen)

    def time(self, pos: float) -> float:
        """pos 번째 샘플이 마이크에 들어온 시각. 아직 소리를 하나도 못 받았으면 지금."""
        if self.anchor is None:
            return time.monotonic()
        return self.anchor + pos / SAMPLE_RATE


class RemoteMic:
    """브라우저 마이크. ``MicStream(source=remote.blocks())`` 로 노트북 마이크 자리에 끼운다.

    push() 는 WebSocket 을 받는 쪽에서, blocks() 는 MicStream 의 읽는 스레드에서 부른다.
    """

    def __init__(self) -> None:
        self.clock = StreamClock()
        self._queue: queue.SimpleQueue[tuple[np.ndarray, float] | None] = queue.SimpleQueue()
        self._next = 0  # 다음에 받을 샘플 번호. 이미 받은 구간은 버린다
        self._rest = np.empty(0, np.int16)  # 블록(30ms)이 안 되는 끝부분
        self._rest_pos = 0
        self._last_t = -np.inf
        self._lock = threading.Lock()

    def push(self, pos: int, pcm: np.ndarray, arrived: float | None = None) -> None:
        """pos 번째 샘플부터 이어지는 소리(16kHz 모노 int16)를 받았다."""
        arrived = time.monotonic() if arrived is None else arrived
        with self._lock:
            end = pos + len(pcm)
            if end <= self._next:
                return  # 이미 받은 소리(호출 때 다시 보낸 앞부분)
            if pos < self._next:
                pcm, pos = pcm[self._next - pos :], self._next
            self.clock.update(end, arrived)
            if pos != self._rest_pos + len(self._rest):
                # 쉬었다가 다시 보낸다(턴 사이). 끊긴 앞 조각은 버리고 새로 센다
                self._rest, self._rest_pos = np.empty(0, np.int16), pos
            joined = np.concatenate([self._rest, pcm])
            whole = len(joined) // BLOCK * BLOCK
            for i in range(0, whole, BLOCK):
                t = max(self.clock.time(self._rest_pos + i), self._last_t + 1e-6)
                self._last_t = t
                self._queue.put((joined[i : i + BLOCK], t))
            self._rest, self._rest_pos = joined[whole:], self._rest_pos + whole
            self._next = end

    def time_of(self, pos: float) -> float:
        """브라우저가 붙인 샘플 번호(호출어 감지 · 카메라 장면)를 서버 시각으로."""
        return self.clock.time(pos)

    def blocks(self) -> Iterator[tuple[np.ndarray, float]]:
        """(30ms 블록, 시각)을 받는 대로 낸다. close() 하면 끝난다."""
        while (item := self._queue.get()) is not None:
            yield item

    def close(self) -> None:
        self._queue.put(None)


class RemoteOutput:
    """``Speaker(output=RemoteOutput(send, channel))``. 쓴 PCM(24kHz)을 브라우저로 보낸다.

    send(data) 는 bytes(소리: 채널 1바이트 + PCM)나 dict(알림)를 받아 브라우저로 보낸다.
    채널은 브라우저가 소리를 따로 줄 세우는 번호다(대답과 시작음 · 종료음이 서로 끊지 않게).
    """

    def __init__(
        self,
        send: Callable[[bytes | dict], None],
        channel: int,
        lead: float = LEAD_SEC,
        latency: float = LATENCY_SEC,
    ) -> None:
        self.send, self.channel = send, channel
        self.lead, self.latency = lead, latency

    def __call__(self) -> _Playback:
        return _Playback(self)


class _Playback:
    """재생 한 번. sounddevice RawOutputStream 과 같은 모양(write · abort · latency, with 문)."""

    def __init__(self, out: RemoteOutput) -> None:
        self.out = out
        self.latency = out.latency
        self._until = 0.0  # 보낸 소리가 브라우저에서 다 나는 시각(monotonic)
        self._aborted = False

    def __enter__(self) -> _Playback:
        self._until = time.monotonic() + self.latency
        return self

    def write(self, pcm: bytes) -> None:
        self.out.send(bytes([self.out.channel]) + pcm)
        now = time.monotonic()
        self._until = max(self._until, now) + len(pcm) / 2 / speaker.SAMPLE_RATE
        ahead = self._until - now - self.latency - self.out.lead
        if ahead > 0:
            time.sleep(ahead)

    def abort(self) -> None:
        """남은 소리를 버린다. 브라우저도 받아 둔 것을 버린다."""
        self._aborted = True
        self.out.send({"type": "flush", "ch": self.out.channel})

    def __exit__(self, *exc: object) -> None:
        # 스피커처럼 마지막 소리가 다 날 때까지 기다린다(그 뒤에 게이트가 열린다)
        if not self._aborted:
            time.sleep(max(0.0, self._until - time.monotonic()))
