"""MicStream 이 마이크 하나를 여럿이 같이 읽게 해 주는지 확인한다. 마이크 없이 돈다."""

import threading
import time

import numpy as np

from jarviseo.voice.mic import BLOCK, SAMPLE_RATE, MicStream

BLOCK_SEC = BLOCK / SAMPLE_RATE


def blocks(n, start=0):
    """번호를 값으로 채운 블록 n 개와 그 시각."""
    return [(np.full(BLOCK, i, np.int16), i * BLOCK_SEC) for i in range(start, start + n)]


def test_readers_get_same_blocks_in_order():
    """여럿이 같은 블록을 같은 순서로 받는다."""
    with MicStream(source=blocks(10)) as mic:
        a = [int(b[0]) for b, _ in mic.blocks(since=0.0)]
        b = [int(b[0]) for b, _ in mic.blocks(since=0.0)]

    assert a == b == list(range(10))


def test_since_returns_blocks_after_that_time():
    """`since` 를 주면 그 시각 이후 블록부터 준다."""
    with MicStream(source=blocks(10)) as mic:
        got = [t for _, t in mic.blocks(since=4 * BLOCK_SEC)]

    assert got[0] == 4 * BLOCK_SEC and len(got) == 6


def test_keeps_only_recent_history_sec():
    """들고 있는 것은 최근 `history_sec` 뿐이다."""
    with MicStream(source=blocks(10), history_sec=3 * BLOCK_SEC) as mic:
        got = [int(b[0]) for b, _ in mic.blocks(since=0.0)]

    assert got == [7, 8, 9]  # 그보다 오래된 블록은 버렸다


def test_without_since_returns_blocks_after_call():
    """`since` 가 없으면 부른 뒤에 들어온 블록부터 준다."""
    go = threading.Event()

    def source():
        yield from blocks(3)
        go.wait(5)
        yield from blocks(3, start=3)

    with MicStream(source=source()) as mic:
        deadline = time.monotonic() + 5
        while mic._count < 3 and time.monotonic() < deadline:  # 앞의 3개가 다 들어올 때까지
            time.sleep(0.01)
        live = mic.blocks()
        go.set()
        got = [int(b[0]) for b, _ in live]

    assert got == [3, 4, 5]
