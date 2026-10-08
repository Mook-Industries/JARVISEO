"""MicStream 이 마이크 하나를 여럿이 같이 읽게 해 주는지 확인한다. 마이크 없이 돈다."""

import threading
import time

import numpy as np

from jarviseo.voice.mic import BLOCK, SAMPLE_RATE, MicStream

BLOCK_SEC = BLOCK / SAMPLE_RATE


def blocks(n, start=0):
    """번호를 값으로 채운 블록 n 개와 그 시각."""
    return [(np.full(BLOCK, i, np.int16), i * BLOCK_SEC) for i in range(start, start + n)]


def test_여럿이_같은_블록을_같은_순서로_받는다():
    with MicStream(source=blocks(10)) as mic:
        a = [int(b[0]) for b, _ in mic.blocks(since=0.0)]
        b = [int(b[0]) for b, _ in mic.blocks(since=0.0)]

    assert a == b == list(range(10))


def test_since_를_주면_그_시각_이후_블록부터_준다():
    with MicStream(source=blocks(10)) as mic:
        got = [t for _, t in mic.blocks(since=4 * BLOCK_SEC)]

    assert got[0] == 4 * BLOCK_SEC and len(got) == 6


def test_들고_있는_것은_최근_history_sec_뿐이다():
    with MicStream(source=blocks(10), history_sec=3 * BLOCK_SEC) as mic:
        got = [int(b[0]) for b, _ in mic.blocks(since=0.0)]

    assert got == [7, 8, 9]  # 그보다 오래된 블록은 버렸다


def test_since_가_없으면_부른_뒤에_들어온_블록부터_준다():
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
