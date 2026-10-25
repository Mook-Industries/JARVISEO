"""MicStream 이 마이크 하나를 여럿이 같이 읽게 해 주는지, 블록 시각이 실제 시각을 따라가는지
확인한다. 마이크 없이 돈다.
"""

import itertools
import random
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from jarviseo.voice import mic as mic_module
from jarviseo.voice.mic import BLOCK, SAMPLE_RATE, BlockClock, Microphone, MicStream

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


def test_default_history_keeps_call_read_after_reply():
    """기본값이면 다시 부르고 4초 뒤에 읽어도 호출 1초 앞부터 남아 있다."""
    # 되묻기 대답을 기다리다 다시 불렀으면(#105) 다음 턴은 그 대답을 받아쓴 뒤에야
    # 호출 시각 1초 앞부터(config.WAKE_LOOKBACK_SEC) 다시 읽는다
    wake = 2.0
    with MicStream(source=blocks(round((wake + 4.0) / BLOCK_SEC))) as mic:
        got = [t for _, t in mic.blocks(since=wake - 1.0)]

    assert got[0] == pytest.approx(wake - 1.0, abs=BLOCK_SEC)


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


# --- 블록 시각 (#69) ---------------------------------------------------------------

LATENCY = 0.05  # 장치가 알려 주는 입력 지연(초)


def simulate(rate=1.0, seconds=600.0, late=0.0, burst=1, stall=None, seed=0):
    """가짜 장치로 BlockClock 을 돌려 (실제 시작 시각, 낸 시각) 목록을 준다.

    rate 는 장치가 실제로 주는 속도 / 16kHz (0.985 면 1.5% 느림). 읽는 쪽은 블록마다 0~5ms 늦게
    받고, late 초를 주면 블록 20개 중 하나꼴로 0~late 초 늦게 받는다. burst 는 드라이버가 블록을
    몇 개씩 몰아 주는지. stall=(언제, 몇 초) 면 그 시각에 읽는 쪽이 그만큼 멈췄다가 몰아 읽는다.
    """
    rng = random.Random(seed)
    clock = BlockClock(LATENCY)
    real_block = BLOCK_SEC / rate
    reader_free = 0.0
    out = []
    for n in range(int(seconds / real_block)):
        start = 100.0 + n * real_block
        delivered = 100.0 + (n // burst + 1) * burst * real_block + LATENCY
        delay = rng.uniform(0, late) if rng.random() < 0.05 else rng.uniform(0, 0.005)
        read_at = max(delivered, reader_free) + delay
        reader_free = read_at
        if stall is not None and start - 100.0 >= stall[0]:
            reader_free += stall[1]
            stall = None
        out.append((start, clock.stamp(read_at)))
    return out


def errors(stamps, skip_sec=1.0):
    """처음 skip_sec 를 뺀 (낸 시각 − 실제 시각) 목록."""
    return [t - start for start, t in stamps if start - 100.0 >= skip_sec]


@pytest.mark.parametrize("rate", [0.985, 0.995, 1.0, 1.01])
def test_clock_follows_device_rate(rate):
    """장치가 16kHz 보다 느리거나 빨라도 10분 동안 블록 시각 오차가 50ms 를 넘지 않는다."""
    errs = errors(simulate(rate=rate))

    assert max(abs(e) for e in errs) < 0.05


def test_counting_samples_only_drifts_on_slow_device():
    """고치기 전처럼 샘플 수로만 세면 1.5% 느린 장치에서 10분에 8초 넘게 늦어진다."""
    stamps = simulate(rate=0.985)
    first_start, first_t = stamps[0]
    counted = [first_t + i * BLOCK_SEC - start for i, (start, _) in enumerate(stamps)]

    assert counted[-1] < -8.0
    assert max(abs(e) for e in errors(stamps)) < 0.05  # BlockClock 은 따라간다


def test_clock_handles_late_reads():
    """읽는 쪽이 가끔 200ms 까지 늦게 받아도 늦은 블록에 휘둘리지 않는다."""
    errs = errors(simulate(rate=0.985, late=0.2))

    assert max(abs(e) for e in errs) < 0.05


def test_clock_handles_bursty_driver():
    """드라이버가 블록을 4개씩 몰아 줘도 늦은 블록에 휘둘리지 않는다."""
    errs = errors(simulate(rate=0.985, burst=4))

    assert max(abs(e) for e in errs) < 0.05


@pytest.mark.parametrize("stall_sec", [0.2, 0.5])
def test_clock_ignores_short_reader_stall(stall_sec):
    """읽는 쪽이 잠깐 멈췄다가 쌓인 블록을 몰아 읽어도 시각을 잘못 당기지 않는다."""
    errs = errors(simulate(rate=0.985, seconds=60.0, stall=(30.0, stall_sec)))

    assert max(abs(e) for e in errs) < 0.05


@pytest.mark.parametrize("stall_sec", [3.0, 10.0])
def test_clock_recovers_after_long_reader_stall(stall_sec):
    """오래 멈추면 밀린 만큼은 틀리지만, 몰아 읽는 동안 엉뚱하게 당기지 않고 다 읽은 뒤 맞춘다."""
    stamps = simulate(rate=0.985, seconds=60.0, stall=(30.0, stall_sec))
    drift = 0.015 * stall_sec  # 멈춘 동안은 맞출 수 없다

    during = [t - s for s, t in stamps if 30.0 <= s - 100.0 < 30.0 + stall_sec]
    after = [t - s for s, t in stamps if s - 100.0 >= 30.0 + stall_sec + 2.0]
    assert max(abs(e) for e in during) < drift + 0.05
    assert max(abs(e) for e in after) < 0.05


def test_clock_never_goes_backward():
    """맞추느라 시각을 옮겨도 블록 시각은 늘 앞 블록보다 크다."""
    for rate in (0.985, 1.01):
        times = [t for _, t in simulate(rate=rate, seconds=120.0, burst=4)]
        assert all(b > a for a, b in zip(times, times[1:], strict=False))


def test_overflow_reanchors_to_read_time():
    """놓친 샘플(overflow)이 있으면 센 수를 버리고 받은 시각에 다시 맞춘다."""
    clock = BlockClock(LATENCY)
    clock.stamp(10.0)
    clock.stamp(10.03)

    t = clock.stamp(15.0, overflowed=True)  # 그사이 5초를 놓쳤다

    assert t == pytest.approx(15.0 - BLOCK_SEC - LATENCY)


class FakeStream:
    """sounddevice.InputStream 대신. read() 마다 가짜 시계를 장치 속도만큼 흘린다."""

    latency = LATENCY

    def __init__(self, clock, rate):
        self.clock, self.rate = clock, rate

    def read(self, frames):
        self.clock.now += frames / SAMPLE_RATE / self.rate
        return np.zeros((frames, 1), np.int16), False


def test_microphone_blocks_follow_slow_device(monkeypatch):
    """`Microphone.blocks()` 가 1.5% 느린 장치에서도 5분 뒤 블록 시각을 50ms 안으로 낸다."""
    fake = SimpleNamespace(now=100.0)
    monkeypatch.setattr(mic_module, "time", SimpleNamespace(monotonic=lambda: fake.now))
    mic = Microphone()
    mic._stream = FakeStream(fake, rate=0.985)

    *_, (_, t) = itertools.islice(mic.blocks(), 10_000)  # 블록 1만 개 = 장치로 약 5분

    # 마지막 블록은 방금(fake.now) 다 들어왔다. 첫 샘플은 그보다 블록 길이 + 지연만큼 앞이다.
    started = fake.now - BLOCK_SEC / 0.985 - LATENCY
    assert abs(t - started) < 0.05


def test_clip_returns_held_blocks_between_times():
    """`clip(start, end)` 는 들고 있는 블록 중 그 시각 사이에 들어온 것만 이어 붙인다."""
    with MicStream(source=blocks(10)) as mic:
        list(mic.blocks(since=0.0))  # 다 들어올 때까지 기다린다
        got = mic.clip(3 * BLOCK_SEC, 6 * BLOCK_SEC)
        none = mic.clip(20 * BLOCK_SEC, 30 * BLOCK_SEC)

    assert [int(v) for v in got[::BLOCK]] == [3, 4, 5]
    assert len(none) == 0 and none.dtype == np.int16
