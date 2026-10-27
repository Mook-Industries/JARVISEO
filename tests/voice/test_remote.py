"""브라우저 마이크 · 스피커(voice/remote.py) — 샘플 번호를 시각으로, 소리는 실시간 속도로 보낸다.

브라우저 · 네트워크는 쓰지 않는다. 받은 시각을 직접 넣고, 보낸 것은 목록에 모은다.
"""

import time

import numpy as np

from jarviseo.voice.gate import MicGate
from jarviseo.voice.mic import BLOCK, SAMPLE_RATE, MicStream
from jarviseo.voice.remote import RemoteMic, RemoteOutput, StreamClock
from jarviseo.voice.speaker import SAMPLE_RATE as SPEAKER_RATE
from jarviseo.voice.speaker import Speaker


def tone(n, value=1000):
    return np.full(n, value, np.int16)


def drain(remote):
    remote.close()
    return list(remote.blocks())


def test_stream_clock_uses_least_delayed_arrival():
    """가장 덜 늦게 받은 소리로 샘플 0 의 시각을 잡는다. 늦게 받은 것은 기준을 늦추지 않는다."""
    clock = StreamClock()
    clock.update(16000, arrived=101.0)  # 1초어치를 101초에 받았다 → 샘플 0 은 100초
    clock.update(24000, arrived=101.8)  # 0.5초어치를 0.3초 늦게 받았다
    assert clock.anchor == 100.0
    assert clock.time(8000) == 100.5


def test_preroll_burst_keeps_its_own_timeline():
    """호출 때 한꺼번에 보낸 직전 3초도 블록마다 제 시각을 받는다(받은 순간으로 뭉치지 않는다)."""
    remote = RemoteMic()
    remote.push(48000, tone(3 * SAMPLE_RATE), arrived=203.0)  # 3~6초 구간을 6초 시점에 받음
    blocks = drain(remote)
    assert len(blocks) == 3 * SAMPLE_RATE // BLOCK
    first_t, last_t = blocks[0][1], blocks[-1][1]
    assert first_t == 200.0  # 샘플 48000 = 203 − 6 + 3
    assert abs(last_t - (203.0 - BLOCK / SAMPLE_RATE)) < 1e-9
    assert remote.time_of(96000) == 203.0


def test_resent_part_is_dropped_and_gap_restarts_blocks():
    """다시 보낸 앞부분은 버리고, 쉬었다가 보낸 소리는 끊긴 끝 조각 없이 새로 블록을 센다."""
    remote = RemoteMic()
    remote.push(0, tone(BLOCK + 100, 1), arrived=10.0)
    remote.push(0, tone(BLOCK + 100, 1), arrived=10.1)  # 같은 구간을 또 받음 → 버림
    remote.push(16000, tone(BLOCK, 2), arrived=11.1)  # 1초 쉬고 다시
    blocks = drain(remote)
    assert [int(b[0]) for b, _ in blocks] == [1, 2]
    assert blocks[1][1] > blocks[0][1]


def test_mic_stream_reads_remote_blocks_since_wake():
    """노트북 마이크 자리(MicStream)에 끼우면 호출 시각 앞부터 다시 읽을 수 있다."""
    remote = RemoteMic()
    mic = MicStream(source=remote.blocks()).__enter__()
    try:
        remote.push(0, tone(SAMPLE_RATE), arrived=50.0)  # 샘플 0 = 49초, 블록 33개(30ms)
        deadline = time.monotonic() + 2
        while len(mic.clip(49.0, 50.0)) < 33 * BLOCK and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(mic.clip(49.5, 50.0)) == 16 * BLOCK  # 49.51 ~ 49.96초에 시작한 블록
    finally:
        remote.close()  # 읽는 스레드가 끝나야 MicStream 이 닫힌다
        mic.close()


def test_speaker_sends_pcm_at_playback_speed_and_mutes_gate():
    """Speaker 가 브라우저로 소리를 보내고, 다 날 때까지 게이트를 닫아 둔다(스피커와 같다)."""
    sent = []
    gate = MicGate(grace_sec=0.3)
    out = RemoteOutput(sent.append, channel=0, lead=0.05, latency=0.0)
    speaker = Speaker(gate=gate, output=out)
    pcm = b"\x01\x00" * (SPEAKER_RATE // 5)  # 0.2초
    started = time.monotonic()
    speaker.play([pcm])
    took = time.monotonic() - started
    assert b"".join(m[1:] for m in sent) == pcm
    assert all(m[0] == 0 for m in sent)
    assert 0.18 < took < 0.6  # 소리 길이만큼 기다렸다
    assert not gate.is_open(time.monotonic())  # 끝나고 0.3초는 닫혀 있다


def test_speaker_stop_flushes_browser_queue():
    """끊으면(끼어들기) 남은 소리를 보내지 않고, 브라우저에 받아 둔 것을 버리라고 알린다."""
    sent = []
    out = RemoteOutput(sent.append, channel=1, lead=0.0, latency=0.0)
    speaker = Speaker(gate=MicGate(), output=out)

    def chunks():
        yield b"\x00\x00" * 2400  # 0.1초
        speaker.stop()
        yield b"\x00\x00" * 2400

    speaker.play(chunks())
    assert speaker.interrupted
    assert sent[-1] == {"type": "flush", "ch": 1}
    assert sum(len(m) - 1 for m in sent if isinstance(m, bytes)) == 4800
