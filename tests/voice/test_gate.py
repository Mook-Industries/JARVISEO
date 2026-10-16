"""마이크 게이트가 재생 중 소리를 STT 에 넘기지 않고, 호출어 훅을 제때 부르는지 확인한다.

마이크·스피커·API 는 부르지 않는다. 훅은 받은 값을 모으기만 하는 더미 콜백이다.
"""

import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jarviseo.voice import stt
from jarviseo.voice.gate import MIC_GATE, MicGate
from jarviseo.voice.mic import BLOCK, SAMPLE_RATE
from jarviseo.voice.speaker import PIECE_BYTES, Speaker
from jarviseo.voice.tts import TextToSpeech

BLOCK_SEC = BLOCK / SAMPLE_RATE
DATA = Path(__file__).parents[1] / "data"


def test_only_blocks_during_playback_and_grace_are_muted():
    """재생 중과 끝나고 유예 안에 들어온 블록만 무음이 된다."""
    gate = MicGate(grace_sec=0.3)
    before = time.monotonic() - 1.0
    with gate.playing():
        during = time.monotonic()
    after = time.monotonic()
    block = np.ones(BLOCK, np.int16)
    times = [before, during, after + 0.2, after + 0.4]

    out = list(gate.mute((block, t) for t in times))

    assert [t for _, t in out] == times  # 시각은 그대로 넘긴다
    assert [b.any() for b, _ in out] == [True, False, False, True]


def test_calls_callbacks_on_playback_start_and_end():
    """재생이 시작하고 끝날 때 더미 콜백을 부른다."""
    gate, calls = MicGate(), []
    gate.on_playback(calls.append)

    with gate.playing():
        assert calls == [True]

    assert calls == [True, False]


def test_playback_error_still_opens_gate():
    """재생이 예외로 끝나도 끝을 알리고 게이트가 열린다."""
    gate, calls = MicGate(), []
    gate.on_playback(calls.append)

    with pytest.raises(RuntimeError), gate.playing():
        raise RuntimeError("재생 실패")

    assert calls == [True, False]
    assert gate.is_open(time.monotonic() + 0.4)


@pytest.fixture
def speaker_log(monkeypatch):
    """sounddevice 대신 스피커에 일어난 일을 순서대로 적는다."""
    log = []

    class FakeOutputStream:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            log.append("stop")  # 실제로는 여기서 버퍼에 남은 소리를 다 낸다

        def write(self, data):
            log.append(data)

        def abort(self):
            log.append("abort")  # 버퍼에 남은 소리를 버린다

    sounddevice = SimpleNamespace(RawOutputStream=FakeOutputStream)
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    return log


def test_speaker_closes_gate_before_sound_and_opens_after(speaker_log):
    """스피커는 첫 소리 전에 게이트를 닫고 소리를 다 낸 뒤에 연다."""
    gate = MicGate()
    gate.on_playback(speaker_log.append)

    Speaker(gate=gate).play(iter([b"\x01\x00", b"\x02\x00"]))

    assert speaker_log == [True, b"\x01\x00", b"\x02\x00", "stop", False]


def test_no_chunks_leaves_gate(speaker_log):
    """조각이 하나도 없으면 게이트를 건드리지 않는다."""
    gate = MicGate()
    gate.on_playback(speaker_log.append)

    Speaker(gate=gate).play(iter([]))

    assert speaker_log == []
    assert gate.is_open(time.monotonic())


def test_stop_during_playback_drops_rest_and_opens_gate(speaker_log):
    """재생 중 `stop()` 하면 남은 소리를 버리고 게이트를 유예 없이 연다."""
    gate = MicGate(grace_sec=0.3)
    gate.on_playback(speaker_log.append)
    speaker = Speaker(gate=gate)

    def chunks():
        yield b"\x01\x00"
        speaker.stop()  # 다른 스레드에서 호출어를 잡았다
        yield b"\x02\x00"
        raise AssertionError("끊긴 뒤에는 더 읽지 않는다")

    speaker.play(chunks())

    assert speaker_log == [True, b"\x01\x00", "abort", "stop", False]
    assert speaker.interrupted
    assert gate.is_open(speaker.stopped_at)  # 끝까지 재생했다면 0.3초 더 막혀 있다


def test_large_chunk_written_in_20ms_slices_can_stop(monkeypatch):
    """큰 조각은 20ms 씩 나눠 써서 그 사이에도 멈춘다."""
    speaker, written = Speaker(gate=MicGate()), []

    class FakeOutputStream:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def write(self, data):
            written.append(data)
            speaker.stop()  # 첫 20ms 를 쓰자마자 호출어가 들렸다

        def abort(self):
            written.append("abort")

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(RawOutputStream=FakeOutputStream)
    )
    chunk = b"\x01" * (PIECE_BYTES * 3)

    speaker.play(iter([chunk]))

    assert written == [chunk[:PIECE_BYTES], "abort"]


def test_stop_before_first_chunk_plays_nothing(speaker_log):
    """첫 조각을 기다리는 동안 `stop()` 하면 소리를 내지 않는다."""
    gate = MicGate()
    gate.on_playback(speaker_log.append)
    speaker = Speaker(gate=gate)

    def chunks():
        speaker.stop()
        yield b"\x01\x00"

    speaker.play(chunks())

    assert speaker_log == []
    assert speaker.interrupted


def test_stop_before_playback_does_not_affect_next(speaker_log):
    """재생 전에 부른 `stop()` 은 다음 재생에 영향이 없다."""
    speaker = Speaker(gate=MicGate())
    speaker.stop()

    speaker.play(iter([b"\x01\x00"]))

    assert speaker_log == [b"\x01\x00", "stop"]
    assert not speaker.interrupted


def test_reopen_only_shrinks_closed_span():
    """`reopen()` 은 닫힌 구간을 줄이기만 한다."""
    gate = MicGate(grace_sec=0.3)
    with gate.playing():
        start = time.monotonic()
    end = time.monotonic()

    gate.reopen(end + 1.0)  # 늘리지 않는다
    assert gate.is_open(end + 0.5)

    gate.reopen(start)  # 지난 시각까지 거슬러 연다
    assert gate.is_open(start)


def test_stt_and_tts_share_gate():
    """STT 와 TTS 는 같은 게이트를 쓴다."""
    assert TextToSpeech(client=object()).speaker.gate is stt.MIC_GATE is MIC_GATE


@pytest.fixture
def mic_from(monkeypatch):
    """마이크 대신 "자비서, 저거 뭐야?" 녹음을 흘린다. 첫 블록의 시각은 넣어 둔 값을 쓴다."""
    with wave.open(str(DATA / "jarviseo_question.wav"), "rb") as f:
        voice = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    silence = np.zeros(SAMPLE_RATE, np.int16)
    audio = np.concatenate([silence, voice, silence])
    starts = []

    class Mic:
        def __init__(self, device):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def blocks(self):
            t0 = starts.pop(0)
            for n in range(len(audio) // BLOCK):
                yield audio[n * BLOCK : (n + 1) * BLOCK], t0 + n * BLOCK_SEC

    monkeypatch.setattr(stt, "Microphone", Mic)
    return starts


class FakeClient:
    """openai 클라이언트 대신 받아쓰기 요청을 세고, 정해 둔 글자를 돌려준다."""

    def __init__(self):
        self.requests = []
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(text="자비서, 저거 뭐야?")


def test_listen_ignores_speech_during_playback_and_grace(monkeypatch, mic_from):
    """`listen()` 은 재생 중에 들어온 말은 흘리고 유예가 지난 말은 받아쓴다."""
    gate, client = MicGate(), FakeClient()
    monkeypatch.setattr(stt, "MIC_GATE", gate)
    listener = stt.SpeechToText(client=client)

    with gate.playing():
        mic_from.append(time.monotonic())
        assert listener.listen(timeout=2.0) is None  # 스피커에서 나온 소리로 친다
    assert client.requests == []

    mic_from.append(time.monotonic() + gate.grace_sec)
    assert listener.listen(timeout=2.0).text == "저거 뭐야?"
    assert len(client.requests) == 1
