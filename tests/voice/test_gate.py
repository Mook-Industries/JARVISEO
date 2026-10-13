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


def test_재생_중과_끝나고_유예_안에_들어온_블록만_무음이_된다():
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


def test_재생이_시작하고_끝날_때_더미_콜백을_부른다():
    gate, calls = MicGate(), []
    gate.on_playback(calls.append)

    with gate.playing():
        assert calls == [True]

    assert calls == [True, False]


def test_재생이_예외로_끝나도_끝을_알리고_게이트가_열린다():
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


def test_스피커는_첫_소리_전에_게이트를_닫고_소리를_다_낸_뒤에_연다(speaker_log):
    gate = MicGate()
    gate.on_playback(speaker_log.append)

    Speaker(gate=gate).play(iter([b"\x01\x00", b"\x02\x00"]))

    assert speaker_log == [True, b"\x01\x00", b"\x02\x00", "stop", False]


def test_조각이_하나도_없으면_게이트를_건드리지_않는다(speaker_log):
    gate = MicGate()
    gate.on_playback(speaker_log.append)

    Speaker(gate=gate).play(iter([]))

    assert speaker_log == []
    assert gate.is_open(time.monotonic())


def test_재생_중_stop_하면_남은_소리를_버리고_게이트를_유예_없이_연다(speaker_log):
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


def test_큰_조각은_20ms_씩_나눠_써서_그_사이에도_멈춘다(monkeypatch):
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


def test_첫_조각을_기다리는_동안_stop_하면_소리를_내지_않는다(speaker_log):
    gate = MicGate()
    gate.on_playback(speaker_log.append)
    speaker = Speaker(gate=gate)

    def chunks():
        speaker.stop()
        yield b"\x01\x00"

    speaker.play(chunks())

    assert speaker_log == []
    assert speaker.interrupted


def test_재생_전에_부른_stop_은_다음_재생에_영향이_없다(speaker_log):
    speaker = Speaker(gate=MicGate())
    speaker.stop()

    speaker.play(iter([b"\x01\x00"]))

    assert speaker_log == [b"\x01\x00", "stop"]
    assert not speaker.interrupted


def test_reopen_은_닫힌_구간을_줄이기만_한다():
    gate = MicGate(grace_sec=0.3)
    with gate.playing():
        start = time.monotonic()
    end = time.monotonic()

    gate.reopen(end + 1.0)  # 늘리지 않는다
    assert gate.is_open(end + 0.5)

    gate.reopen(start)  # 지난 시각까지 거슬러 연다
    assert gate.is_open(start)


def test_STT_와_TTS_는_같은_게이트를_쓴다():
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


def test_listen_은_재생_중에_들어온_말은_흘리고_유예가_지난_말은_받아쓴다(monkeypatch, mic_from):
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
