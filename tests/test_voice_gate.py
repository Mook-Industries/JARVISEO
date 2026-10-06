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
from jarviseo.voice.speaker import Speaker
from jarviseo.voice.tts import TextToSpeech

BLOCK_SEC = BLOCK / SAMPLE_RATE
DATA = Path(__file__).parent / "data"


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
