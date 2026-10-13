"""끼어들기(BargeIn)가 호출어를 받으면 읽던 답을 끊고, 그 호출을 다음 턴으로 넘기는지 확인한다.

마이크·스피커·API·호출어 모델은 부르지 않는다.
"""

import sys
import time
import wave
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jarviseo.voice import BargeIn, Wake, stt
from jarviseo.voice.gate import MicGate
from jarviseo.voice.mic import BLOCK, SAMPLE_RATE, MicStream
from jarviseo.voice.speaker import PIECE_BYTES
from jarviseo.voice.tts import CHUNK_BYTES, TextToSpeech
from jarviseo.voice.wakeword import FRAME, WakeWordDetector

DATA = Path(__file__).parent / "data"


class StopCounter:
    """tts 대신 stop() 이 불린 횟수만 센다."""

    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1


def test_호출어가_들리면_읽던_답을_끊고_그_호출을_넘긴다():
    tts = StopCounter()
    barge = BargeIn(tts)

    barge.on_detect(12.5, 0.995)

    assert tts.stops == 1
    assert barge.wait(timeout=0.1) == Wake(12.5, 0.995)
    assert barge.wait(timeout=0.01) is None  # 한 번 넘긴 호출은 다시 안 준다


def test_쌓인_호출이_여럿이면_가장_최근_것만_준다():
    barge = BargeIn(StopCounter())
    barge.on_detect(1.0, 0.98)
    barge.on_detect(4.0, 0.99)

    assert barge.wait(timeout=0.1).t == 4.0
    assert barge.wait(timeout=0.01) is None


def test_clear_는_쌓인_호출을_버린다():
    barge = BargeIn(StopCounter())
    barge.on_detect(1.0, 0.98)

    barge.clear()

    assert barge.wait(timeout=0.01) is None


# -- 감지기 → 끼어들기 → TTS → 게이트 → STT 흐름 ------------------------------------------
# 실제 WakeWordDetector·TextToSpeech·Speaker·MicGate·SpeechToText 를 잇고, 모델 점수·스피커·
# OpenAI API·마이크만 가짜로 바꾼다. 가짜 점수는 프레임에서 가장 큰 샘플 값 ÷ 1000 이다.


class SpeechClient:
    """openai 클라이언트 대신 TTS 는 100ms 조각을 chunks 개 흘리고, STT 는 정해 둔 글자를 준다."""

    def __init__(self, chunks=30):
        self.chunks, self.transcribed = chunks, 0
        speech = SimpleNamespace(with_streaming_response=SimpleNamespace(create=self._speech))
        transcriptions = SimpleNamespace(create=self._transcribe)
        self.audio = SimpleNamespace(speech=speech, transcriptions=transcriptions)

    @contextmanager
    def _speech(self, **kwargs):
        yield SimpleNamespace(iter_bytes=lambda size: (b"\x01" * size for _ in range(self.chunks)))

    def _transcribe(self, **kwargs):
        self.transcribed += 1
        return SimpleNamespace(text="자비서, 저거 뭐야?")


@pytest.fixture
def loop(monkeypatch):
    """게이트 하나를 같이 쓰는 TTS·감지기·끼어들기. 스피커에 쓸 때마다 on_write() 를 부른다."""
    gate = MicGate(grace_sec=0.3)
    monkeypatch.setattr(stt, "MIC_GATE", gate)
    written = []

    class FakeOutputStream:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def write(self, data):
            written.append(len(data))
            ns.on_write(len(written))

        def abort(self):
            written.append("abort")

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(RawOutputStream=FakeOutputStream)
    )
    client = SpeechClient()
    tts = TextToSpeech(client=client)
    tts.speaker.gate = gate
    barge = BargeIn(tts)
    detector = WakeWordDetector(
        None,
        on_detect=barge.on_detect,
        threshold=0.97,
        playback_threshold=0.99,
        gate=gate,
        scorer=lambda frame: float(frame.max()) / 1000,
    )
    ns = SimpleNamespace(
        gate=gate, tts=tts, barge=barge, detector=detector, client=client, written=written
    )
    ns.on_write = lambda n: None
    return ns


def hear(detector, score):
    """지금 마이크에 점수 score 인 소리 80ms 가 들어왔다."""
    detector.feed(np.full(FRAME, round(score * 1000), np.int16), time.monotonic())


def test_재생_중_호출어가_들리면_끊고_이어_말한_질문을_자르지_않고_받는다(loop):
    # 스피커에 세 번째 20ms 를 쓰고 나서 "자비서"가 들렸다
    loop.on_write = lambda n: n == 3 and hear(loop.detector, 0.995)

    loop.tts.speak("긴 대답")

    assert loop.tts.last_interrupted
    assert loop.written == [PIECE_BYTES] * 3 + ["abort"]
    wake = loop.barge.wait(timeout=0.1)
    assert wake.score == pytest.approx(0.995)
    stopped_at = loop.tts.speaker.stopped_at
    assert wake.t <= stopped_at + FRAME / SAMPLE_RATE

    # 호출어에 바로 이어(멈추고 0.05초 뒤) 질문을 시작했다.
    # 0.3초 유예가 남아 있었다면 앞부분이 무음이 된다
    with wave.open(str(DATA / "jarviseo_question.wav"), "rb") as f:
        voice = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    audio = np.concatenate([voice, np.zeros(SAMPLE_RATE, np.int16)])
    t0 = stopped_at + 0.05
    blocks = [
        (audio[i * BLOCK : (i + 1) * BLOCK], t0 + i * BLOCK / SAMPLE_RATE)
        for i in range(len(audio) // BLOCK)
    ]
    assert all(loop.gate.is_open(t) for _, t in blocks)

    with MicStream(history_sec=5.0, source=blocks) as mic:
        utterance = stt.SpeechToText(mic=mic, client=loop.client).listen(timeout=2.0, since=wake.t)

    assert utterance.text == "저거 뭐야?"
    assert loop.client.transcribed == 1


def test_재생_중에는_임계값이_높아_자기_목소리_점수로는_끊지_않는다(loop):
    # 평소 임계값(0.97)은 넘지만 재생 중 임계값(0.99)은 못 넘는다
    loop.on_write = lambda n: n % 10 == 0 and hear(loop.detector, 0.98)

    loop.tts.speak("긴 대답")

    assert not loop.tts.last_interrupted
    assert "abort" not in loop.written
    assert sum(loop.written) == CHUNK_BYTES * loop.client.chunks
    assert loop.barge.wait(timeout=0.01) is None
