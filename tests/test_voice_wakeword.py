"""호출어 감지기가 언제 깨어나는지 가짜 점수로 확인한다. 모델·마이크·API 는 부르지 않는다.

가짜 점수는 프레임 안에서 가장 큰 샘플 값 ÷ 1000 이다. 값이 900 인 블록이 든 프레임은 0.9 점이다.
"""

import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jarviseo.voice import stt
from jarviseo.voice.gate import MicGate
from jarviseo.voice.mic import BLOCK, SAMPLE_RATE, MicStream
from jarviseo.voice.wakeword import FRAME, WakeWordDetector

BLOCK_SEC = BLOCK / SAMPLE_RATE
DATA = Path(__file__).parent / "data"


def fake_score(frame):
    return float(frame.max()) / 1000


def detector(hits, gate=None, on_detect=None, mic=None):
    return WakeWordDetector(
        mic,
        on_detect=on_detect or (lambda t, s: hits.append((t, s))),
        threshold=0.5,
        playback_threshold=0.8,
        gate=gate or MicGate(),
        scorer=fake_score,
    )


def feed(det, values, t0=0.0):
    """값이 values 인 블록을 t0 부터 30ms 간격으로 넣는다."""
    for i, v in enumerate(values):
        det.feed(np.full(BLOCK, v, np.int16), t0 + i * BLOCK_SEC)


def test_30ms_블록을_80ms_로_모아_점수를_낸다():
    frames = []
    det = WakeWordDetector(None, gate=MicGate(), scorer=lambda f: frames.append(len(f)) or 0.0)

    feed(det, [0] * 8)  # 8 × 480 = 3840 샘플

    assert frames == [FRAME] * 3


def test_임계값을_넘으면_그_프레임이_끝난_시각으로_한_번_부른다():
    hits = []
    det = detector(hits)

    feed(det, [0] * 10 + [900] * 3 + [0] * 10)  # 0.30~0.39초에 호출어

    # 900 이 처음 들어간 프레임은 0.24~0.32초 프레임이다
    assert hits == [(pytest.approx(0.32), 0.9)]


def test_2초_안의_감지는_한_번으로_치고_2초가_지나면_다시_깨어난다():
    hits = []
    det = detector(hits)

    feed(det, [900] * 40)  # 1.2초 내내 높은 점수
    feed(det, [0] * 40, t0=1.2)
    feed(det, [900] * 8, t0=2.4)  # 첫 감지 뒤 2초가 지났다

    assert [round(t, 2) for t, _ in hits] == [0.08, 2.48]


def test_재생_중과_끝난_뒤_유예_안에서는_임계값이_0_8_이다():
    gate, hits = MicGate(grace_sec=0.3), []
    det = detector(hits, gate=gate)

    with gate.playing():
        now = time.monotonic()
        feed(det, [600] * 8, t0=now)  # 0.6: 평소라면 깨지만 재생 중이라 안 깬다
        feed(det, [900] * 8, t0=now + 0.24)  # 0.9: 재생 중이어도 깬다(끼어들기)
    end = time.monotonic()
    feed(det, [600] * 8, t0=end + 0.05)  # 끝나고 0.3초 안: 아직 0.8
    feed(det, [600] * 8, t0=end + 3.0)  # 유예가 지났고 앞 감지에서 2초도 지났다

    assert [s for _, s in hits] == [0.9, 0.6]
    assert hits[1][0] > end + 3.0


def test_콜백이_예외를_내도_감지는_계속된다():
    calls = []

    def boom(t, s):
        calls.append(t)
        raise RuntimeError("콜백 오류")

    det = detector([], on_detect=boom)

    feed(det, [900] * 8)
    feed(det, [900] * 8, t0=3.0)

    assert len(calls) == 2


def test_start_하면_MicStream_을_읽어_감지하고_stop_하면_멈춘다():
    go, hits = threading.Event(), []

    def source():
        go.wait(5)
        for v in [0] * 10 + [900] * 3 + [0] * 10:
            yield np.full(BLOCK, v, np.int16), 0.0
            time.sleep(0.001)

    with MicStream(source=source()) as mic:
        det = detector(hits, mic=mic)
        det.start()
        go.set()
        det._thread.join(5)  # 소리를 다 흘리면 MicStream 이 닫히고 감지 스레드도 끝난다
        det.stop()

    assert len(hits) == 1
    assert not det._thread.is_alive()


class FakeClient:
    """openai 클라이언트 대신 받아쓰기 요청을 세고, 정해 둔 글자를 돌려준다."""

    def __init__(self):
        self.requests = []
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(text="자비서, 저거 뭐야?")


def test_listen_은_공유_마이크에서_since_뒤에_말한_것만_받아쓴다(monkeypatch):
    """호출어를 잡은 뒤 listen() 을 늦게 불러도, since 이후에 이미 지나간 말을 받아쓴다."""
    with wave.open(str(DATA / "jarviseo_question.wav"), "rb") as f:
        voice = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    silence = np.zeros(SAMPLE_RATE, np.int16)
    audio = np.concatenate([silence, voice, silence, silence])
    blocks = [
        (audio[i : i + BLOCK], i / SAMPLE_RATE) for i in range(0, len(audio) - BLOCK + 1, BLOCK)
    ]
    monkeypatch.setattr(stt, "MIC_GATE", MicGate())  # 다른 테스트의 재생 구간과 섞이지 않게

    with MicStream(source=blocks, history_sec=10.0) as mic:
        time.sleep(0.2)  # 소리가 다 지나간 뒤에 부른다
        client = FakeClient()
        listener = stt.SpeechToText(mic=mic, client=client)
        before = listener.listen(since=0.5, timeout=2.0)  # 말 시작(1초) 앞을 잡았다
        after = listener.listen(since=1.2 + len(voice) / SAMPLE_RATE, timeout=2.0)  # 말이 끝난 뒤

    assert before.text == "저거 뭐야?"
    assert before.started_at == pytest.approx(1.0, abs=0.06)
    assert after is None and len(client.requests) == 1
