"""VAD 가 발화를 맞게 잘라 내는지 확인한다. 마이크 없이 돈다."""

import wave
from pathlib import Path

import numpy as np
import pytest

from jarviseo.voice.mic import BLOCK, SAMPLE_RATE
from jarviseo.voice.vad import record_speech

T0 = 100.0  # 첫 블록의 monotonic 시각. 값 자체는 상관없다
BLOCK_SEC = BLOCK / SAMPLE_RATE
DATA = Path(__file__).parent / "data"


def blocks_of(audio: np.ndarray):
    """마이크 대신 오디오를 30ms 블록과 시각으로 흘려 넣는다."""
    for n in range(len(audio) // BLOCK):
        yield audio[n * BLOCK : (n + 1) * BLOCK], T0 + n * BLOCK_SEC


def test_말이_없으면_timeout_뒤에_None_을_준다():
    judged = []

    def never(block):
        judged.append(block)
        return False

    silence = np.zeros(SAMPLE_RATE * 10, np.int16)

    assert record_speech(blocks_of(silence), timeout=2.0, is_speech=never) is None
    assert len(judged) * BLOCK_SEC == pytest.approx(2.0, abs=BLOCK_SEC)  # 10초를 다 읽지 않는다


def test_말이_길면_max_sec_에서_자른다():
    audio = np.zeros(SAMPLE_RATE * 15, np.int16)

    speech = record_speech(blocks_of(audio), max_sec=10.0, is_speech=lambda block: True)

    assert speech.ended_at - speech.started_at == pytest.approx(10.0, abs=BLOCK_SEC)


def test_녹음_파일로_started_at_을_검증한다():
    """jarviseo_question.wav 는 gpt-4o-mini-tts 로 만든 "자비서, 저거 뭐야?" 다.

    앞뒤 무음을 잘라 둬서, 앞에 1초를 붙이면 실제 말 시작은 정확히 1초 지점이다.
    """
    with wave.open(str(DATA / "jarviseo_question.wav"), "rb") as f:
        assert f.getframerate() == SAMPLE_RATE
        voice = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    lead = SAMPLE_RATE
    audio = np.concatenate([np.zeros(lead), voice, np.zeros(SAMPLE_RATE * 2)])
    audio += np.random.default_rng(0).normal(0, 60, len(audio))  # 조용한 방 수준의 잡음
    starts = []

    speech = record_speech(blocks_of(audio.astype(np.int16)), on_speech_start=starts.append)

    assert starts == [speech.started_at]  # 시작 콜백은 한 번, 같은 시각으로
    assert speech.started_at == pytest.approx(T0 + lead / SAMPLE_RATE, abs=0.06)
    assert speech.ended_at == pytest.approx(T0 + (lead + len(voice)) / SAMPLE_RATE, abs=0.15)
