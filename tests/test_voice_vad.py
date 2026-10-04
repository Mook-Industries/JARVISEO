"""VAD 가 발화를 맞게 잘라 내는지 확인한다. 마이크 없이 돈다."""

import numpy as np
import pytest

from jarviseo.voice.mic import BLOCK, SAMPLE_RATE
from jarviseo.voice.vad import record_speech

T0 = 100.0  # 첫 블록의 monotonic 시각. 값 자체는 상관없다
BLOCK_SEC = BLOCK / SAMPLE_RATE


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
