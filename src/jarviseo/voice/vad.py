"""발화 구간 검출 (VAD).  담당: 문태현

webrtcvad 로 30ms 블록마다 사람 말인지 판정하고, 발화 하나를 잘라 낸다.

- 시작: 최근 300ms(10블록) 중 8블록 이상이 말일 때. 블록 하나로 정하면 첫 블록의 잡음이나
  숨소리 한 번에도 시작으로 본다.
- 끝: 시작 뒤 700ms 동안 말이 없을 때.

스파이크 02(#13)에서 소리 크기 문턱값 방식은 조용한 방에서 말 끝을 7초 넘게 늦게 잡았다.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import webrtcvad

from jarviseo.voice.mic import BLOCK, SAMPLE_RATE

__all__ = ["Speech", "record_speech"]

BLOCK_SEC = BLOCK / SAMPLE_RATE
WINDOW, TRIGGER = 10, 8  # 최근 10블록 중 8블록 이상이 말이면 시작


@dataclass
class Speech:
    """잘라 낸 발화 한 건."""

    audio: np.ndarray  # int16, 16kHz 모노. 시작 앞 300ms 까지 붙어 있어 첫 음절이 안 잘린다


def _webrtcvad() -> Callable[[np.ndarray], bool]:
    vad = webrtcvad.Vad(3)  # 0~3. 3 이 말로 판정하는 기준이 가장 엄격하다
    return lambda block: vad.is_speech(block.tobytes(), SAMPLE_RATE)


def record_speech(
    blocks: Iterable[tuple[np.ndarray, float]],
    silence_sec: float = 0.7,
    is_speech: Callable[[np.ndarray], bool] | None = None,
) -> Speech | None:
    """블록을 읽다가 발화 하나를 잘라 돌려준다.

    is_speech 는 테스트에서 판정을 바꿔 끼울 때만 넘긴다.
    """
    is_speech = is_speech or _webrtcvad()
    ring: deque[tuple[np.ndarray, bool]] = deque(maxlen=WINDOW)
    kept: list[np.ndarray] = []
    started, voiced_until = False, 0.0
    for block, t in blocks:
        voiced = is_speech(block)
        if not started:
            ring.append((block, voiced))
            if sum(v for _, v in ring) >= TRIGGER:
                started, voiced_until = True, t + BLOCK_SEC
                kept = [b for b, _ in ring]
            continue
        kept.append(block)
        if voiced:
            voiced_until = t + BLOCK_SEC
        if t + BLOCK_SEC - voiced_until >= silence_sec:
            break
    return Speech(np.concatenate(kept)) if started else None
