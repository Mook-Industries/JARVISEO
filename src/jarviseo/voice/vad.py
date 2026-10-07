"""발화 구간 검출 (VAD).  담당: 문태현

webrtcvad 로 30ms 블록마다 사람 말인지 판정하고, 발화 하나를 잘라 낸다.

- 시작: 최근 300ms(10블록) 중 8블록 이상이 말일 때. 블록 하나로 정하면 첫 블록의 잡음이나
  숨소리 한 번에도 시작으로 본다. 시작 시각은 그 300ms 안에서 처음 말로 판정된 블록의 시각이다.
- 끝: 시작 뒤 700ms 동안 말이 없을 때. 끝 시각은 마지막으로 말한 블록이 끝난 시각이다.
  말이 10초를 넘으면 거기서 자른다.

스파이크 02(#13)에서 소리 크기 문턱값 방식은 조용한 방에서 말 끝을 7초 넘게 늦게 잡았다.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import webrtcvad

from jarviseo.voice.mic import BLOCK, SAMPLE_RATE

__all__ = ["Speech", "record_speech", "voiced_seconds"]

BLOCK_SEC = BLOCK / SAMPLE_RATE
WINDOW, TRIGGER = 10, 8  # 최근 10블록 중 8블록 이상이 말이면 시작


@dataclass
class Speech:
    """잘라 낸 발화 한 건."""

    audio: np.ndarray  # int16, 16kHz 모노. 시작 앞 300ms 까지 붙어 있어 첫 음절이 안 잘린다
    started_at: float  # 말을 시작한 시각 (monotonic)
    ended_at: float  # 마지막으로 말한 블록이 끝난 시각 (monotonic)


def _webrtcvad(mode: int = 3) -> Callable[[np.ndarray], bool]:
    vad = webrtcvad.Vad(mode)  # 0~3. 3 이 말로 판정하는 기준이 가장 엄격하다
    return lambda block: vad.is_speech(block.tobytes(), SAMPLE_RATE)


def voiced_seconds(audio: np.ndarray) -> float:
    """audio(int16, 16kHz 모노) 안에서 사람 말로 판정된 길이(초). 클립에 말이 들었는지 볼 때 쓴다.

    발화를 자를 때(모드 3)보다 너그러운 모드 2 로 센다. 모드 3 은 짧게 부른 "자비서" 실제 녹음도
    0.24초 남짓으로 쳐서, 소리가 거의 없는 합성 클립(0.21초 이하)과 구분이 안 됐다.
    모드 2 로는 실제 녹음이 가장 짧아도 0.39초였다.
    """
    is_speech = _webrtcvad(mode=2)
    blocks = (audio[i : i + BLOCK] for i in range(0, len(audio) - BLOCK + 1, BLOCK))
    return sum(is_speech(block) for block in blocks) * BLOCK_SEC


def record_speech(
    blocks: Iterable[tuple[np.ndarray, float]],
    on_speech_start: Callable[[float], None] | None = None,
    timeout: float = 5.0,
    max_sec: float = 10.0,
    silence_sec: float = 0.7,
    is_speech: Callable[[np.ndarray], bool] | None = None,
) -> Speech | None:
    """블록을 읽다가 발화 하나를 잘라 돌려준다.

    말 시작을 잡는 순간 ``on_speech_start(started_at)`` 을 부른다. 프레임은 이때 골라야 한다.
    ``FrameBuffer`` 는 약 1초치만 들고 있어서, 받아쓰기가 끝난 뒤에 고르면 이미 늦다.

    timeout 초 안에 시작을 못 잡으면 None. 말이 max_sec 를 넘으면 거기서 자른다.
    is_speech 는 테스트에서 판정을 바꿔 끼울 때만 넘긴다.
    """
    is_speech = is_speech or _webrtcvad()
    ring: deque[tuple[np.ndarray, float, bool]] = deque(maxlen=WINDOW)
    kept: list[np.ndarray] = []
    first_t = started_at = None
    voiced_until = 0.0
    for block, t in blocks:
        if first_t is None:
            first_t = t
        voiced = is_speech(block)
        if started_at is None:
            ring.append((block, t, voiced))
            if sum(v for _, _, v in ring) >= TRIGGER:
                started_at = next(bt for _, bt, v in ring if v)
                voiced_until = t + BLOCK_SEC
                kept = [b for b, _, _ in ring]
                if on_speech_start is not None:
                    on_speech_start(started_at)
            elif t + BLOCK_SEC - first_t >= timeout:
                return None
            continue
        kept.append(block)
        if voiced:
            voiced_until = t + BLOCK_SEC
        if t + BLOCK_SEC - voiced_until >= silence_sec or t + BLOCK_SEC - started_at >= max_sec:
            break
    if started_at is None:
        return None
    return Speech(np.concatenate(kept), started_at, voiced_until)


if __name__ == "__main__":  # 마이크로 직접 확인: python -m jarviseo.voice.vad [장치번호]
    import sys
    import time

    from jarviseo.voice.mic import Microphone

    with Microphone(int(sys.argv[1]) if len(sys.argv) > 1 else None) as mic:
        print("말해 보세요")
        speech = record_speech(mic.blocks(), lambda t: print(f"시작 {t:.3f}"))
        stopped = time.monotonic()
    if speech is None:
        print("말을 못 잡았다")
    else:
        took = speech.ended_at - speech.started_at
        print(f"끝   {speech.ended_at:.3f}  말한 길이 {took:.2f}s")
        print(f"말 끝 → 녹음 멈춤 {(stopped - speech.ended_at) * 1000:.0f}ms")
