"""스피커 출력.  담당: 문태현

TTS 가 받는 대로 내는 PCM 조각을 순서대로 재생한다. 합성이 다 끝날 때까지 기다리지 않고
첫 조각이 오면 바로 소리를 낸다. 재생하는 동안은 마이크 게이트(``gate.py``)를 닫는다.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

from jarviseo.voice.gate import MIC_GATE, MicGate

__all__ = ["SAMPLE_RATE", "Speaker"]

# OpenAI TTS 의 response_format="pcm" 은 24kHz 16비트 모노(리틀엔디언)로 온다.
SAMPLE_RATE = 24000


class Speaker:
    """``Speaker().play(tts.synthesize(text))``"""

    def __init__(self, device: int | None = None, gate: MicGate | None = None) -> None:
        """gate 는 테스트에서만 넘긴다. 기본은 STT 와 같이 쓰는 ``MIC_GATE``."""
        self.device = device
        self.gate = gate or MIC_GATE

    def play(self, chunks: Iterable[bytes]) -> None:
        """조각을 받는 순서대로 재생하고, 마지막 소리가 다 나올 때까지 기다린다.

        첫 조각이 오기 전에는 스피커도 게이트도 건드리지 않는다. chunks 가 도중에 예외를 내면
        그때까지 받은 소리는 마저 내고 예외를 그대로 올린다.
        """
        chunks = iter(chunks)
        first = next(chunks, None)
        if first is None:
            return
        # 리눅스 CI 에는 PortAudio 가 없어서, 맨 위에서 import 하면 이 모듈을 읽기만 해도 죽는다.
        import sounddevice as sd

        # with 를 빠져나갈 때 stop() 이 버퍼에 남은 소리를 다 낼 때까지 기다린다(약 0.2초).
        # 게이트는 그다음에 풀어야 해서 바깥쪽에 둔다.
        with (
            self.gate.playing(),
            sd.RawOutputStream(
                samplerate=SAMPLE_RATE, device=self.device, channels=1, dtype="int16"
            ) as out,
        ):
            for chunk in itertools.chain([first], chunks):
                out.write(chunk)
