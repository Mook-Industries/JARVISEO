"""시작음 · 종료음.  담당: 문태현

"자비서"를 알아들으면 시작음을 내서 듣고 있다고 알리고, 턴이 끝나 다시 호출을 기다리게 되면
종료음을 내서 다시 불러도 된다고 알린다(#104). 소리는 ``sounds/`` 의 WAV(24kHz 16비트 모노)다.
받은 파일(mp3·m4a)의 형식만 바꿨고, 앞뒤 무음은 읽을 때 잘라서 시작음 0.39초, 종료음 0.35초다.

시작음은 사용자가 "자비서, 이거 뭐야"처럼 이어 말하는 도중에 난다. 그래서 마이크 게이트를
닫지 않는다. 닫으면 시작음이 나는 동안 말한 질문이 무음으로 바뀐다. 대신 시작음이 마이크에
같이 들어간다. 받아쓰기에 끼치는 영향은 docs/experiments.md (#104) 에 있다.
종료음은 TTS 처럼 게이트를 닫고 낸다. 이때는 사용자가 말하지 않고, 자기 소리를 받아쓰면 안 된다.
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np

from jarviseo import config
from jarviseo.voice.gate import MicGate
from jarviseo.voice.speaker import SAMPLE_RATE, Speaker
from jarviseo.voice.tts import trim_silence

__all__ = ["END_WAV", "START_WAV", "Chime", "load_sound"]

SOUNDS_DIR = Path(__file__).with_name("sounds")
START_WAV = SOUNDS_DIR / "start.wav"
END_WAV = SOUNDS_DIR / "end.wav"


class Chime:
    """호출을 받은 직후 ``start()``, 턴이 끝나 다시 호출을 기다릴 때 ``end()``."""

    def __init__(
        self,
        volume: float = config.CHIME_VOLUME,
        device: int | None = None,
        gate: MicGate | None = None,
    ) -> None:
        """volume 은 0~1 (0 이면 아무 소리도 안 낸다). device 는 sounddevice 출력 장치 번호.

        gate 는 테스트에서만 넘긴다. 기본은 STT·TTS 와 같이 쓰는 ``MIC_GATE``.
        """
        self.start_pcm = load_sound(START_WAV, volume)
        self.end_pcm = load_sound(END_WAV, volume)
        # 소리를 통째로 들고 있어서 출력 버퍼를 작게 잡아도 끊기지 않는다. 첫 소리가 0.1초쯤 빠르다.
        self.speaker = Speaker(device, gate, latency="low")
        self._playing: threading.Thread | None = None

    def start(self) -> None:
        """시작음을 다른 스레드에서 내고 바로 돌아온다. 마이크 게이트는 닫지 않는다.

        바로 이어서 ``listen()`` 을 불러야 시작음이 나는 동안 말한 질문도 받아쓴다.
        첫 조각을 스피커에 쓴 시각은 ``speaker.started_at`` 에 남는다.
        """
        if not self.start_pcm:
            return
        self._playing = threading.Thread(
            target=self.speaker.play,
            args=([self.start_pcm],),
            kwargs={"mute_mic": False},
            name="chime-start",
            daemon=True,
        )
        self._playing.start()

    def end(self) -> None:
        """종료음을 내고 다 날 때까지 기다린다. 그동안과 끝난 뒤 0.3초는 마이크 게이트를 닫는다."""
        self.wait()  # 스피커 하나로 두 소리를 내서, 시작음이 아직 나고 있으면 먼저 끝낸다
        if self.end_pcm:
            self.speaker.play([self.end_pcm])

    def wait(self) -> None:
        """시작음이 나고 있으면 다 날 때까지 기다린다."""
        if self._playing is not None:
            self._playing.join()
            self._playing = None


def load_sound(path: Path, volume: float = 1.0) -> bytes:
    """WAV(24kHz 16비트 모노)를 읽어 앞뒤 무음을 자르고 volume 을 곱한 PCM 을 돌려준다.

    volume 이 0 이하면 빈 바이트다. 형식이 다르면 ValueError (TTS 와 같은 형식이어야
    ``Speaker`` 가 그대로 낸다).
    """
    if volume <= 0:
        return b""
    with wave.open(str(path), "rb") as w:
        shape = (w.getframerate(), w.getnchannels(), w.getsampwidth())
        if shape != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path.name} 는 24kHz 16비트 모노 WAV 여야 한다: {shape}")
        pcm = w.readframes(w.getnframes())
    samples = np.frombuffer(trim_silence(pcm), np.int16) * volume
    return np.clip(np.round(samples), -32768, 32767).astype(np.int16).tobytes()
