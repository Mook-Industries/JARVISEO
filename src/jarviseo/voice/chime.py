"""시작음 · 종료음.  담당: 문태현

"자비서"를 알아들으면 시작음을 내서 듣고 있다고 알리고, 사용자가 질문을 다 말해 말이 끝났다고
판정하면 종료음을 내서 다 들었다고 알린다(#104). 대답이 끝났다는 소리가 아니다.
소리는 ``sounds/`` 의 WAV(24kHz 16비트 모노)다. 받은 파일(mp3·m4a)의 형식만 바꿨고,
앞뒤 무음은 읽을 때 잘라서 시작음 0.39초, 종료음 0.35초다.

둘 다 다른 스레드에서 내고 바로 돌아온다. 시작음은 듣기와 같이, 종료음은 받아쓰기와 같이 간다.
마이크 게이트는 닫지 않는다. 시작음은 사용자가 "자비서, 이거 뭐야"처럼 이어 말하는 도중에 나서,
닫으면 그동안 말한 질문이 무음으로 바뀐다. 종료음도 말끝 판정이 틀렸을 때(이어서 더 말할 때)
말을 지우지 않게 닫지 않는다. 대신 소리가 마이크에 들어가는 구간을 ``MIC_GATE.sounding()`` 으로
표시해서 STT 의 VAD 가 말로 보지 않게 한다. 받아쓰기에 끼치는 영향은
docs/experiments.md (#104) 에 있다.
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
    """호출을 받은 직후 ``start()``, 사용자 말이 끝났다고 판정하면 ``end()``.

    ``end`` 는 ``SpeechToText(on_speech_end=chime.end)`` 로 넘기면 받아쓰기 전에 불린다.
    """

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
        # 쓰는 것도 통째로 한 번에 한다. 20ms 씩 나눠 쓰면 다른 파이썬 스레드가 GIL 을 잡고 바쁠 때
        # 조각 사이에 버퍼가 비어 소리가 끊겼다(중간이 최대 0.1~0.8초 빔, check_chime_load.py).
        self.speaker = Speaker(device, gate, latency="low", piece_bytes=None)
        # 마지막 시작음·종료음의 첫 조각을 스피커에 쓴 시각(monotonic). 소리는 출력 지연 뒤에 난다.
        self.start_played_at: float | None = None
        self.end_played_at: float | None = None
        self._playing: threading.Thread | None = None

    def start(self) -> None:
        """시작음을 다른 스레드에서 내고 바로 돌아온다. 이어서 ``listen()`` 을 부른다."""
        self._play(self.start_pcm, "start")

    def end(self) -> None:
        """종료음을 다른 스레드에서 내고 바로 돌아온다. 받아쓰기를 늦추지 않는다."""
        self._play(self.end_pcm, "end")

    def wait(self) -> None:
        """내고 있는 소리가 있으면 다 날 때까지 기다린다."""
        if self._playing is not None:
            self._playing.join()

    def _play(self, pcm: bytes, which: str) -> None:
        if not pcm:
            return
        before = self._playing

        def play() -> None:
            # 스피커 하나로 두 소리를 내서, 앞 소리가 아직 나고 있으면 그게 끝난 뒤에 낸다
            if before is not None:
                before.join()
            self.speaker.play([pcm], mute_mic=False)
            if which == "start":
                self.start_played_at = self.speaker.started_at
            else:
                self.end_played_at = self.speaker.started_at

        self._playing = threading.Thread(target=play, name=f"chime-{which}", daemon=True)
        self._playing.start()


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
