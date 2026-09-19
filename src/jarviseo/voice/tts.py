"""음성 합성 (TTS).  담당: 문태현

답변을 소리로 내보낸다.

되먹임 방지가 이 파일의 책임이다
--------------------------------
speak() 는 재생을 시작할 때 웨이크워드 감지기를 재우고,
재생이 끝난 뒤 유예시간(기본 0.3초)까지 지나고 나서 깨워야 한다.
유예를 두는 이유는 스피커 잔향과 오디오 버퍼 지연 때문이다.
재생 종료 즉시 깨우면 마지막 음절이 마이크에 잡힌다.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["TextToSpeech"]


class TextToSpeech:
    """글자를 소리로 바꿔 재생한다."""

    def __init__(
        self,
        voice: str = "ko",
        mic_gate: Callable[[float], None] | None = None,
        gate_tail_sec: float = 0.3,
    ) -> None:
        """
        Args:
            mic_gate: 재생하는 동안 마이크를 막을 콜백.
                보통 WakeWordDetector.mute 를 넘긴다.
            gate_tail_sec: 재생이 끝난 뒤 추가로 막을 시간.
        """
        self.voice = voice
        self.mic_gate = mic_gate
        self.gate_tail_sec = gate_tail_sec
        raise NotImplementedError

    def speak(self, text: str) -> None:
        """재생이 끝날 때까지 기다린다(블로킹)."""
        raise NotImplementedError
