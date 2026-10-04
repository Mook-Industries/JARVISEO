"""음성 합성 (TTS).  담당: 문태현

답변 문장을 OpenAI TTS 로 합성해 재생한다. 모델 이름은 ``config.TTS_MODEL`` 하나만 본다.
``gpt-4o-mini-tts`` 에는 speed 파라미터가 없어서, 말하기 속도(``user_setting.tts_speed``)는
instructions 문구로 바꿔 보낸다.
"""

from __future__ import annotations

from jarviseo import config

__all__ = ["TextToSpeech"]


class TextToSpeech:
    """글자를 소리로 바꿔 재생한다."""

    def __init__(
        self, model: str = config.TTS_MODEL, voice: str = "coral", speed: str = "NORMAL"
    ) -> None:
        """voice · speed 는 ``user_setting`` 의 tts_voice · tts_speed (SLOW/NORMAL/FAST)."""
        self.model = model
        self.voice = voice
        self.speed = speed

    def speak(self, text: str) -> float:
        """text 를 읽어 준다. 재생이 끝날 때까지 기다린다.

        첫 오디오 청크를 받기까지 걸린 시간(ms)을 돌려준다. 이 값이 ``turn_voice.tts_ms`` 다.
        """
        raise NotImplementedError("다음 이슈에서 구현한다. 그때까지는 FakeTextToSpeech 를 쓴다.")
