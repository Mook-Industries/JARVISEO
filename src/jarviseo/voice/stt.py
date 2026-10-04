"""받아쓰기 (STT).  담당: 문태현

호출어 뒤의 발화 한 번만 OpenAI 받아쓰기 API 로 보낸다.
상시 듣는 호출어 감지는 로컬(``wakeword.py``)에 남는다.
모델 이름은 ``config.STT_MODEL`` 하나만 본다.

결과에서 가장 중요한 값은 글자가 아니라 ``Utterance.started_at`` (말을 시작한 시각)이다.
프레임은 이 시각 기준으로 고르는데, ``FrameBuffer`` 는 약 1초치만 들고 있다.
그래서 ``listen()`` 은 받아쓰기가 끝날 때까지 기다리게 하지 않고, 말 시작을 잡는 순간
``on_speech_start(started_at)`` 을 불러 준다. 프레임은 그 콜백에서 고른다.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from jarviseo import config
from jarviseo.types import Utterance

if TYPE_CHECKING:
    import numpy as np

__all__ = ["SpeechToText"]

_TODO = "다음 이슈에서 구현한다. 그때까지는 FakeSpeechToText 를 쓴다."


class SpeechToText:
    """마이크에서 발화 하나를 받아 글자로 바꾼다."""

    def __init__(
        self, model: str = config.STT_MODEL, language: str = "ko", device: int | None = None
    ) -> None:
        """device 는 sounddevice 입력 장치 번호다. None 이면 기본 마이크."""
        self.model = model
        self.language = language
        self.device = device

    def listen(
        self, on_speech_start: Callable[[float], None] | None = None, timeout: float = 5.0
    ) -> Utterance | None:
        """마이크에서 발화 하나를 받아쓴다.

        말 시작을 잡는 순간 ``on_speech_start(started_at)`` 을 부른다.
        timeout 초 안에 말이 시작되지 않으면 None.
        """
        raise NotImplementedError(_TODO)

    def transcribe(self, audio: np.ndarray, started_at: float, ended_at: float) -> Utterance:
        """이미 녹음된 발화(16kHz 모노 int16)를 받아쓴다. 녹음 파일로 재현할 때 쓴다."""
        raise NotImplementedError(_TODO)
