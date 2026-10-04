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

import io
import logging
import wave
from collections.abc import Callable
from typing import TYPE_CHECKING

from openai import OpenAI, OpenAIError

from jarviseo import config
from jarviseo.types import Utterance
from jarviseo.voice.mic import SAMPLE_RATE, Microphone
from jarviseo.voice.vad import record_speech

if TYPE_CHECKING:
    import numpy as np

__all__ = ["SpeechToText"]

log = logging.getLogger(__name__)

# 한 번 기다리는 최대 시간(초)과 다시 보내는 횟수. 왕복은 보통 1~2.5초였다(스파이크 02).
TIMEOUT_SEC, RETRIES = 8.0, 1


class SpeechToText:
    """마이크에서 발화 하나를 받아 글자로 바꾼다."""

    def __init__(
        self,
        model: str = config.STT_MODEL,
        language: str = "ko",
        device: int | None = None,
        client: OpenAI | None = None,
    ) -> None:
        """device 는 sounddevice 입력 장치 번호다. None 이면 기본 마이크.

        client 는 테스트에서 가짜 API 를 끼울 때만 넘긴다.
        """
        self.model = model
        self.language = language
        self.device = device
        # 키가 비어 있으면 None 으로 넘긴다. 그래야 첫 요청이 아니라 여기서 바로 알려 준다.
        self.client = client or OpenAI(
            api_key=config.OPENAI_API_KEY or None, timeout=TIMEOUT_SEC, max_retries=RETRIES
        )

    def listen(
        self, on_speech_start: Callable[[float], None] | None = None, timeout: float = 5.0
    ) -> Utterance | None:
        """마이크에서 발화 하나를 받아쓴다.

        말 시작을 잡는 순간 ``on_speech_start(started_at)`` 을 부른다.
        timeout 초 안에 말이 시작되지 않거나, 받아쓰기가 끝내 실패하면 None.
        """
        # 받아쓰는 동안 마이크를 붙잡고 있지 않게, 녹음이 끝나면 바로 닫는다.
        # 호출어 감지와 마이크 하나를 같이 쓰는 것은 웨이크워드 이슈에서 다룬다.
        with Microphone(self.device) as mic:
            speech = record_speech(mic.blocks(), on_speech_start, timeout=timeout)
        if speech is None:
            return None
        try:
            return self.transcribe(speech.audio, speech.started_at, speech.ended_at)
        except OpenAIError as e:
            # 말이 없던 것과 같이 다룬다. 그래프는 호출어 대기로 돌아가면 된다.
            log.warning("받아쓰기 실패: %s", e)
            return None

    def transcribe(self, audio: np.ndarray, started_at: float, ended_at: float) -> Utterance:
        """이미 녹음된 발화(16kHz 모노 int16)를 받아쓴다. 받아쓴 원문을 그대로 담아 돌려준다."""
        result = self.client.audio.transcriptions.create(
            model=self.model,
            file=("speech.wav", _to_wav(audio)),
            language=self.language,
            # 힌트가 없으면 호출어 "자비서"가 "자비스"로 받아써졌다(스파이크 02).
            prompt=config.WAKE_WORD,
        )
        return Utterance(result.text, started_at, ended_at)


def _to_wav(audio: np.ndarray) -> bytes:
    """int16 배열을 WAV 로 감싼다. API 는 헤더를 보고 형식과 샘플레이트를 안다."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.astype("<i2").tobytes())
    return buf.getvalue()
