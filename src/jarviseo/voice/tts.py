"""음성 합성 (TTS).  담당: 문태현

답변 문장을 OpenAI TTS 로 합성한다. 모델 이름은 ``config.TTS_MODEL`` 하나만 본다.
``synthesize()`` 가 PCM 조각을 받는 대로 내고, 재생과 ``speak()`` 는 다음 이슈에서 붙인다.
"""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from jarviseo import config
from jarviseo.voice.tts_text import normalize

__all__ = ["SAMPLE_RATE", "TextToSpeech"]

# response_format="pcm" 은 24kHz 16비트 모노(리틀엔디언)로 온다.
SAMPLE_RATE = 24000
CHUNK_BYTES = SAMPLE_RATE * 2 // 10  # 100ms 씩 받는다
# user_setting.tts_speed → OpenAI speed 값. instructions 로 "천천히"를 부탁하면 같은 문장도
# 길이가 ±0.6초씩 흔들려 효과가 안 보였고, speed 는 0.8 → 1.31배, 1.25 → 0.79배로 일정했다.
SPEEDS = {"SLOW": 0.85, "NORMAL": 1.0, "FAST": 1.2}
# 한 번 기다리는 최대 시간(초)과 다시 보내는 횟수.
TIMEOUT_SEC, RETRIES = 10.0, 1


class TextToSpeech:
    """글자를 소리로 바꿔 재생한다."""

    def __init__(
        self,
        model: str = config.TTS_MODEL,
        voice: str = "coral",
        speed: str = "NORMAL",
        client: OpenAI | None = None,
    ) -> None:
        """voice · speed 는 ``user_setting`` 의 tts_voice · tts_speed (SLOW/NORMAL/FAST).

        client 는 테스트에서 가짜 API 를 끼울 때만 넘긴다.
        """
        self.model = model
        self.voice = voice
        self.speed = speed
        self.client = client or OpenAI(
            api_key=config.OPENAI_API_KEY or None, timeout=TIMEOUT_SEC, max_retries=RETRIES
        )

    def speak(self, text: str) -> float:
        """text 를 읽어 준다. 재생이 끝날 때까지 기다린다.

        첫 오디오 청크를 받기까지 걸린 시간(ms)을 돌려준다. 이 값이 ``turn_voice.tts_ms`` 다.
        """
        raise NotImplementedError("다음 이슈에서 synthesize() 와 재생을 묶어 구현한다.")

    def synthesize(self, text: str) -> Iterator[bytes]:
        """text 를 읽기 좋게 다듬어 합성하고, PCM(24kHz 16비트 모노) 조각을 받는 대로 낸다."""
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=normalize(text),
            speed=SPEEDS.get(self.speed.upper(), 1.0),
            response_format="pcm",
        ) as response:
            yield from response.iter_bytes(CHUNK_BYTES)
