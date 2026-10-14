"""음성 합성 (TTS).  담당: 문태현

답변 문장을 OpenAI TTS 로 합성한다. 모델 이름은 ``config.TTS_MODEL`` 하나만 본다.
``synthesize()`` 가 PCM 조각을 받는 대로 내고, ``speak()`` 는 그 조각을 받는 대로 재생한다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import httpx2
from openai import OpenAI, OpenAIError

from jarviseo import config
from jarviseo.voice.speaker import SAMPLE_RATE, Speaker
from jarviseo.voice.tts_text import normalize

__all__ = ["SAMPLE_RATE", "TextToSpeech"]

log = logging.getLogger(__name__)

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
        device: int | None = None,
        client: OpenAI | None = None,
    ) -> None:
        """voice · speed 는 ``user_setting`` 의 tts_voice · tts_speed (SLOW/NORMAL/FAST).
        device 는 sounddevice 출력 장치 번호다. None 이면 기본 스피커.

        client 는 테스트에서 가짜 API 를 끼울 때만 넘긴다.
        """
        self.model = model
        self.voice = voice
        self.speed = speed
        self.speaker = Speaker(device)
        self.client = client or OpenAI(
            api_key=config.OPENAI_API_KEY or None, timeout=TIMEOUT_SEC, max_retries=RETRIES
        )
        # 마지막 합성에서 첫 조각을 받기까지 걸린 시간(ms, TTFB). 아직 못 받았으면 None.
        # 그래프가 latency_ms["tts"] 로 넘기면 log_turn 이 turn_voice.tts_ms 에 적는다.
        self.last_latency_ms: float | None = None
        # 마지막 speak() 이 stop() 으로 중간에 끊겼으면 True (turn_voice.is_interrupted).
        self.last_interrupted = False

    def speak(self, text: str) -> float | None:
        """text 를 합성하면서 받는 대로 읽어 준다. 재생이 끝나거나 stop() 으로 끊길 때까지 기다린다.

        첫 오디오 청크를 받기까지 걸린 시간(ms)을 돌려준다. 이 값이 ``turn_voice.tts_ms`` 다.
        합성이 도중에 실패하면 그때까지 받은 소리만 내고 멈춘다. 소리를 하나도 못 냈으면 None.
        끝까지 읽었는지는 ``last_interrupted`` 로 본다.
        """
        self.last_interrupted = False
        chunks = self.synthesize(text)
        try:
            self.speaker.play(chunks)
        except (OpenAIError, httpx2.HTTPError) as e:
            # 요청이 실패하면 OpenAIError 로 오지만, 받는 도중에 끊기면 openai 가 감싸지 않은
            # httpx2.ReadTimeout 같은 것이 그대로 올라온다. 어느 쪽이든 음성 루프는 멈추지 않는다.
            log.warning("합성 실패, 재생을 멈춘다: %s", e)
        finally:
            # 끊겨서 덜 읽은 응답 스트림을 바로 닫는다. 그냥 두면 GC 될 때까지 연결이 남는다.
            chunks.close()
        self.last_interrupted = self.speaker.interrupted
        return self.last_latency_ms

    def prepare(self, text: str) -> bytes:
        """text 를 미리 끝까지 합성해 PCM 을 통째로 돌려준다. 실패하면 빈 바이트.

        자주 읽는 짧은 말("잠깐만요")을 시작할 때 한 번 만들어 둔다. 그때그때 합성하면
        첫 소리까지 약 1초가 걸려서, 기다리는 사람에게 너무 늦게 나간다.
        """
        try:
            return b"".join(self.synthesize(text))
        except (OpenAIError, httpx2.HTTPError) as e:
            log.warning("미리 합성 실패: %s", e)
            return b""

    def play(self, pcm: bytes) -> None:
        """prepare() 로 만든 소리를 바로 재생한다. speak() 처럼 stop() 으로 끊긴다."""
        self.last_interrupted = False
        if pcm:
            self.speaker.play([pcm])
        self.last_interrupted = self.speaker.interrupted

    def stop(self) -> None:
        """읽던 것을 바로 멈춘다(끼어들기). 호출어 감지 스레드에서 부른다.

        speak() 은 남은 합성을 받지 않고 곧 돌아오고, last_interrupted 가 True 가 된다.
        읽는 중이 아니면 아무 일도 없다.
        """
        self.speaker.stop()

    def synthesize(self, text: str) -> Iterator[bytes]:
        """text 를 읽기 좋게 다듬어 합성하고, PCM(24kHz 16비트 모노) 조각을 받는 대로 낸다.

        첫 조각을 받기까지 걸린 시간(ms)을 ``last_latency_ms`` 에 남긴다.
        """
        speed = SPEEDS.get(self.speed.upper(), 1.0)
        self.last_latency_ms = None
        asked_at = time.monotonic()
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=normalize(text),
            # NORMAL 은 speed 를 아예 안 보낸다. 속도를 바꾸면 첫 소리가 늦어졌다
            # (첫 조각 중앙값: 안 보냄 약 1.0초, 1.2 는 약 1.8초).
            **({} if speed == 1.0 else {"speed": speed}),
            response_format="pcm",
        ) as response:
            for chunk in response.iter_bytes(CHUNK_BYTES):
                if self.last_latency_ms is None:
                    self.last_latency_ms = (time.monotonic() - asked_at) * 1000
                yield chunk
