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
import time
import wave
from collections.abc import Callable
from typing import TYPE_CHECKING

from openai import OpenAI, OpenAIError

from jarviseo import config
from jarviseo.types import Utterance
from jarviseo.voice.gate import MIC_GATE
from jarviseo.voice.mic import SAMPLE_RATE, Microphone, MicStream
from jarviseo.voice.transcript import clean, is_question

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
        mic: MicStream | None = None,
        client: OpenAI | None = None,
    ) -> None:
        """device 는 sounddevice 입력 장치 번호다. None 이면 기본 마이크.

        mic 를 주면 호출어 감지와 같이 쓰는 그 마이크를 읽고, listen(since=...) 로 지나간 소리부터
        받을 수 있다. 없으면 listen() 때마다 마이크를 열었다 닫는다.
        client 는 테스트에서 가짜 API 를 끼울 때만 넘긴다.
        """
        self.model = model
        self.language = language
        self.device = device
        self.mic = mic
        # 키가 비어 있으면 None 으로 넘긴다. 그래야 첫 요청이 아니라 여기서 바로 알려 준다.
        self.client = client or OpenAI(
            api_key=config.OPENAI_API_KEY or None, timeout=TIMEOUT_SEC, max_retries=RETRIES
        )
        # 마지막 받아쓰기 요청의 왕복 시간(ms). 실패했으면 None.
        # 그래프가 latency_ms["stt"] 로 넘기면 log_turn 이 turn_voice.stt_ms 에 적는다.
        self.last_latency_ms: float | None = None
        # 마지막 받아쓰기 원문. listen() 은 정리한 질문을 돌려주고, 원문은 여기 남긴다
        # (turn_voice.stt_raw_text 용).
        self.last_raw_text: str | None = None

    def listen(
        self,
        on_speech_start: Callable[[float], None] | None = None,
        timeout: float = 5.0,
        since: float | None = None,
    ) -> Utterance | None:
        """마이크에서 발화 하나를 받아쓰고, 호출어·필러를 뺀 질문을 돌려준다.

        말 시작을 잡는 순간 ``on_speech_start(started_at)`` 을 부른다.
        since 는 호출어를 잡은 시각(``on_detect`` 의 t)이다. 주면 그 뒤부터 읽어서, 호출어에
        이어 한 번에 말한 질문이 잘리지 않는다. 공유 마이크(mic)가 있을 때만 쓰고,
        없으면 지금부터 듣는다.
        timeout 초 안에 말이 시작되지 않거나, 받아쓰기가 끝내 실패하거나,
        정리하고 나니 비었거나 너무 짧으면 None.
        """
        # webrtcvad 는 여기서 읽는다. jarviseo.voice 를 import 만 하는 쪽(Fake 사용)은
        # 필요 없고, python -m jarviseo.voice.vad 도 미리 읽힌 모듈 경고 없이 돈다.
        from jarviseo.voice.vad import record_speech

        self.last_raw_text = self.last_latency_ms = None
        # TTS 가 재생 중이거나 막 끝났으면 그동안 들어온 소리는 무음으로 바뀐다(gate.py).
        if self.mic is not None:
            blocks = self.mic.blocks(since=since)
            speech = record_speech(MIC_GATE.mute(blocks), on_speech_start, timeout=timeout)
        else:
            # 받아쓰는 동안 마이크를 붙잡고 있지 않게, 녹음이 끝나면 바로 닫는다.
            with Microphone(self.device) as mic:
                blocks = MIC_GATE.mute(mic.blocks())
                speech = record_speech(blocks, on_speech_start, timeout=timeout)
        if speech is None:
            return None
        try:
            raw = self.transcribe(speech.audio, speech.started_at, speech.ended_at)
        except OpenAIError as e:
            # 말이 없던 것과 같이 다룬다. 그래프는 호출어 대기로 돌아가면 된다.
            log.warning("받아쓰기 실패: %s", e)
            return None
        text = clean(raw.text)
        if not is_question(text):
            return None
        return Utterance(text, raw.started_at, raw.ended_at)

    def transcribe(self, audio: np.ndarray, started_at: float, ended_at: float) -> Utterance:
        """이미 녹음된 발화(16kHz 모노 int16)를 받아쓴다. 받아쓴 원문을 그대로 담아 돌려준다."""
        self.last_latency_ms = None
        sent_at = time.monotonic()
        result = self.client.audio.transcriptions.create(
            model=self.model,
            file=("speech.wav", _to_wav(audio)),
            language=self.language,
            # 힌트가 없으면 호출어 "자비서"가 "자비스"로 받아써졌다(스파이크 02).
            prompt=config.WAKE_WORD,
        )
        self.last_latency_ms = (time.monotonic() - sent_at) * 1000
        self.last_raw_text = result.text
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
