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
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from openai import OpenAI, OpenAIError

from jarviseo import config
from jarviseo.types import Utterance
from jarviseo.voice.gate import MIC_GATE
from jarviseo.voice.mic import SAMPLE_RATE, Microphone, MicStream
from jarviseo.voice.transcript import MIN_LETTERS, clean, is_question, is_wake_only

if TYPE_CHECKING:
    import numpy as np

    from jarviseo.voice.vad import Speech

__all__ = ["SpeechToText"]

log = logging.getLogger(__name__)

# 한 번 기다리는 최대 시간(초)과 다시 보내는 횟수. 왕복은 보통 1~2.5초였다(스파이크 02).
TIMEOUT_SEC, RETRIES = 8.0, 1
# 호출어 감지가 호출어 끝보다 늦는 최대 시간(초). 평가셋 실제 녹음 37개에서 중앙 0.12, 최대 0.33초.
# 질문은 그 뒤에 시작하므로 말 시작 시각을 감지 시각에서 이만큼 전보다 앞으로 두지 않는다.
WAKE_LAG_SEC = 0.35
# 감지 뒤 이 시간 안에 말이 끝났으면 호출어만 말하고 쉰 것으로 본다("자비서." … 질문).
# 감지는 호출어 끝보다 최대 0.33초 늦고, 끼어들기면 재생을 멈추기까지 0.2초쯤 재생 소리가 남는다.
# 이어 말한 질문은 이보다 늦게 끝난다.
WAKE_ONLY_SEC = 0.4


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
        min_letters: int = MIN_LETTERS,
        hints: Sequence[str] = (),
    ) -> Utterance | None:
        """마이크에서 발화 하나를 받아쓰고, 호출어·필러를 뺀 질문을 돌려준다.

        말 시작을 잡는 순간 ``on_speech_start(started_at)`` 을 부른다.
        since 는 호출어를 잡은 시각(``on_detect`` 의 t)이다. 공유 마이크(mic)가 있을 때만 쓰고,
        없으면 지금부터 듣는다. 주면 그보다 ``config.WAKE_LOOKBACK_SEC``(1초) 앞부터 읽는다.
        감지는 호출어가 끝나고 조금 뒤에 돼서, 바로 이어 말하면 감지 시각에는 질문이 이미
        시작돼 있다. 호출어도 같이 받아써지고 ``clean()`` 이 지운다. 말 시작 시각은 호출어가
        끝났을 때(감지 ``WAKE_LAG_SEC`` 전)보다 앞으로 두지 않는다. 프레임을 이 시각으로 고른다.
        "자비서." 하고 쉬었다가 질문하면 녹음이 호출어에서 끝나는데, 그때는 받아쓰지 않고
        이어서 한 번 더 듣는다(감지 ``WAKE_ONLY_SEC`` 안에 말이 끝났거나, 호출어만 받아써졌을 때).

        timeout 초 안에 말이 시작되지 않거나, 받아쓰기가 끝내 실패하거나,
        정리하고 나니 비었거나 min_letters 글자보다 짧으면 None.
        되묻기 대답("컵")을 들을 때는 min_letters=1 로 부른다.

        hints 는 받아쓰기에 힌트로 줄 낱말이다. 주면 호출어 힌트 대신 쓴다. 되묻기 대답을 들을 때
        후보 이름("컵", "병", "왼쪽", "오른쪽")을 주면 짧은 대답이 덜 틀린다(합성 대답 16개에서
        gpt-transcribe 맞음 6 → 12, "컵"을 "컷"으로 쓰던 것이 준다).
        """
        # webrtcvad 는 여기서 읽는다. jarviseo.voice 를 import 만 하는 쪽(Fake 사용)은
        # 필요 없고, python -m jarviseo.voice.vad 도 미리 읽힌 모듈 경고 없이 돈다.
        from jarviseo.voice.vad import record_speech

        self.last_raw_text = self.last_latency_ms = None

        def question(speech: Speech | None) -> Utterance | None:
            return self._question(speech, min_letters, hints)

        # TTS 가 재생 중이거나 막 끝났으면 그동안 들어온 소리는 무음으로 바뀐다(gate.py).
        if self.mic is None:
            # 받아쓰는 동안 마이크를 붙잡고 있지 않게, 녹음이 끝나면 바로 닫는다.
            with Microphone(self.device) as mic:
                blocks = MIC_GATE.mute(mic.blocks())
                speech = record_speech(blocks, on_speech_start, timeout=timeout)
            return question(speech)
        if since is None:
            blocks = MIC_GATE.mute(self.mic.blocks())
            speech = record_speech(blocks, on_speech_start, timeout=timeout)
            return question(speech)

        floor = since - WAKE_LAG_SEC

        def started(t: float) -> None:
            if on_speech_start is not None:
                on_speech_start(max(t, floor))

        blocks = MIC_GATE.mute(self.mic.blocks(since=since - config.WAKE_LOOKBACK_SEC))
        speech = record_speech(blocks, started, timeout=timeout)
        if speech is not None and speech.ended_at <= since + WAKE_ONLY_SEC:
            # 호출어(와 그 앞 소리)에서 말이 끝났다. 받아쓰면 STT 가 호출어를 "자비다"처럼 잘못 써서
            # 질문으로 넘어갈 수 있다. 같은 마이크 줄에서 질문을 더 기다린다.
            speech = record_speech(blocks, started, timeout=timeout)
        utterance = question(speech)
        if utterance is None and speech is not None and is_wake_only(self.last_raw_text):
            # 늘여 말해서 감지 뒤에도 호출어가 이어졌다. 호출어만 받아써졌으면 한 번 더 듣는다.
            speech = record_speech(blocks, started, timeout=timeout)
            utterance = question(speech)
        if utterance is None:
            return None
        return Utterance(utterance.text, max(utterance.started_at, floor), utterance.ended_at)

    def _question(
        self, speech: Speech | None, min_letters: int, hints: Sequence[str]
    ) -> Utterance | None:
        """녹음한 발화를 받아쓰고 정리한다. 질문으로 쓸 만하지 않으면 None."""
        if speech is None:
            return None
        try:
            raw = self.transcribe(speech.audio, speech.started_at, speech.ended_at, hints)
        except OpenAIError as e:
            # 말이 없던 것과 같이 다룬다. 그래프는 호출어 대기로 돌아가면 된다.
            log.warning("받아쓰기 실패: %s", e)
            return None
        text = clean(raw.text)
        if not is_question(text, min_letters):
            return None
        return Utterance(text, raw.started_at, raw.ended_at)

    def transcribe(
        self, audio: np.ndarray, started_at: float, ended_at: float, hints: Sequence[str] = ()
    ) -> Utterance:
        """이미 녹음된 발화(16kHz 모노 int16)를 받아쓴다. 받아쓴 원문을 그대로 담아 돌려준다.

        hints 가 없으면 호출어를 힌트로 준다.
        """
        self.last_latency_ms = None
        sent_at = time.monotonic()
        result = self.client.audio.transcriptions.create(
            model=self.model,
            file=("speech.wav", _to_wav(audio)),
            language=self.language,
            # 힌트가 없으면 호출어 "자비서"가 "자비스"로 받아써졌다(스파이크 02).
            prompt=", ".join(hints) if hints else config.WAKE_WORD,
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
