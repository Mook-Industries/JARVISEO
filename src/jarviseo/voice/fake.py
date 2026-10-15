"""키 없이 쓰는 가짜 STT · TTS.  담당: 문태현

실제 클래스와 메서드 모양이 같다(``tests/test_voice_fake.py`` 가 확인한다).
그래프는 이걸로 개발하다가 실제 클래스로 바꿔 끼우면 된다. CI 에서도 API 를 부르지 않는다.

    from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech

    stt = FakeSpeechToText(["자비서, 저거 뭐야?"])
    picked = []
    utterance = stt.listen(on_speech_start=lambda t: picked.append(buffer.pick_best(around=t)))

사진 폴더(``ImageFolderSource``)로 돌릴 때는 사진 시각이 0초, 0.033초…라서 말 시작 시각도
그 시각으로 정해 줘야 그 근처 사진이 골린다.

    stt = FakeSpeechToText(["자비서, 저거 뭐야?"], starts=[0.5])  # 0.5초에 말을 시작했다
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

from jarviseo.types import Utterance
from jarviseo.voice.transcript import MIN_LETTERS, clean, is_question

if TYPE_CHECKING:
    import numpy as np

__all__ = ["FakeSpeechToText", "FakeTextToSpeech"]


class FakeSpeechToText:
    """정해 둔 문장을 차례로 돌려준다. 다 쓰면 말이 없는 것(None)으로 친다.

    listen() 은 실제처럼 호출어·필러를 뺀 질문을 주고, 비었거나 너무 짧으면 None 을 준다.
    원문은 last_raw_text 에 남는다.

    starts 는 문장마다 말을 시작한 시각(초)이다. 차례로 쓰고, 없거나 다 쓰면 지금 시각
    (``time.monotonic()``)이다. 말은 시작하고 1초 뒤에 끝난 것으로 친다.
    """

    def __init__(
        self, texts: Iterable[str] = ("자비서, 저거 뭐야?",), starts: Iterable[float] = ()
    ) -> None:
        self._texts = list(texts)
        self._starts = list(starts)
        self.last_latency_ms: float | None = None
        self.last_raw_text: str | None = None

    def listen(
        self,
        on_speech_start: Callable[[float], None] | None = None,
        timeout: float = 5.0,
        since: float | None = None,
        min_letters: int = MIN_LETTERS,
    ) -> Utterance | None:
        self.last_raw_text = self.last_latency_ms = None
        if not self._texts:
            return None
        self.last_raw_text, self.last_latency_ms = self._texts.pop(0), 0.0
        started_at = self._starts.pop(0) if self._starts else time.monotonic()
        if on_speech_start is not None:
            on_speech_start(started_at)
        text = clean(self.last_raw_text)
        if not is_question(text, min_letters):
            return None
        return Utterance(text, started_at, started_at + 1.0)

    def transcribe(self, audio: np.ndarray, started_at: float, ended_at: float) -> Utterance:
        self.last_raw_text, self.last_latency_ms = self._texts.pop(0) if self._texts else "", 0.0
        return Utterance(self.last_raw_text, started_at, ended_at)


class FakeTextToSpeech:
    """소리를 내지 않고, 받은 문장을 ``spoken`` 에 모아 둔다.

    prepare() 는 글자를 그대로 바이트로 주고, play() 는 그걸 글자로 되돌려 ``spoken`` 에 넣는다.
    그래서 미리 만든 "잠깐만요"를 읽은 것도 ``spoken`` 에서 보인다.
    """

    def __init__(self) -> None:
        self.spoken: list[str] = []
        # prefetch() 로 미리 합성하라고 받은 문장
        self.prefetched: list[str] = []
        self.last_latency_ms: float | None = None
        self.last_interrupted = False

    def speak(self, text: str) -> float | None:
        self.spoken.append(text)
        self.last_latency_ms, self.last_interrupted = 0.0, False
        return 0.0

    def prefetch(self, text: str) -> None:
        self.prefetched.append(text)

    def prepare(self, text: str) -> bytes:
        return text.encode()

    def play(self, pcm: bytes) -> None:
        self.spoken.append(pcm.decode())
        self.last_interrupted = False

    def stop(self) -> None:
        """바로 끝나는 speak() 이라 끊을 것이 없다."""

    def synthesize(self, text: str) -> Iterator[bytes]:
        self.last_latency_ms = 0.0
        yield bytes(4800)  # 무음 100ms (24kHz 16비트)
