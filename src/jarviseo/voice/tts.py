"""음성 합성 (TTS).  담당: 문태현

답변 문장을 OpenAI TTS 로 합성한다. 모델 이름은 ``config.TTS_MODEL`` 하나만 본다.
``synthesize()`` 가 PCM 조각을 받는 대로 내고, ``speak()`` 는 그 조각을 받는 대로 재생한다.

답이 늦어 "잠깐만요"를 읽는 동안 답이 오면, ``prefetch()`` 로 대답 합성을 바로 시작해 둔다.
필러가 끝나고 speak() 를 부를 때는 받아 둔 조각부터 읽어서, 첫 조각 왕복(약 0.9초)을
필러 뒤에 다시 기다리지 않는다.

speak() 는 미리 받지 않았을 때도 합성을 다른 스레드에서 받는다. 서버가 느리면 첫 조각까지
몇 초가 걸리는데(#105 실측 11~16초), 그동안 "자비서"로 끼어들어도 바로 돌아오게 하려는 것이다.

합성이 멈추면(읽기 타임아웃) ``config.TTS_FALLBACK_MODEL`` 로 한 번 더 요청해 처음부터 읽는다.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Generator, Iterator

import httpx2
import numpy as np
from openai import APITimeoutError, OpenAI, OpenAIError

from jarviseo import config
from jarviseo.voice.speaker import SAMPLE_RATE, Speaker
from jarviseo.voice.tts_text import normalize

__all__ = ["SAMPLE_RATE", "TextToSpeech", "trim_silence"]

log = logging.getLogger(__name__)

CHUNK_BYTES = SAMPLE_RATE * 2 // 10  # 100ms 씩 받는다
# user_setting.tts_speed → OpenAI speed 값. instructions 로 "천천히"를 부탁하면 같은 문장도
# 길이가 ±0.6초씩 흔들려 효과가 안 보였고, speed 는 0.8 → 1.31배, 1.25 → 0.79배로 일정했다.
SPEEDS = {"SLOW": 0.85, "NORMAL": 1.0, "FAST": 1.2}
# 한 번 기다리는 최대 시간(초)과 다시 보내는 횟수.
# 받는 쪽(read)은 따로 짧게 건다. 첫 조각은 보통 0.6~1.5초에 오는데(D12 21턴), 서버가 느릴 때
# 10초를 다 기다리고 다시 보내서 첫 소리까지 18.4초가 걸린 적이 있다(D5). 응답이 없는 서버로
# 재 보면 10초는 20.6초 만에, read 4초는 8.5초 만에 포기한다. 3초 걸리는 응답은 둘 다 받는다.
# 응답을 받기 시작한 뒤 멈춘 것은 openai 가 다시 보내지 않아서 speak() 가 폴백 모델로 한 번 더
# 요청한다.
TIMEOUT_SEC, READ_TIMEOUT_SEC, RETRIES = 10.0, 4.0, 1
TIMEOUT = httpx2.Timeout(TIMEOUT_SEC, read=READ_TIMEOUT_SEC)
# 조각을 기다리는 동안 이만큼마다 끼어들기(stop)를 확인한다. 스피커에 쓰는 단위(20ms)와 같다.
WAIT_POLL_SEC = 0.02
# 미리 만든 소리의 앞뒤 무음을 자를 때, 샘플 크기(16비트 절댓값)가 이 값 이하면 무음으로 본다.
# "잠깐만요." 합성은 2.25~2.65초인데 말소리는 0.11~1.21초뿐이었다. 뒤 무음은 최대 7,
# 말소리는 최대 1만쯤이다(10/14). 자르지 않으면 대답이 그 무음이 끝날 때까지 1초 넘게 기다린다.
SILENCE_LEVEL = 100
# 자른 뒤 앞뒤에 남기는 여유(초). 첫소리·끝소리가 뚝 끊겨 들리지 않게 한다.
TRIM_PAD_SEC = 0.05


class TextToSpeech:
    """글자를 소리로 바꿔 재생한다."""

    def __init__(
        self,
        model: str = config.TTS_MODEL,
        voice: str = "coral",
        speed: str = "NORMAL",
        device: int | None = None,
        client: OpenAI | None = None,
        fallback_model: str = config.TTS_FALLBACK_MODEL,
    ) -> None:
        """voice · speed 는 ``user_setting`` 의 tts_voice · tts_speed (SLOW/NORMAL/FAST).
        device 는 sounddevice 출력 장치 번호다. None 이면 기본 스피커.
        fallback_model 은 합성이 멈춰서 speak() 가 한 번 더 요청할 때 쓰는 모델이다.

        client 는 테스트에서 가짜 API 를 끼울 때만 넘긴다.
        """
        self.model = model
        self.fallback_model = fallback_model
        self.voice = voice
        self.speed = speed
        self.speaker = Speaker(device)
        self.client = client or OpenAI(
            api_key=config.OPENAI_API_KEY or None, timeout=TIMEOUT, max_retries=RETRIES
        )
        # 마지막 speak() 에서 첫 조각이 나오기까지 기다린 시간(ms). 아직 못 받았으면 None.
        # synthesize() 만 불렀으면 그 합성의 첫 조각 왕복(TTFB)이다.
        # 그래프가 latency_ms["tts"] 로 넘기면 log_turn 이 turn_voice.tts_ms 에 적는다.
        self.last_latency_ms: float | None = None
        # 마지막 speak() 이 stop() 으로 중간에 끊겼으면 True (turn_voice.is_interrupted).
        self.last_interrupted = False
        # prefetch() 로 미리 합성하기 시작한 것. 답을 받는 스레드가 넣고 speak() 가 꺼낸다.
        self._prefetched: _Prefetch | None = None
        self._lock = threading.Lock()

    def speak(self, text: str) -> float | None:
        """text 를 합성하면서 받는 대로 읽어 준다. 재생이 끝나거나 stop() 으로 끊길 때까지 기다린다.

        speak() 를 부른 뒤 첫 오디오 조각이 나오기까지 기다린 시간(ms)을 돌려준다. 이 값이
        ``turn_voice.tts_ms`` 다. 보통은 합성 첫 조각 왕복이고, 같은 text 를 prefetch() 해
        두었으면 미리 받은 만큼 짧다(다 받아 두었으면 거의 0).
        합성이 멈추면 폴백 모델로 한 번 더 요청해 처음부터 읽는다. 그것도 실패하면 그때까지
        받은 소리만 내고 멈춘다. 소리를 하나도 못 냈으면 None.
        끝까지 읽었는지는 ``last_interrupted`` 로 본다.
        """
        self.last_interrupted = False
        self.last_latency_ms = None
        asked_at = time.monotonic()
        with self._lock:
            prefetched, self._prefetched = self._prefetched, None
        if prefetched is not None and prefetched.text != text:
            prefetched.cancel()  # 다른 글자(버린 답)를 미리 받던 것이다
            prefetched = None

        def first() -> None:
            self.last_latency_ms = (time.monotonic() - asked_at) * 1000

        source = prefetched if prefetched is not None else _Prefetch(text, self._stream(text))
        chunks = _on_first(self._receive(text, source), first)
        try:
            self.speaker.play(chunks)
        except (OpenAIError, httpx2.HTTPError) as e:
            # 요청 자체가 실패했거나(openai 가 이미 RETRIES 만큼 다시 보낸 뒤), 폴백 요청도 멈췄다.
            # 받은 소리까지만 내고, 음성 루프는 멈추지 않는다.
            log.warning("합성 실패, 재생을 멈춘다: %s", e)
        finally:
            # 끊겨서 덜 읽은 합성을 그만 받게 한다. 그냥 두면 응답 스트림이 끝까지 열려 있다.
            chunks.close()
        self.last_interrupted = self.speaker.interrupted
        return self.last_latency_ms

    def _receive(self, text: str, source: _Prefetch) -> Generator[bytes]:
        """source 의 조각을 낸다. 합성이 멈췄으면 폴백 모델로 한 번 더 요청해 처음부터 이어 낸다.

        다시 요청하는 것까지 play() 한 번 안에서 한다. play() 를 다시 부르면 그사이 온
        stop() 표시가 지워져서, 끼어든 뒤에 앞 대답을 처음부터 읽었다.
        stop() 이 와 있으면 실패해도 다시 요청하지 않고 그냥 끝낸다(Speaker 가 끊김으로 본다).
        """
        stopped = self.speaker.stop_requested
        received = False
        try:
            for chunk in source.chunks(stopped):
                received = True
                yield chunk
        except (OpenAIError, httpx2.HTTPError) as e:
            if stopped():
                return
            # 응답을 받기 시작한 뒤 서버가 멈추면 openai 가 감싸지 않은 httpx2.ReadTimeout 이
            # 그대로 올라오고, openai 도 다시 보내 주지 않는다. 그래서 여기서 새로 한 번 더
            # 요청한다. 멈춤이 적은 폴백 모델로 보내고, 이미 읽던 중이었어도 처음부터 읽는다.
            # 앞부분("네")이 한 번 더 나오지만, 받은 만큼만 읽고 끝내면 대답을 못 듣는다(#130).
            # 응답 머리도 못 받고 멈추면 openai 가 같은 모델로 RETRIES 만큼 다시 보낸 뒤
            # APITimeoutError("Request timed out.")를 올린다. 이것도 서버가 멈춘 것이라 폴백
            # 모델로 보낸다(#130 실측 speak() 10번 중 1번, 8.8초 동안 소리 없이 끝났다).
            if not isinstance(e, (httpx2.HTTPError, APITimeoutError)):
                raise
            again = "처음부터 다시" if received else "한 번 더"
            log.warning("합성이 멈춰서 %s 로 %s 요청한다: %s", self.fallback_model, again, e)
            yield from _Prefetch(text, self._stream(text, self.fallback_model)).chunks(stopped)

    def prefetch(self, text: str) -> None:
        """text 합성을 다른 스레드에서 지금 시작해 둔다. 바로 돌아온다.

        곧 같은 text 로 speak() 를 부르면 새로 요청하지 않고 받아 둔 조각부터 읽는다.
        ResponseCollector 가 답을 받자마자 부른다. 다른 text 로 speak() 하거나 다시
        prefetch() 하면 받던 것은 버린다(버린 답의 합성).
        """
        new = _Prefetch(text, self._stream(text))
        with self._lock:
            old, self._prefetched = self._prefetched, new
        if old is not None:
            old.cancel()

    def prepare(self, text: str) -> bytes:
        """text 를 미리 끝까지 합성해 PCM 을 통째로 돌려준다. 실패하면 빈 바이트.

        자주 읽는 짧은 말("잠깐만요")을 시작할 때 한 번 만들어 둔다. 그때그때 합성하면
        첫 소리까지 약 1초가 걸려서, 기다리는 사람에게 너무 늦게 나간다.
        앞뒤 무음은 잘라 둔다(``trim_silence``). play() 는 무음까지 다 읽어야 돌아온다.
        """
        try:
            return trim_silence(b"".join(self.synthesize(text)))
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
        self.last_latency_ms = None
        asked_at = time.monotonic()

        def first() -> None:
            self.last_latency_ms = (time.monotonic() - asked_at) * 1000

        return _on_first(self._stream(text), first)

    def _stream(self, text: str, model: str | None = None) -> Generator[bytes]:
        """합성 요청 하나를 보내고 PCM 조각을 받는 대로 낸다. 닫으면 응답 스트림도 닫는다.

        model 을 안 주면 self.model 로 보낸다.
        객체의 상태(last_latency_ms 등)는 건드리지 않는다. prefetch() 가 다른 스레드에서 쓴다.
        """
        speed = SPEEDS.get(self.speed.upper(), 1.0)
        with self.client.audio.speech.with_streaming_response.create(
            model=model or self.model,
            voice=self.voice,
            input=normalize(text),
            # NORMAL 은 speed 를 아예 안 보낸다. 속도를 바꾸면 첫 소리가 늦어졌다
            # (첫 조각 중앙값: 안 보냄 약 1.0초, 1.2 는 약 1.8초).
            **({} if speed == 1.0 else {"speed": speed}),
            response_format="pcm",
        ) as response:
            yield from response.iter_bytes(CHUNK_BYTES)


def _on_first(chunks: Generator[bytes], first: Callable[[], None]) -> Generator[bytes]:
    """chunks 를 그대로 내면서 첫 조각이 나올 때 first() 를 부른다. 닫으면 chunks 도 닫는다."""
    try:
        for i, chunk in enumerate(chunks):
            if i == 0:
                first()
            yield chunk
    finally:
        chunks.close()


class _Prefetch:
    """합성 조각을 다른 스레드에서 받아 쌓아 두고, speak() 가 chunks() 로 꺼내 읽는다.

    prefetch() 는 미리 만들어 두고, 미리 받지 않은 speak() 는 그 자리에서 만든다.
    받는 스레드가 응답 스트림을 혼자 읽고 닫아서, 읽는 쪽은 조각을 기다리다 언제든 그만둘 수 있다.
    """

    def __init__(self, text: str, stream: Generator[bytes]) -> None:
        self.text = text
        self._stream = stream
        # 조각(bytes), 받다가 난 예외, 끝 표시(None) 가 차례로 들어간다
        self._queue: queue.SimpleQueue[bytes | Exception | None] = queue.SimpleQueue()
        self._cancelled = threading.Event()
        threading.Thread(target=self._receive, name="tts-prefetch", daemon=True).start()

    def _receive(self) -> None:
        try:
            for chunk in self._stream:
                if self._cancelled.is_set():
                    break
                self._queue.put(chunk)
        except Exception as e:  # speak() 쪽에서 다시 올려 평소 실패처럼 다룬다
            self._queue.put(e)
        finally:
            self._stream.close()  # 응답 스트림을 닫는다
            self._queue.put(None)

    def chunks(self, stopped: Callable[[], bool]) -> Generator[bytes]:
        """받아 둔 조각부터 차례로 낸다. 아직 안 온 조각은 올 때까지 기다린다.

        기다리는 동안 stopped() 가 True 가 되면(끼어들기) 그만 기다리고 끝난다. 서버가 느려도
        조각이 올 때까지 막혀 있지 않는다. 받다가 예외가 났으면 그 자리에서 올린다.
        닫으면 남은 합성을 그만 받는다.
        """
        try:
            while True:
                try:
                    item = self._queue.get(timeout=WAIT_POLL_SEC)
                except queue.Empty:
                    if stopped():
                        return
                    continue
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            self.cancel()

    def cancel(self) -> None:
        """남은 조각을 그만 받게 한다. 받는 중인 조각 하나가 끝나면 스트림을 닫는다."""
        self._cancelled.set()


def trim_silence(pcm: bytes) -> bytes:
    """PCM(16비트 모노)의 앞뒤 무음을 자르고 TRIM_PAD_SEC 만큼만 남긴다. 전부 무음이면 빈 바이트."""
    samples = np.frombuffer(pcm, np.int16)
    loud = np.flatnonzero(np.abs(samples.astype(np.int32)) > SILENCE_LEVEL)
    if loud.size == 0:
        return b""
    pad = round(TRIM_PAD_SEC * SAMPLE_RATE)
    start, end = max(loud[0] - pad, 0), min(loud[-1] + 1 + pad, len(samples))
    return samples[start:end].tobytes()
