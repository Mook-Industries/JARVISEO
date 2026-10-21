"""스피커 출력.  담당: 문태현

TTS 가 받는 대로 내는 PCM 조각을 순서대로 재생한다. 합성이 다 끝날 때까지 기다리지 않고
첫 조각이 오면 바로 소리를 낸다. 재생하는 동안은 마이크 게이트(``gate.py``)를 닫는다.

재생 중에 ``stop()`` 을 부르면(끼어들기) 남은 소리를 버리고 바로 멈춘다. 이때는 게이트도
유예 없이 바로 연다. 사용자가 "자비서" 뒤에 이어 말한 질문 앞부분이 막히지 않게 하려는 것이다.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Iterable
from contextlib import nullcontext
from typing import TYPE_CHECKING

from jarviseo.voice.gate import MIC_GATE, MicGate

if TYPE_CHECKING:
    import sounddevice as sd

__all__ = ["SAMPLE_RATE", "Speaker"]

# OpenAI TTS 의 response_format="pcm" 은 24kHz 16비트 모노(리틀엔디언)로 온다.
SAMPLE_RATE = 24000
# 스피커에는 20ms 씩 나눠 쓴다. write() 는 버퍼에 자리가 날 때까지 막히므로, 멈춤 신호는
# 이만큼마다 확인된다. TTS 조각(100ms)을 통째로 쓰면 멈추기까지 최대 0.1초가 더 걸린다.
PIECE_BYTES = SAMPLE_RATE * 2 // 50


class Speaker:
    """``Speaker().play(tts.synthesize(text))``. 다른 스레드에서 ``stop()`` 으로 끊는다."""

    def __init__(
        self,
        device: int | None = None,
        gate: MicGate | None = None,
        latency: str = "high",
        piece_bytes: int | None = PIECE_BYTES,
    ) -> None:
        """gate 는 테스트에서만 넘긴다. 기본은 STT 와 같이 쓰는 ``MIC_GATE``.

        latency 는 sounddevice 출력 버퍼 크기("high"/"low"). 조각이 네트워크로 띄엄띄엄 오는
        TTS 는 "high", 소리를 통째로 들고 있는 시작음·종료음은 "low" 로 첫 소리를 당긴다
        (노트북 스피커 MME 에서 출력 지연 182ms → 93ms, #104).

        piece_bytes 는 스피커에 한 번에 쓰는 크기다. None 이면 받은 조각을 통째로 한 번에 쓴다.
        write() 는 버퍼에 자리가 날 때까지 GIL 을 놓고 C 안에서 기다리는데, 나눠 쓰면 조각마다
        GIL 을 다시 받아야 해서 다른 파이썬 스레드가 바쁘면 버퍼가 비어 소리가 끊긴다.
        대신 stop() 은 통째로 쓴 조각이 다 나간 뒤에야 듣는다. 끊을 일이 없는 짧은 소리용이다.
        """
        self.device = device
        self.gate = gate or MIC_GATE
        self.latency = latency
        self.piece_bytes = piece_bytes
        self._stop = threading.Event()
        # 마지막 play() 가 stop() 으로 끊겼으면 True, 그때 소리를 멈춘 시각(monotonic).
        self.interrupted = False
        self.stopped_at: float | None = None
        # 마지막 play() 에서 첫 조각을 스피커에 쓰기 시작한 시각(monotonic). 소리는 출력 지연만큼
        # 뒤에 난다. 소리를 하나도 안 냈으면 None.
        self.started_at: float | None = None

    def stop(self) -> None:
        """재생 중이면 남은 소리를 버리고 멈추게 한다. 어느 스레드에서 불러도 된다.

        재생 중이 아니면 아무 일도 없다. 다음 play() 는 처음부터 끝까지 재생한다.
        첫 조각을 기다리는 동안 부르면 소리를 하나도 내지 않고 끝난다.
        """
        self._stop.set()

    def stop_requested(self) -> bool:
        """지금 play() 에 stop() 이 와 있으면 True. play() 를 시작할 때 지운다.

        조각을 넘기는 쪽(TTS)이 조각을 기다리는 동안 이것을 보고 그만 기다린다.
        """
        return self._stop.is_set()

    def play(self, chunks: Iterable[bytes], mute_mic: bool = True) -> None:
        """조각을 받는 순서대로 재생하고, 마지막 소리가 다 나올 때까지 기다린다.

        첫 조각이 오기 전에는 스피커도 게이트도 건드리지 않는다. chunks 가 도중에 예외를 내면
        그때까지 받은 소리는 마저 내고 예외를 그대로 올린다. stop() 으로 끊기면 chunks 를
        더 읽지 않고 돌아온다(닫는 것은 chunks 를 만든 쪽이 한다).

        조각을 기다리는 동안에도 바로 끊기려면 chunks 가 ``stop_requested()`` 를 보고 끝나야 한다
        (TTS 가 그렇게 넘긴다). stop() 뒤에 chunks 가 끝나면 끊긴 것으로 본다. 그렇지 않은
        chunks 는 다음 조각이 올 때까지 돌아오지 못한다.

        mute_mic=False 면 게이트를 닫지 않는다. 사용자가 말하는 도중이나 말끝에 내는 시작음·종료음
        용이다. 대신 소리가 마이크에 들어가는 구간을 게이트에 표시한다(``MicGate.sounding``).
        """
        self._stop.clear()
        self.interrupted, self.stopped_at, self.started_at = False, None, None
        chunks = iter(chunks)
        first = next(chunks, None)
        if self._stop.is_set():  # 첫 조각을 기다리는 동안 끊겼다. 게이트는 닫힌 적이 없다
            self._interrupted()
            return
        if first is None:
            return
        # 리눅스 CI 에는 PortAudio 가 없어서, 맨 위에서 import 하면 이 모듈을 읽기만 해도 죽는다.
        import sounddevice as sd

        # with 를 빠져나갈 때 stop() 이 버퍼에 남은 소리를 다 낼 때까지 기다린다(약 0.2초).
        # 게이트는 그다음에 풀어야 해서 바깥쪽에 둔다. 끊긴 경우는 abort() 로 버퍼를 버려서
        # 기다리지 않는다(그 뒤 stop() 은 이미 멈춘 스트림이라 바로 돌아온다).
        with (
            self.gate.playing() if mute_mic else nullcontext(),
            sd.RawOutputStream(
                samplerate=SAMPLE_RATE,
                device=self.device,
                channels=1,
                dtype="int16",
                latency=self.latency,
            ) as out,
        ):
            self.started_at = time.monotonic()
            # 마이크를 안 막는 소리(시작음·종료음)는 마이크에 들어가는 구간만 표시한다(VAD 가 무시)
            with nullcontext() if mute_mic else self.gate.sounding(out.latency):
                for chunk in itertools.chain([first], chunks):
                    if not self._write(out, chunk):
                        out.abort()
                        break
                else:
                    if self._stop.is_set():  # 다음 조각을 기다리다 끊겨서 chunks 가 끝났다
                        self._interrupted()
                        out.abort()
        if self.interrupted and mute_mic:
            self.gate.reopen(self.stopped_at)

    def _write(self, out: sd.RawOutputStream, chunk: bytes) -> bool:
        """chunk 를 나눠 쓴다. 도중에 멈춤 신호가 오면 멈춘 시각을 남기고 False."""
        step = self.piece_bytes or len(chunk)
        for i in range(0, len(chunk), step):
            if self._stop.is_set():
                self._interrupted()
                return False
            out.write(chunk[i : i + step])
        return True

    def _interrupted(self) -> None:
        self.interrupted, self.stopped_at = True, time.monotonic()
