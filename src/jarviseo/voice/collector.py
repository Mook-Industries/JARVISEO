"""Response Collector — 답을 기다리는 동안의 처리.  담당: 문태현

VLM 답이 ``config.FILLER_AFTER_SEC``(1.5초) 안에 안 오면 "잠깐만요"를 먼저 읽는다.
안경에는 화면이 없어서, 아무 소리가 없으면 사용자는 못 들은 줄 알고 다시 부른다.

기다리는 동안 사용자가 다시 "자비서"라고 부르면 지금 기다리던 답은 버린다.
새 질문을 하려는 것이라, 늦게 온 옛 답을 읽으면 엉뚱한 대답이 된다.

답이 오면 그 자리에서 ``tts.prefetch(답)`` 으로 합성을 시작해 둔다. "잠깐만요"를 읽는 동안
대답 첫 조각을 받아 두면, 필러가 끝나고 speak() 가 바로 대답을 읽는다.

    collector = ResponseCollector(tts, called_since=barge.called_since)
    answer = collector.collect(lambda: vlm.ask(frame, question))
    collector.filler_sent   # "잠깐만요"를 읽었는지 (turn_voice.is_filler_sent)
    collector.dropped       # 다시 불러서 답을 버렸는지 (그러면 answer 는 None)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from jarviseo import config

if TYPE_CHECKING:
    from jarviseo.voice.fake import FakeTextToSpeech
    from jarviseo.voice.tts import TextToSpeech

__all__ = ["FILLER", "ResponseCollector"]

FILLER = "잠깐만요."
# 답이 왔는지, 다시 불렀는지 이만큼마다 본다.
POLL_SEC = 0.05


class ResponseCollector:
    """답을 다른 스레드에서 받아 오고, 늦으면 "잠깐만요"를 먼저 읽는다."""

    def __init__(
        self,
        tts: TextToSpeech | FakeTextToSpeech,
        called_since: Callable[[float], bool] | None = None,
        after_sec: float = config.FILLER_AFTER_SEC,
        filler: str = FILLER,
    ) -> None:
        """called_since(t) 는 t 뒤에 다시 불렀으면 True 를 주는 함수다(``BargeIn.called_since``).

        filler 는 여기서 한 번 미리 합성해 둔다. 합성이 실패하면 필러 없이 기다리기만 한다.
        """
        self.tts = tts
        self.called_since = called_since or (lambda t: False)
        self.after_sec = after_sec
        self._filler_text = filler
        self._filler = tts.prepare(filler)
        self._filler_gen = 0  # refresh_filler() 를 부를 때마다 하나씩 는다
        # 버린 답의 요청은 끝까지 돌게 둔다. 일꾼이 하나면 그게 끝날 때까지 다음 질문이 막힌다.
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="answer")
        # 마지막 collect() 의 결과
        self.filler_sent = False
        self.dropped = False
        # collect() 를 부를 때마다 하나씩 는다. 버린 옛 답이 다음 질문 중에 와도 미리 합성하지 않게
        self._turn = 0

    def collect(self, ask: Callable[[], str | None]) -> str | None:
        """ask() 를 다른 스레드에서 부르고 그 답을 돌려준다.

        기다리는 동안 다시 불렸으면 답을 기다리지 않고 None 을 돌려주고 ``dropped`` 가 True 다.
        """
        self.filler_sent = self.dropped = False
        started = time.monotonic()
        self._turn += 1
        turn = self._turn

        def answer() -> str | None:
            text = ask()
            # 답이 오자마자 합성을 시작한다. 그사이 다시 불러 버린 답이면 하지 않는다.
            if text and turn == self._turn and not self.dropped:
                self.tts.prefetch(text)
            return text

        future = self._pool.submit(answer)
        while not future.done():
            if self.called_since(started):
                self.dropped = True
                return None
            waited = time.monotonic() - started
            if not self.filler_sent and self._filler and waited >= self.after_sec:
                # 읽는 동안에도 답은 다른 스레드에서 계속 온다. 다시 부르면 여기서 바로 끊긴다.
                self.tts.play(self._filler)
                self.filler_sent = True
                continue
            time.sleep(POLL_SEC)
        if self.called_since(started):
            self.dropped = True
            return None
        return future.result()

    def refresh_filler(self) -> None:
        """목소리 · 속도를 바꿨을 때 부른다(``TextToSpeech.configure``). 바로 돌아온다.

        "잠깐만요"를 다른 스레드에서 새 목소리로 다시 만든다. 다 될 때까지는 앞의 것을 읽고,
        잇달아 바꾸면 마지막에 바꾼 것만 남긴다. 새로 만들기를 실패하면 앞의 것을 그대로 둔다.
        """
        self._filler_gen += 1
        gen = self._filler_gen

        def prepare() -> None:
            pcm = self.tts.prepare(self._filler_text)
            if pcm and gen == self._filler_gen:
                self._filler = pcm

        threading.Thread(target=prepare, name="filler", daemon=True).start()

    def close(self) -> None:
        """받아 오던 답을 기다려 마치고 닫는다. 프로그램을 끝낼 때 부른다."""
        self._pool.shutdown(wait=True)
