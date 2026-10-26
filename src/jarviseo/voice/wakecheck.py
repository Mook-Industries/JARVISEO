"""호출어 2단계 확인.  담당: 문태현

호출어 모델(``wakeword.py``)이 깨어나면 감지 직전 ``config.WAKE_CHECK_SEC`` 초를 따로 받아써서,
받아쓴 글자에 호출어("자비서"·"자비스" 같은 표기, ``transcript.has_wake_word``)가 있을 때만
질문 처리를 이어 간다(#149). TV 를 틀어 두면 멀리서 작게 들리는 방송 말소리에 모델이 깨는데,
재학습으로는 그 구간이 줄지 않았다(docs/experiments.md 학습 기록 6회차).

질문 받아쓰기 결과로는 확인할 수 없다. "자비서" 하고 쉬었다가 질문하면 ``listen()`` 은 호출어
부분을 받아쓰지 않고 질문만 다시 듣는다(stt.py ``WAKE_ONLY_SEC``). 실제 91턴 중 원문에 호출어가
있던 턴은 21개뿐이었다. 그래서 호출어 구간을 따로 보내고, 질문을 듣는 동안 백그라운드에서 돌린다.
구간이 짧아 보통 질문 받아쓰기보다 먼저 끝나므로 진짜 호출은 늦어지지 않는다.

받아쓰기가 실패하거나 ``TIMEOUT_SEC`` 안에 안 끝나면 통과로 본다. 인터넷이 끊겨도 호출은 돼야 한다.

    check = WakeCheck(mic)
    verdict = check.start(wake.t)                     # 받아쓰기를 띄우고 바로 돌아온다
    stt.listen(since=wake.t, stop=verdict.rejected)   # 호출어가 아니면 듣다가 멈춘다
    if not verdict.passed():                          # 결과가 나올 때까지 기다린다
        ...                                           # 대답·되묻기 없이 호출어 대기로
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import numpy as np
from openai import OpenAI

from jarviseo import config
from jarviseo.voice.mic import MicStream
from jarviseo.voice.stt import to_wav
from jarviseo.voice.transcript import has_wake_word

__all__ = ["HINT", "TIMEOUT_SEC", "WakeCheck", "WakeVerdict"]

log = logging.getLogger(__name__)

# 확인을 시작하고 이만큼 지나도 받아쓰기가 안 끝나면 통과로 본다. 질문 받아쓰기 왕복이 보통
# 1~2.5초였고(스파이크 02), 이 구간은 그보다 짧다.
TIMEOUT_SEC = 3.0
# 받아쓰기에 "자비서"를 힌트로 줄지. 질문 받아쓰기는 힌트가 없으면 "자비스"로 썼는데(스파이크 02),
# 확인에서는 힌트가 TV 소리를 "자비서"로 지어내게 할 수 있어서 따로 쟀다(wakeword/verify.py).
HINT = True


class WakeVerdict:
    """확인 하나의 결과. ``WakeCheck.start()`` 가 돌려준다."""

    def __init__(self, future: Future[str | None], deadline: float) -> None:
        self._future = future
        self.deadline = deadline  # 이 시각(monotonic)까지 결과가 없으면 통과

    @property
    def text(self) -> str | None:
        """받아쓴 글자. 아직 안 끝났거나 받아쓰지 못했으면 None."""
        return self._future.result() if self._future.done() else None

    def rejected(self) -> bool:
        """받아쓰기가 끝났고 호출어가 없었으면 True. 기다리지 않는다(``listen(stop=...)`` 용)."""
        text = self.text
        return text is not None and not has_wake_word(text)

    def passed(self) -> bool:
        """결과가 나올 때까지 기다렸다가, 호출어가 있었거나 확인하지 못했으면 True."""
        try:
            self._future.result(timeout=max(0.0, self.deadline - time.monotonic()))
        except FutureTimeout:
            log.warning("호출어 확인이 %.1f초 안에 안 끝나 통과로 본다", TIMEOUT_SEC)
            return True
        return not self.rejected()


class WakeCheck:
    """감지 시각을 받으면 그 직전 구간을 받아써서 호출어였는지 확인한다."""

    def __init__(
        self,
        mic: MicStream | None = None,
        seconds: float = config.WAKE_CHECK_SEC,
        hint: bool = HINT,
        model: str = config.STT_MODEL,
        client: OpenAI | None = None,
    ) -> None:
        """mic 는 호출어 감지와 같이 쓰는 ``MicStream``. 녹음 파일로 잴 때(verify.py)는 없이
        만들고 ``transcribe()`` 만 부른다. client 는 테스트에서 가짜 API 를 끼울 때만 넘긴다.
        """
        self.mic = mic
        self.seconds = seconds
        self.hint = hint
        self.model = model
        # 다시 보내지 않는다. 다시 보낼 시간이면 통과로 보는 편이 낫다.
        self.client = client or OpenAI(
            api_key=config.OPENAI_API_KEY or None, timeout=TIMEOUT_SEC, max_retries=0
        )
        # 질문 받아쓰기·대답과 따로 돈다. 끼어들기로 확인이 겹칠 수 있어 둘까지 같이 돌린다.
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="wakecheck")

    def start(self, t: float) -> WakeVerdict:
        """감지 시각 t(``on_detect`` 의 t) 직전 구간을 받아쓰기 시작한다. 바로 돌아온다."""
        if self.mic is None:
            raise ValueError("start() 는 mic 를 넘겨 만든 WakeCheck 에서만 부른다")
        audio = self.mic.clip(t - self.seconds, t)
        return WakeVerdict(self._pool.submit(self.check, audio), time.monotonic() + TIMEOUT_SEC)

    def transcribe(self, audio: np.ndarray, hint: bool | None = None) -> str:
        """구간 하나(16kHz 모노 int16)를 받아쓴 원문. 실패하면 OpenAIError 가 난다."""
        hint = self.hint if hint is None else hint
        result = self.client.audio.transcriptions.create(
            model=self.model,
            file=("wake.wav", to_wav(audio)),
            language="ko",
            **({"prompt": config.WAKE_WORD} if hint else {}),
        )
        return result.text

    def check(self, audio: np.ndarray) -> str | None:
        """구간 하나를 받아쓴 원문. 구간이 비었거나 받아쓰지 못했으면 None(= 통과로 본다).

        ``start()`` 가 스레드에서 부르고, 브라우저 호출어(api/main.py ``/api/wake-check``)는
        바로 부른다.
        """
        if len(audio) == 0:
            return None  # 마이크가 그 구간을 들고 있지 않다. 확인하지 못한 것으로 본다
        sent_at = time.monotonic()
        try:
            text = self.transcribe(audio)
        except Exception as e:
            # 어떤 오류든 통과로 본다. 확인 때문에 진짜 호출을 놓치면 안 된다.
            log.warning("호출어 확인 받아쓰기 실패: %s", e)
            return None
        ms = (time.monotonic() - sent_at) * 1000
        verdict = "통과" if has_wake_word(text) else "거절"
        log.info("호출어 확인 %s %r (%.0fms)", verdict, text, ms)
        return text
