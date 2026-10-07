"""호출어 감지.  담당: 문태현

마이크(``MicStream``)를 계속 읽어 openWakeWord 모델에 80ms 씩 넣고, 점수가 임계값을 넘으면
``on_detect(시각, 점수)`` 를 부른다. 별도 스레드에서 돈다.
상시 듣는 것은 이것뿐이고 로컬에서만 돈다. 클라우드로 나가는 것은 호출어 뒤의 발화 한 번이다.

재생 중 게이트는 두 단이다(gate.py). TTS 가 재생 중이거나 끝나고 0.3초 안이면
STT 는 그 소리를 아예 못 듣고(1단), 호출어는 계속 듣되 임계값을 0.8 로 올린다(2단).
자기 목소리에 깨지 않으면서도 재생 중에 "자비서"로 끼어들 수 있게 하려는 것이다.

모델은 ``config.WAKEWORD_MODEL`` 로 바꿔 끼운다. 기성 모델 이름(hey_jarvis)이나 .onnx 경로를 받는다.
기성 hey_jarvis 는 "자비서"를 못 잡는다(docs/experiments.md baseline, FRR 100%).
커스텀 모델이 나오기 전까지는 "Hey Jarvis" 로 감지기가 도는지만 확인한다.

책임 지표는 FAR(안 불렀는데 깨어남)와 FRR(불렀는데 안 깨어남)이다.
``scripts/wakeword/evaluate.py`` 로 잰다.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np

from jarviseo import config
from jarviseo.voice.gate import MIC_GATE, MicGate
from jarviseo.voice.mic import SAMPLE_RATE, MicStream

__all__ = ["FRAME", "WakeWordDetector", "load_scorer"]

log = logging.getLogger(__name__)

FRAME = 1280  # openWakeWord 는 80ms(16kHz 1280샘플)씩 받는다. 마이크 블록은 30ms 라 모아서 넣는다
# 한 번 깨어나면 이만큼은 다시 깨지 않는다. 점수는 호출어가 끝난 뒤에도 몇 프레임 높게 남는다.
# evaluate.py 가 시간당 오탐을 셀 때도 같은 값을 쓴다.
REFRACTORY_SEC = 2.0


def load_scorer(model: str) -> Callable[[np.ndarray], float]:
    """openWakeWord 모델을 열어, 80ms 프레임 하나를 받아 0~1 점수를 내는 함수를 돌려준다."""
    # 리눅스(CI·Docker)에는 openwakeword 가 설치되지 않는다(requirements.txt). 쓸 때만 읽는다.
    from openwakeword.model import Model
    from openwakeword.utils import download_models

    if not model.endswith(".onnx"):
        download_models([model])  # 기성 모델은 처음 한 번만 받는다(약 1MB)
    oww = Model(wakeword_models=[model], inference_framework="onnx")
    (name,) = oww.models
    return lambda frame: float(oww.predict(frame)[name])


class WakeWordDetector:
    """``detector.start()`` 하면 마이크를 계속 듣다가 호출어가 들릴 때마다 on_detect 를 부른다."""

    def __init__(
        self,
        mic: MicStream,
        on_detect: Callable[[float, float], None] | None = None,
        model: str = config.WAKEWORD_MODEL,
        threshold: float = config.WAKEWORD_THRESHOLD,
        playback_threshold: float = config.WAKEWORD_PLAYBACK_THRESHOLD,
        gate: MicGate | None = None,
        scorer: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        """mic 는 STT 와 같이 쓰는 ``MicStream``. gate·scorer 는 테스트에서만 넘긴다.

        gate 가 닫혀 있던 동안(재생 중과 끝난 뒤 유예) 들어온 소리에는 playback_threshold 를 댄다.

        on_detect(t, score) 의 t 는 호출어를 잡은 프레임이 끝난 시각(monotonic)이다.
        ``listen(since=t)`` 로 넘기면 호출어 바로 뒤에 이어 말한 질문부터 받아쓴다.
        감지 스레드에서 불리므로, 오래 걸리는 일은 다른 스레드로 넘긴다.
        """
        self.mic = mic
        self.on_detect = on_detect
        self.threshold = threshold
        self.playback_threshold = playback_threshold
        self.gate = gate or MIC_GATE
        self._score = scorer or load_scorer(model)
        self._quiet_until = float("-inf")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # 아직 80ms 가 안 돼 모아 두는 샘플과, 그 첫 샘플의 시각
        self._pending = np.empty(0, np.int16)
        self._pending_t = 0.0
        # 마지막 프레임의 점수. 화면에 띄우거나 임계값을 고를 때 본다.
        self.last_score = 0.0

    def feed(self, block: np.ndarray, t: float) -> None:
        """마이크 블록 하나(30ms)와 그 첫 샘플의 시각을 넣는다. 80ms 가 모일 때마다 점수를 낸다."""
        # 모아 둔 샘플의 첫 시각을 이 블록 시각에서 거꾸로 잡는다. 앞 블록 시각에 길이를 더해 가면
        # 마이크가 샘플을 놓쳐 시각을 다시 맞췄을 때(mic.py) 그만큼 밀린다.
        self._pending_t = t - len(self._pending) / SAMPLE_RATE
        self._pending = np.concatenate([self._pending, block])
        while len(self._pending) >= FRAME:
            frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
            # 게이트에는 소리가 마이크에 들어온 시각으로 묻는다. 감지가 밀려 돌아도 구간이 맞고,
            # 재생 시작·끝 콜백(on_playback)으로 바꿀 때처럼 끝난 뒤 유예를 따로 셀 필요가 없다.
            playing = not self.gate.is_open(self._pending_t)
            threshold = self.playback_threshold if playing else self.threshold
            self._pending_t += FRAME / SAMPLE_RATE  # 이 프레임이 끝난 시각 = 다음 샘플의 시각
            self.last_score = self._score(frame)
            if self.last_score >= threshold and self._pending_t >= self._quiet_until:
                self._quiet_until = self._pending_t + REFRACTORY_SEC
                self._detected(self._pending_t, self.last_score)

    def _detected(self, t: float, score: float) -> None:
        log.info("호출어 감지 %.2f (점수 %.3f)", t, score)
        if self.on_detect is None:
            return
        try:
            self.on_detect(t, score)
        except Exception:
            # 콜백이 터져도 감지는 계속한다. 상시 듣는 스레드가 죽으면 다시 부를 방법이 없다.
            log.exception("on_detect 에서 오류")

    def start(self) -> None:
        """감지 스레드를 띄운다. mic 가 열려 있어야 한다."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="wakeword", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """감지를 멈추고 스레드가 끝날 때까지 기다린다(마이크 블록 하나, 30ms 안쪽)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        for block, t in self.mic.blocks():
            if self._stop.is_set():
                return
            self.feed(block, t)
