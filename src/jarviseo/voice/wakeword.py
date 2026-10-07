"""호출어 감지.  담당: 문태현

마이크(``MicStream``)를 계속 읽어 openWakeWord 모델에 80ms 씩 넣고, 프레임마다 0~1 점수를 낸다.
상시 듣는 것은 이것뿐이고 로컬에서만 돈다. 클라우드로 나가는 것은 호출어 뒤의 발화 한 번이다.

모델은 ``config.WAKEWORD_MODEL`` 로 바꿔 끼운다. 기성 모델 이름(hey_jarvis)이나 .onnx 경로를 받는다.
기성 hey_jarvis 는 "자비서"를 못 잡는다(docs/experiments.md baseline, FRR 100%).
커스텀 모델이 나오기 전까지는 "Hey Jarvis" 로 감지기가 도는지만 확인한다.

책임 지표는 FAR(안 불렀는데 깨어남)와 FRR(불렀는데 안 깨어남)이다.
``scripts/wakeword/evaluate.py`` 로 잰다.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from jarviseo import config
from jarviseo.voice.mic import SAMPLE_RATE, MicStream

__all__ = ["FRAME", "WakeWordDetector", "load_scorer"]

FRAME = 1280  # openWakeWord 는 80ms(16kHz 1280샘플)씩 받는다. 마이크 블록은 30ms 라 모아서 넣는다


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
    """마이크를 계속 듣고 80ms 마다 호출어 점수를 낸다."""

    def __init__(
        self,
        mic: MicStream,
        model: str = config.WAKEWORD_MODEL,
        scorer: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        """mic 는 STT 와 같이 쓰는 ``MicStream``. scorer 는 테스트에서 가짜 점수를 낼 때 넘긴다."""
        self.mic = mic
        self._score = scorer or load_scorer(model)
        # 아직 80ms 가 안 돼 모아 두는 샘플과, 그 첫 샘플의 시각
        self._pending = np.empty(0, np.int16)
        self._pending_t = 0.0
        # 마지막 프레임의 점수. 화면에 띄우거나 임계값을 고를 때 본다.
        self.last_score = 0.0

    def feed(self, block: np.ndarray, t: float) -> None:
        """마이크 블록 하나(30ms)와 그 첫 샘플의 시각을 넣는다. 80ms 가 모일 때마다 점수를 낸다."""
        if not len(self._pending):
            self._pending_t = t
        self._pending = np.concatenate([self._pending, block])
        while len(self._pending) >= FRAME:
            frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
            self._pending_t += FRAME / SAMPLE_RATE  # 이 프레임이 끝난 시각 = 다음 샘플의 시각
            self.last_score = self._score(frame)
