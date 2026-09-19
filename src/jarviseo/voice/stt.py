"""음성 인식 (STT).  담당: 문태현

로컬에서 돌린다. 상시 동작하는 것은 클라우드에 올리지 않는다.

started_at 이 핵심이다
----------------------
알려진 함정: 발화가 끝난 뒤에 프레임을 캡처하면 이미 고개가 돌아가서
엉뚱한 장면이 잡힌다. 그래서 VAD 가 발화 '시작' 시각을 반환해야 하고,
그 시각으로 FrameBuffer 에서 프레임을 고른다.

Utterance.started_at 을 대충 채우면 이 함정이 그대로 살아난다.
"""

from __future__ import annotations

import numpy as np

from jarviseo.types import Utterance

__all__ = ["SpeechToText"]


class SpeechToText:
    """녹음된 오디오를 글자로 바꾼다."""

    def __init__(self, model_name: str = "base", language: str = "ko") -> None:
        self.model_name = model_name
        self.language = language
        raise NotImplementedError

    def listen(self, timeout: float = 5.0) -> Utterance | None:
        """웨이크워드 이후 사용자 발화를 듣고 받아쓴다.

        VAD 로 발화 시작·끝을 잡고, started_at 에 **시작** 시각을 넣는다.
        timeout 안에 말이 없으면 None.
        """
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray, started_at: float) -> Utterance:
        """이미 녹음된 오디오를 받아쓴다. 테스트·재현용."""
        raise NotImplementedError
