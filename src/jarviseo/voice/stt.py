"""받아쓰기 (STT).  담당: 문태현

호출어 뒤의 발화 한 번만 OpenAI 받아쓰기 API 로 보낸다.
상시 듣는 호출어 감지는 로컬(``wakeword.py``)에 남는다.

모델 이름은 이 파일에 적지 않는다. ``config.STT_MODEL``
(``.env`` 의 ``JARVISEO_STT_MODEL``) 하나만 본다.

녹음 · VAD · 받아쓰기는 스파이크 02 결과를 보고 다음 이슈에서 채운다.
그때도 결과에서 가장 중요한 값은 글자가 아니라 ``Utterance.started_at``
(말을 시작한 시각)이다. 프레임을 이 시각 기준으로 고른다.
"""

from __future__ import annotations

from jarviseo import config

__all__ = ["SpeechToText"]


class SpeechToText:
    """녹음된 발화를 글자로 바꾼다."""

    def __init__(self, model: str = config.STT_MODEL, language: str = "ko") -> None:
        self.model = model
        self.language = language
