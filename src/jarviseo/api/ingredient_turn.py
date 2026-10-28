"""성분 질문을 한 턴 안에서 이어 가기 (음성 모드).  담당: 권용현

"이거 먹어도 돼?" 한 번에 끝까지 간다. 다시 "자비서"를 부르지 않아도 된다.

    1. 가리킨 물건의 크롭으로 판정(바코드 → 성분표 OCR, nutrition/service.py)
       못 하면 "성분표를 보여 주세요." → 시작음(띠딩) → 대답을 듣는다
    2. "여기" → 그 순간의 장면 전체로 다시 판정
       못 읽으면 "성분표를 다시 보여 주세요." → 시작음 → 대답을 듣는다
    3. "여기" → 또 못 읽으면 "성분표를 읽지 못했습니다." 로 턴을 끝낸다

대답이 없거나(``REPLY_SEC``) "그만" · "취소"라고 하면 3과 같이 끝낸다.
대답은 아무 말이나 된다("여기", "이거", "됐어?"). 말한 순간의 장면을 고른다(FrameBuffer.pick_best).

그래프(workflow, 최홍묵)는 그대로 두고 성분 판정 부품(checker)만 감싼다. 되묻기(clarify)처럼
팀 부품을 그대로 쓴다: 읽기(TextToSpeech) · 시작음(Chime.start) · 받아쓰기(SpeechToText).
마지막 결과의 문장은 그래프의 speak 노드가 읽는다.

입력창 질문(말이 아니라 글자로 물은 턴)에는 듣지 않는다. 마이크 앞에 없을 수 있어서다.
그때는 active() 가 False 이고, 턴을 넘어 세는 RetakeLadder 로 두 번째부터 다시 찍어 달라고 한다.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from jarviseo.nutrition.response import RETAKE_GAVE_UP, RETAKE_SHOW_PANEL, RETAKE_SHOW_PANEL_AGAIN

__all__ = ["REPLY_SEC", "IngredientFollowUp"]

log = logging.getLogger(__name__)

# 보여 달라고 한 뒤 대답을 기다리는 시간(초). 포장을 돌려 카메라에 대는 시간을 넉넉히 준다.
REPLY_SEC = 8.0
# 이 말이 들어 있으면 보여 주기를 그만둔다(띄어쓰기를 지우고 본다)
STOP_WORDS = ("그만", "취소", "됐어", "안할래", "필요없")


class IngredientFollowUp:
    """성분 판정기(checker)를 감싸 못 읽으면 같은 턴 안에서 보여 달라고 하고 다시 읽는다."""

    def __init__(
        self,
        inner,
        tts,
        stt,
        frames,
        start_chime: Callable[[], None] | None = None,
        on_listen: Callable[[], None] | None = None,
        active: Callable[[], bool] | None = None,
    ) -> None:
        """inner 는 판정기(check(image, allergens)).

        tts · stt · frames 는 그래프에 넣은 것과 같은 부품이다.

        on_listen 은 시작음을 낸 뒤 부른다(화면에 "말씀하세요"를 띄우려고).
        active() 가 False 면(입력창 질문) 듣지 않고 RetakeLadder 로만 안내한다.
        """
        from jarviseo.api.assistant_web import RetakeLadder

        self.inner, self.tts, self.stt, self.frames = inner, tts, stt, frames
        self.start_chime, self.on_listen = start_chime, on_listen
        self.active = active
        self.ladder = RetakeLadder()
        self.steps: list[str] = []  # 이번 턴에 보여 달라고 한 문장(기록 · 시험용)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def check(self, image: np.ndarray, allergens: list[str], *args, **kwargs):
        self.steps = []
        result = self.inner.check(image, allergens, *args, **kwargs)
        if self.active is not None and not self.active():
            return self.ladder.judge(result)
        for prompt in (RETAKE_SHOW_PANEL, RETAKE_SHOW_PANEL_AGAIN):
            if not result.retake:
                return result
            shown = self._ask_to_show(prompt)
            if shown is None:
                break  # 대답이 없거나 그만두었다
            result = self.inner.check(shown, allergens)
        if result.retake:
            result = replace(result, retake=RETAKE_GAVE_UP, message=RETAKE_GAVE_UP)
        return result

    def _ask_to_show(self, prompt: str) -> np.ndarray | None:
        """보여 달라고 말하고 시작음 뒤 대답을 듣는다. 대답한 순간의 장면 전체를 돌려준다."""
        self.steps.append(prompt)
        self.tts.speak(prompt)
        if getattr(self.tts, "last_interrupted", False):
            return None  # 읽는 중에 "자비서"로 끼어들었다. 그 말은 새 질문이다
        if self.start_chime is not None:
            self.start_chime()
        if self.on_listen is not None:
            self.on_listen()
        heard = self.stt.listen(timeout=REPLY_SEC, min_letters=1)
        if heard is None:
            log.info("보여 달라고 했는데 대답이 없다")
            return None
        if any(word in "".join(heard.text.split()) for word in STOP_WORDS):
            return None
        if self.frames is None:
            return None
        frame = self.frames.pick_best(around=heard.ended_at) or self.frames.pick_best()
        return frame.image if frame is not None else None
