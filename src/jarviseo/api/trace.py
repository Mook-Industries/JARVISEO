"""환경설정 '미량 혼입 가능성 알림'.  담당: 권용현

끄면(받지 않기) 성분 판정에서 '같은 시설에서 섞였을 수 있음'(혼입 가능)은 말하지 않고
'들어 있음'(함유)만 말한다. 켜면(받기) 지금처럼 둘 다 말한다.

값은 등록한 알레르기 행의 ``user_allergen.is_strict`` 에 둔다. ERD 에서 그 열의 뜻이
"true 면 미량 · 교차오염 의심까지 경고한다"라서다. 화면 스위치는 알레르기마다가 아니라 하나라서
등록한 알레르기 전부에 같은 값을 쓴다. 알레르기를 하나도 등록하지 않았으면 켬(받기)으로 본다.

서버가 만드는 판정기(app.state.checker) 하나를 감싸므로 웹 질문 · 팀 음성 모드 · 브라우저 음성 ·
성분 확인 페이지가 모두 같은 설정을 따른다.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import select

from jarviseo.memory.models import UserAllergen
from jarviseo.nutrition.service import without_may_contain

__all__ = ["DEFAULT_TRACE", "TraceChecker", "get_trace", "set_trace"]

# 등록한 알레르기가 없을 때. 경고를 놓치지 않는 쪽(받기)으로 둔다.
DEFAULT_TRACE = True


def _active(user_id: int):
    return select(UserAllergen).where(
        UserAllergen.user_id == user_id, UserAllergen.deleted_at.is_(None)
    )


def get_trace(store, user_id: int) -> bool:
    """혼입 가능성까지 알릴지. 등록한 알레르기 중 하나라도 is_strict 면 켬."""
    with store.session() as db:
        rows = list(db.scalars(_active(user_id)))
    return any(r.is_strict for r in rows) if rows else DEFAULT_TRACE


def set_trace(store, user_id: int, on: bool) -> None:
    """등록한 알레르기 전부의 is_strict 를 on 으로 맞춘다."""
    with store.session() as db:
        for row in db.scalars(_active(user_id)):
            row.is_strict = on


class TraceChecker:
    """판정기를 감싸 설정이 꺼져 있으면 혼입 가능을 빼고 함유만 말하게 한다."""

    def __init__(self, inner, store, user_id: int) -> None:
        self.inner, self.store, self.user_id = inner, store, user_id

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def check(self, image: np.ndarray, allergens: list[str], *args, barcode_image=None, **kwargs):
        # barcode_image 는 이름을 그대로 둬야 감싸는 쪽(_RetryWholeFrame)이 받는 줄 안다
        if barcode_image is not None:
            kwargs["barcode_image"] = barcode_image
        result = self.inner.check(image, allergens, *args, **kwargs)
        if get_trace(self.store, self.user_id):
            return result
        return without_may_contain(result, allergens, getattr(self.inner, "synonyms", {}) or {})
