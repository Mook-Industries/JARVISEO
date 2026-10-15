"""의도 분류 노드.  담당: 최홍묵

질문 글자로 의도를 정한다(router.py, 키워드 규칙. LLM 을 부르지 않는다).
의도로 갈리는 분기 두 개(의도를 정한 뒤, 크롭한 뒤)도 여기 둔다.
"""

from __future__ import annotations

from typing import Literal

from jarviseo.types import Intent
from jarviseo.workflow.deps import GraphDeps, can_point
from jarviseo.workflow.router import classify_intent
from jarviseo.workflow.state import JarviseoState

__all__ = ["after_crop", "after_route", "route"]


def route(state: JarviseoState, deps: GraphDeps) -> dict:
    intent = classify_intent(state["utterance"].text)
    if intent is Intent.RECALL and not deps.recall_enabled:
        intent = Intent.GENERAL
    # 가리킬 사진이나 부품이 없으면 지시어 질문은 전체 사진(또는 글자만)으로 답한다.
    # 성분 질문은 그래도 VLM 으로 보내지 않는다(after_route 가 사진 전체로 판정하러 보낸다).
    if intent is Intent.POINTING and not can_point(state, deps):
        intent = Intent.GENERAL
    return {"intent": intent}


def after_route(
    state: JarviseoState, deps: GraphDeps
) -> Literal["recall", "needs_target", "ingredient", "answer"]:
    intent = state["intent"]
    if intent is Intent.RECALL:
        return "recall"
    if intent is Intent.INGREDIENT:
        # 가리킬 수 없으면 크롭 없이 사진 전체로 판정한다. VLM 으로는 보내지 않는다.
        return "needs_target" if can_point(state, deps) else "ingredient"
    if intent is Intent.POINTING:
        return "needs_target"
    return "answer"


def after_crop(state: JarviseoState) -> Literal["ingredient", "answer"]:
    return "ingredient" if state["intent"] is Intent.INGREDIENT else "answer"
