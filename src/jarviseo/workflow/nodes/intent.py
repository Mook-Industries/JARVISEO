"""의도 분류 노드.  담당: 최홍묵

질문 글자로 의도를 정한다(router.py, 키워드 규칙. LLM 을 부르지 않는다).
글자로 GENERAL 이어도 사진에 가리키는 손이 보이면 가리킴으로 보낸다(장면 · 찾기 · 영역 질문은
빼고, #217).
의도로 갈리는 분기 두 개(의도를 정한 뒤, 크롭한 뒤)도 여기 둔다.
"""

from __future__ import annotations

from typing import Literal

from jarviseo.pointing.hands import HandPose, pick_pointing_hand
from jarviseo.types import Intent
from jarviseo.workflow.deps import GraphDeps, can_point
from jarviseo.workflow.router import asks_whole_scene, classify_intent
from jarviseo.workflow.state import JarviseoState

__all__ = ["after_crop", "after_route", "route"]


def route(state: JarviseoState, deps: GraphDeps) -> dict:
    text = state["utterance"].text
    intent = classify_intent(text)
    if intent is Intent.GENERAL:
        hand = _pointing_hand(state, deps, text)
        if hand is not None:
            # 지시어가 없어도("이 약 하루에 몇 번 먹어야 돼?") 손으로 가리키면 물건 하나를 묻는
            # 것이다. 글자로는 "이 약"과 "이 근처"를 못 가른다(#147). 찾은 손은 detect 가 다시
            # 찾지 않고 쓴다.
            return {"intent": Intent.POINTING, "hand": hand}
    if intent is Intent.RECALL and not deps.recall_enabled:
        intent = Intent.GENERAL
    # 가리킬 사진이나 부품이 없으면 지시어 질문은 전체 사진(또는 글자만)으로 답한다.
    # 성분 질문은 그래도 VLM 으로 보내지 않는다(after_route 가 사진 전체로 판정하러 보낸다).
    if intent is Intent.POINTING and not can_point(state, deps):
        intent = Intent.GENERAL
    return {"intent": intent}


def _pointing_hand(state: JarviseoState, deps: GraphDeps, text: str) -> HandPose | None:
    """GENERAL 질문 사진의 가리키는 손. 장면 · 찾기 · 영역 질문이거나 가리킬 수 없으면 찾지 않는다.

    손 찾기(약 25ms)는 GENERAL 질문에서만 한 번 돈다. 지시어 · 성분 질문은 detect 가, 기억 질문은
    사진을 안 써서 여기서 찾을 일이 없다. route 는 시간을 재는 노드라 이 시간은 route 칸에 들어간다.
    """
    if asks_whole_scene(text) or deps.hands is None or not can_point(state, deps):
        return None
    return pick_pointing_hand(deps.hands.detect(state["frame"].image))


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
