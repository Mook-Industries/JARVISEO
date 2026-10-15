"""성분 판정 노드.  담당: 최홍묵 (부르는 모듈: nutrition/ 권용현)

VLM 을 부르지 않는다. 판정은 규칙(nutrition/service.py)이 내리고 그 문장을 그대로 읽는다.
"""

from __future__ import annotations

from jarviseo.types import AssistantResponse
from jarviseo.workflow.deps import GraphDeps
from jarviseo.workflow.messages import INGREDIENT_UNAVAILABLE, canned_response
from jarviseo.workflow.state import JarviseoState

__all__ = ["check_ingredient"]


def check_ingredient(state: JarviseoState, deps: GraphDeps) -> dict:
    """성분 판정. VLM 을 부르지 않고, 규칙이 만든 판정 문장을 그대로 읽는다.

    VLM 이 포장 사진만 보고 "땅콩은 없어요"를 지어내면 사용자가 그 말을 믿고 먹는다.
    재촬영 안내(흐림·잘림)가 와도 그 문장을 읽고 이번 턴은 끝낸다.
    """
    crop, frame = state.get("crop"), state.get("frame")
    # 가리킴 부품이 없어 크롭을 못 만들었으면 사진 전체를 대상으로 본다.
    image = crop if crop is not None else (frame.image if frame is not None else None)
    if deps.checker is None or image is None:
        return {"response": canned_response(state, INGREDIENT_UNAVAILABLE)}
    store, user_id = deps.store, state.get("user_id")
    known = store is not None and user_id is not None
    allergens = store.get_user_allergens(user_id) if known else []
    check = deps.checker.check(image, allergens)
    response = AssistantResponse(
        text=check.message,
        intent=state["intent"],
        used_frame=frame,
        target=state.get("target"),
        allergen=check.judgement,
    )
    return {"ingredient": check, "response": response}
