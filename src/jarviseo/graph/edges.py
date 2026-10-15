"""그래프 분기. 상태를 보고 다음에 갈 노드 이름을 돌려준다.  담당: 최홍묵

돌려주는 이름은 pipeline.build_graph 의 매핑 키와 같아야 한다.
부품을 봐야 정할 수 있는 분기(사진 버퍼·가리킴 부품)는 deps 를 받는다.
"""

from __future__ import annotations

from typing import Literal

from jarviseo.capture import FrameBuffer
from jarviseo.graph.nodes import GraphDeps, can_point
from jarviseo.graph.state import JarviseoState
from jarviseo.types import Frame, Intent


def after_listen(state: JarviseoState, deps: GraphDeps) -> Literal["end", "recapture", "route"]:
    if state.get("utterance") is None:
        return "end"
    if deps.frames is not None and _is_bad(state.get("frame"), deps.sharpness_threshold):
        return "recapture"
    return "route"


def after_recapture(
    state: JarviseoState, deps: GraphDeps
) -> Literal["speak", "recapture", "route"]:
    if "response" in state:
        return "speak"  # 상한까지 다시 골라도 안 돼서 안내를 만들었다
    if _is_bad(state.get("frame"), deps.sharpness_threshold):
        return "recapture"
    return "route"


def after_route(
    state: JarviseoState, deps: GraphDeps
) -> Literal["recall", "detect", "check_ingredient", "generate"]:
    intent = state["intent"]
    if intent is Intent.RECALL:
        return "recall"
    if intent is Intent.INGREDIENT:
        # 가리킬 수 없으면 크롭 없이 사진 전체로 판정한다. VLM 으로는 보내지 않는다.
        return "detect" if can_point(state, deps) else "check_ingredient"
    if intent is Intent.POINTING:
        return "detect"
    return "generate"


def after_resolve(state: JarviseoState) -> Literal["clarify", "crop_target"]:
    return "clarify" if state["target"].needs_clarify else "crop_target"


def after_clarify(state: JarviseoState) -> Literal["speak", "resolve"]:
    return "speak" if "response" in state else "resolve"  # 포기 안내를 만들었으면 읽으러


def after_crop(state: JarviseoState) -> Literal["check_ingredient", "generate"]:
    return "check_ingredient" if state["intent"] is Intent.INGREDIENT else "generate"


def after_generate(state: JarviseoState) -> Literal["log", "speak"]:
    return "log" if state["interrupted"] else "speak"  # 버린 답은 읽지 않고 기록만


def _is_bad(frame: Frame | None, threshold: float | None) -> bool:
    """사진이 없으면 True. 흐림 기준이 있으면 그보다 흐려도 True."""
    if frame is None:
        return True
    return threshold is not None and FrameBuffer.is_too_blurry(frame, threshold)
