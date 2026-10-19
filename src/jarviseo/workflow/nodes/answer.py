"""VLM 답 노드.  담당: 최홍묵 (부르는 모듈: vlm/ · voice/collector.py 문태현)

VLM 답을 받는다. 늦으면 "잠깐만요"를 먼저 읽고, 기다리는 동안 다시 불리면 답을 버린다.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from jarviseo.pointing.crop import fit_for_vlm
from jarviseo.types import AssistantResponse
from jarviseo.workflow.deps import GraphDeps
from jarviseo.workflow.messages import SORRY
from jarviseo.workflow.state import JarviseoState

__all__ = ["after_generate", "generate"]


def generate(state: JarviseoState, deps: GraphDeps) -> dict:
    vlm, collector = deps.vlm, deps.collector
    question, frame = state["utterance"].text, state.get("frame")
    memories = state.get("memories")  # 키가 없으면 None: 기억을 안 찾은 질문
    # 대상을 골랐으면 그 크롭만 보낸다. 사진만 바꾼 Frame 이라 VLMClient 는 그대로 쓴다.
    # 선명도는 원본 사진의 값이라 지운다. numpy 배열은 `if crop:` 으로 쓰면 에러라
    # is not None 으로 본다.
    # 크기 조정(64px 확대 · 긴 변 768px)은 VLM 으로 보낼 때만 한다. 성분 판정은 원본으로 읽는다.
    crop = state.get("crop")
    sent = replace(frame, image=fit_for_vlm(crop), sharpness=None) if crop is not None else frame

    def ask() -> str | None:
        return vlm.ask(sent, question, memories)

    # collector 는 답을 다른 스레드에서 기다리다가 늦으면 "잠깐만요"를 먼저 읽는다.
    answer = collector.collect(ask) if collector is not None else ask()
    dropped = collector is not None and collector.dropped
    response = AssistantResponse(
        text="" if dropped else (answer or SORRY),
        intent=state["intent"],
        used_frame=frame,  # 기록에는 원본 장면을 남긴다. 틀린 사례를 다시 보려고
        target=state.get("target"),  # log_turn 이 이걸로 turn_candidate 를 쓴다
        memories=memories or [],
    )
    update: dict = {
        "response": response,
        "filler_sent": collector is not None and collector.filler_sent,
        "interrupted": dropped,
    }
    # 버린 답은 아직 받는 중이라 last_latency_ms 가 이번 값이 아니다.
    if not dropped and vlm.last_latency_ms is not None:
        update["latency_ms"] = {"vlm": vlm.last_latency_ms}
    return update


def after_generate(state: JarviseoState) -> Literal["dropped", "answered"]:
    return "dropped" if state["interrupted"] else "answered"  # 버린 답은 읽지 않고 기록만
