"""VLM 답 노드.  담당: 최홍묵 (부르는 모듈: vlm/ · voice/collector.py 문태현)

VLM 답을 받는다. 늦으면 "잠깐만요"를 먼저 읽고, 기다리는 동안 다시 불리면 답을 버린다.
GraphDeps.vlm_deadline_sec(기본 5초) 안에 답이 없으면 기다리기를 멈추고 안내 문장으로 끝낸다.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Literal

from jarviseo.pointing.crop import fit_for_vlm
from jarviseo.types import AssistantResponse
from jarviseo.workflow.deps import GraphDeps
from jarviseo.workflow.follow_up import asked
from jarviseo.workflow.messages import SORRY, VLM_TOO_SLOW
from jarviseo.workflow.state import JarviseoState

__all__ = ["HAND_CONTEXT", "POINTING_CONTEXT", "after_generate", "generate"]

# VLM 지시문(vlm/client.py)은 사진을 "질문한 순간 보인 장면"이라고 설명한다. 크롭을 보낼 때
# 그대로 두면 잘린 사진을 장면 전체로 믿는다.
# 검출 이름은 넣지 않는다. 박스는 맞아도 이름이 틀리는 일이 잦고(R0 valid 4/11), 틀린 이름을
# 주면 VLM 이 "고양이는 아니고…"처럼 그 이름을 꺼낸다(모의 루프 턴 22, #229).
POINTING_CONTEXT = (
    "사진은 사용자가 가리킨 물건만 잘라낸 것이다. 주변 장면은 보이지 않는다.\n질문: {question}"
)
# 손 우선 크롭은 후보 박스가 아니라 손가락 선을 따라 자른 것이라 손가락이 같이 보인다.
# 그래서 손끝이 향한 물건을 보라고 덧붙인다. 이 문장에도 검출 이름은 넣지 않는다.
HAND_CONTEXT = (
    "사진은 사용자가 손가락으로 가리킨 쪽을 잘라낸 것이다. "
    "주변 장면은 보이지 않고 손가락이 같이 보일 수 있다.\n"
    "손가락 끝이 향한 물건을 보고 답한다.\n"
    "질문: {question}"
)

log = logging.getLogger(__name__)

# 상한이 지나도 이미 나간 HTTP 요청은 끊을 수 없어서, 클라이언트 타임아웃(최악 20초)까지 여기서
# 마저 돌게 둔다. 그사이 연달아 물어도 새 요청이 막히지 않게 일꾼을 넉넉히 둔다.
# 프로그램을 끝낼 때 파이썬이 남은 요청을 기다려서, Ctrl+C 뒤 최악 20초 걸릴 수 있다.
_VLM_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vlm")


def generate(state: JarviseoState, deps: GraphDeps) -> dict:
    vlm, collector = deps.vlm, deps.collector
    question, frame = asked(state), state.get("frame")
    memories = state.get("memories")  # 키가 없으면 None: 기억을 안 찾은 질문
    # 대상을 골랐으면 그 크롭만 보낸다. 사진만 바꾼 Frame 이라 VLMClient 는 그대로 쓴다.
    # 선명도는 원본 사진의 값이라 지운다. numpy 배열은 `if crop:` 으로 쓰면 에러라
    # is not None 으로 본다.
    # 크기 조정(64px 확대 · 긴 변 768px)은 VLM 으로 보낼 때만 한다. 성분 판정은 원본으로 읽는다.
    crop = state.get("crop")
    sent = replace(frame, image=fit_for_vlm(crop), sharpness=None) if crop is not None else frame

    # 맥락은 VLM 에만 간다. 대화 기억에는 맥락 없는 질문(asked)이 들어간다(nodes/memory.py).
    prompt = _with_pointing_context(question, state)

    too_slow = False

    def ask() -> str | None:
        nonlocal too_slow
        future = _VLM_POOL.submit(vlm.ask, sent, prompt, memories)
        try:
            return future.result(timeout=deps.vlm_deadline_sec)
        except TimeoutError:
            too_slow = True
            return None  # collector 는 None 이면 미리 합성하지 않는다. 늦게 온 답은 버려진다

    # collector 는 답을 다른 스레드에서 기다리다가 늦으면 "잠깐만요"를 먼저 읽는다.
    answer = collector.collect(ask) if collector is not None else ask()
    dropped = collector is not None and collector.dropped
    if too_slow and not dropped:
        log.warning("VLM 답이 %.1f초 안에 안 와서 안내 문장으로 끝냈다", deps.vlm_deadline_sec)
    response = AssistantResponse(
        text="" if dropped else (VLM_TOO_SLOW if too_slow else (answer or SORRY)),
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
    # 버린 답 · 늦은 답은 아직 받는 중이라 last_latency_ms 가 이번 값이 아니다.
    # (끊긴 옛 요청이 다음 질문 도중에 끝나면 그 값을 써 넣을 수 있다. 클라이언트가 값 하나를
    # 같이 쓰기 때문이고, 새 요청이 끝날 때 다시 덮어써서 대부분 맞다.)
    if not dropped and not too_slow and vlm.last_latency_ms is not None:
        update["latency_ms"] = {"vlm": vlm.last_latency_ms}
    return update


def after_generate(state: JarviseoState) -> Literal["dropped", "answered"]:
    return "dropped" if state["interrupted"] else "answered"  # 버린 답은 읽지 않고 기록만


def _with_pointing_context(question: str, state: JarviseoState) -> str:
    """크롭을 보낼 때만(POINTING) 질문 앞에 가리킴 맥락을 붙인다. 손 우선 크롭이면 그 맥락을."""
    if state.get("crop") is None:
        return question
    if state.get("hand_crop") is not None:
        return HAND_CONTEXT.format(question=question)
    return POINTING_CONTEXT.format(question=question)
