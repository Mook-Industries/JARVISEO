"""재촬영 노드.  담당: 최홍묵

고른 사진이 없거나 흐리면 버퍼에서 다시 고른다. 2번 넘으면 안내하고 끝낸다.
"""

from __future__ import annotations

from typing import Literal

from jarviseo.capture import FrameBuffer
from jarviseo.types import Frame
from jarviseo.workflow.deps import GraphDeps
from jarviseo.workflow.messages import RECAPTURE_GIVE_UP, canned_response
from jarviseo.workflow.state import MAX_RECAPTURE, JarviseoState

__all__ = ["after_recapture", "is_bad_frame", "recapture"]


def recapture(state: JarviseoState, deps: GraphDeps) -> dict:
    count = state.get("recapture_count", 0)
    if count >= MAX_RECAPTURE:
        return {"response": canned_response(state, RECAPTURE_GIVE_UP)}
    # 실행 중에는 카메라 스레드가 버퍼를 계속 채우니, 다시 고르면 그사이 들어온 사진도 후보다.
    return {"frame": deps.frames.pick_best(), "recapture_count": count + 1}


def after_recapture(
    state: JarviseoState, deps: GraphDeps
) -> Literal["gave_up", "still_bad", "good"]:
    if "response" in state:
        return "gave_up"  # 상한까지 다시 골라도 안 돼서 안내를 만들었다
    if is_bad_frame(state.get("frame"), deps.sharpness_threshold):
        return "still_bad"
    return "good"


def is_bad_frame(frame: Frame | None, threshold: float | None) -> bool:
    """사진이 없으면 True. 흐림 기준이 있으면 그보다 흐려도 True."""
    if frame is None:
        return True
    return threshold is not None and FrameBuffer.is_too_blurry(frame, threshold)
