"""듣기 · 읽기 노드.  담당: 최홍묵 (부르는 모듈: voice/ 문태현)

듣기는 받아쓰고, 말을 시작한 순간 그 시각 근처에서 가장 선명한 사진을 고른다.
호출어 2단계 확인(deps.wake_check)이 호출어가 아니라고 하면 되묻기·대답 없이 끝난다.
읽기는 답을 소리로 읽는다. 도중에 "자비서"가 들리면 바로 끊긴다.
"""

from __future__ import annotations

import time
from typing import Literal

from jarviseo.types import Frame
from jarviseo.workflow.deps import GraphDeps
from jarviseo.workflow.messages import REPROMPT
from jarviseo.workflow.nodes.capture import is_bad_frame
from jarviseo.workflow.state import JarviseoState

__all__ = ["LISTEN_SEC", "REPLY_SEC", "after_listen", "listen", "speak"]

# 호출어 뒤 질문을 기다리는 시간(초). 보통 호출어에 바로 이어 말한다.
LISTEN_SEC = 2.0
# 되물은 뒤(호출어만 불렀을 때, 대상이 애매할 때) 대답을 기다리는 시간(초)
REPLY_SEC = 5.0


def listen(state: JarviseoState, deps: GraphDeps) -> dict:
    stt, frames = deps.stt, deps.frames
    picked: list[Frame] = []

    def pick(started_at: float) -> None:
        # 말을 시작한 순간에 고른다. 버퍼는 1초쯤만 들고 있어서,
        # 받아쓰기가 끝난 뒤에 고르면 그 시각 사진이 이미 없다.
        frame = frames.pick_best(around=started_at) if frames is not None else None
        if frame is not None:
            picked.append(frame)

    wake_at = state.get("wake_at")
    # 호출어 2단계 확인(#149). 질문을 듣는 동안 감지 직전 구간을 따로 받아쓴다.
    # 호출어가 아니라고 나오면 듣던 것도 멈춘다(TV 소리는 끝없이 이어져 10초를 다 채운다).
    verdict = None
    if deps.wake_check is not None and wake_at is not None:
        verdict = deps.wake_check.start(wake_at)
    stop = verdict.rejected if verdict is not None else None
    # 호출어를 잡은 시각부터 거슬러 읽어야 "자비서, 저거 뭐야"의 "저거"가 안 잘린다.
    utterance = stt.listen(on_speech_start=pick, timeout=LISTEN_SEC, since=wake_at, stop=stop)
    # 확인에서 받아쓴 글자는 통과한 턴에도 남긴다. 실측에서 무엇에 깨어났는지 본다.
    checked = {"wake_check_text": verdict.text} if verdict is not None else {}
    if verdict is not None and not verdict.passed():
        # TV 소리 같은 것에 깨어났다. 되묻지도 대답하지도 않고 호출어 대기로 돌아간다.
        return {"wake_rejected": True} | checked
    if utterance is None and wake_at is not None:
        # 호출어만 부르고 멈췄다. 되묻고 한 번 더 듣는다.
        deps.tts.speak(REPROMPT)
        utterance = stt.listen(on_speech_start=pick, timeout=REPLY_SEC)
    if utterance is None:
        return checked

    update: dict = {"utterance": utterance, "stt_raw_text": stt.last_raw_text} | checked
    if stt.last_latency_ms is not None:
        update["latency_ms"] = {"stt": stt.last_latency_ms}
    if frames is not None:
        # 되물었으면 두 번째 질문 때 고른 사진. 그 구간에 사진이 없었으면 버퍼 전체에서 고른다.
        update["frame"] = picked[-1] if picked else frames.pick_best()
    return update


def after_listen(
    state: JarviseoState, deps: GraphDeps
) -> Literal["wake_rejected", "no_question", "bad_frame", "heard"]:
    if state.get("wake_rejected"):
        return "wake_rejected"
    if state.get("utterance") is None:
        return "no_question"
    if deps.frames is not None and is_bad_frame(state.get("frame"), deps.sharpness_threshold):
        return "bad_frame"
    return "heard"


def speak(state: JarviseoState, deps: GraphDeps) -> dict:
    tts = deps.tts
    asked = time.monotonic()
    tts.speak(state["response"].text)  # 도중에 "자비서"가 들리면 바로 끊고 돌아온다
    update: dict = {"interrupted": tts.last_interrupted}
    if tts.last_latency_ms is not None:
        # 말이 끝나고 대답 첫 소리가 나오기까지. "잠깐만요"를 읽었어도 대답 기준으로 잰다.
        total = (asked - state["utterance"].ended_at) * 1000 + tts.last_latency_ms
        update["latency_ms"] = {"tts": tts.last_latency_ms, "total": total}
    return update
