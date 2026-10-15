"""가리킴 노드 — 검출 · 대상 결정 · 되묻기 · 크롭.  담당: 최홍묵

detect      : 사진에서 물체 후보를 찾는다 (pointing/detector.py)
resolve     : 후보 중 가리킨 대상을 고른다 (pointing/resolver.py)
clarify     : 1·2위가 비슷하면 되묻고 대답을 듣는다. 사진은 다시 고르지 않는다.
              2번 넘거나 대답이 없거나 후보가 없으면 "다시 가리켜 주세요"로 끝낸다
crop_target : 고른 대상만 잘라낸다 (pointing/crop.py). VLM · 성분 판정에는 이 크롭이 간다
"""

from __future__ import annotations

from typing import Literal

from jarviseo.pointing.crop import crop_target as cut_target
from jarviseo.workflow.deps import GraphDeps
from jarviseo.workflow.messages import CLARIFY_GIVE_UP, canned_response
from jarviseo.workflow.nodes.voice import REPLY_SEC
from jarviseo.workflow.state import MAX_CLARIFY, JarviseoState

__all__ = [
    "REPLY_HINTS",
    "after_clarify",
    "after_resolve",
    "clarify",
    "crop_target",
    "detect",
    "resolve",
]

# 되묻기 대답("왼쪽 거")을 받아쓸 때 STT 에 주는 힌트. 짧은 대답이 덜 틀린다(voice/stt.py).
# 후보 이름(한글)은 v1 대상 결정이 되묻는 문장을 만들 때 같이 더한다.
REPLY_HINTS = ("왼쪽", "오른쪽", "가운데")


def detect(state: JarviseoState, deps: GraphDeps) -> dict:
    return {"detections": deps.detector.detect(state["frame"].image)}


def resolve(state: JarviseoState, deps: GraphDeps) -> dict:
    # 되묻기 대답("왼쪽 거")만으로는 무엇을 찾는지 모른다. 처음 질문에 이어 붙여 넘긴다.
    text = " ".join([state["utterance"].text, *state.get("clarify_replies", [])])
    # 키워드로 불러야 가짜와 실제 resolve_target 을 바꿔 끼워도 안 깨진다.
    target = deps.resolve(
        candidates=state["detections"],
        fingertip=None,  # 손 단서(v2)가 붙으면 채운다
        gaze_vector=None,
        text=text,
        frame_size=state["frame"].size,
    )
    return {"target": target}


def after_resolve(state: JarviseoState) -> Literal["ambiguous", "chosen"]:
    return "ambiguous" if state["target"].needs_clarify else "chosen"


def clarify(state: JarviseoState, deps: GraphDeps) -> dict:
    target, count = state["target"], state.get("clarify_count", 0)
    if not target.candidates or count >= MAX_CLARIFY:
        return {"response": canned_response(state, CLARIFY_GIVE_UP)}
    deps.tts.speak(target.clarify_question or CLARIFY_GIVE_UP)
    # 사진을 새로 고르지 않는다(on_speech_start 없음). 대답하는 사이 고개를 돌렸어도
    # 처음 가리킨 장면에서 다시 골라야 한다.
    # 대답은 "컵"처럼 한 글자일 수 있다. 기본(두 글자 미만은 버림)대로면 대답이 없던 것이 된다.
    reply = deps.stt.listen(timeout=REPLY_SEC, min_letters=1, hints=REPLY_HINTS)
    if reply is None:
        return {"response": canned_response(state, CLARIFY_GIVE_UP)}
    # 한 개짜리 리스트로 줘야 리듀서(operator.add)가 앞의 대답 뒤에 이어 붙인다.
    return {"clarify_replies": [reply.text], "clarify_count": count + 1}


def after_clarify(state: JarviseoState) -> Literal["gave_up", "replied"]:
    return "gave_up" if "response" in state else "replied"  # 포기 안내를 만들었으면 읽으러


def crop_target(state: JarviseoState, deps: GraphDeps) -> dict:
    bbox = state["target"].chosen.detection.bbox
    return {"crop": cut_target(state["frame"].image, bbox)}
