"""그래프 노드와 부품 묶음.  담당: 최홍묵

노드는 상태와 부품(GraphDeps)을 받아, 바뀐 키만 담은 딕셔너리를 돌려준다.
각 모듈의 함수를 부르기만 하고 로직을 여기 넣지 않는다. 그래야 팀원이 자기 모듈만 고쳐도 된다.
부품은 pipeline.build_graph 가 functools.partial 로 묶어서 그래프에 넣는다.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from jarviseo import config
from jarviseo.capture import FrameBuffer
from jarviseo.graph.router import classify_intent
from jarviseo.graph.state import MAX_CLARIFY, MAX_RECAPTURE, JarviseoState
from jarviseo.pointing.crop import crop_target as cut_target
from jarviseo.types import AssistantResponse, Frame, Intent, TargetResolution

if TYPE_CHECKING:
    from jarviseo.memory import DialogMemory, MemoryStore, VectorMemory
    from jarviseo.nutrition.service import IngredientChecker
    from jarviseo.voice import ResponseCollector

__all__ = ["GraphDeps", "can_point"]

# 호출어 뒤 질문을 기다리는 시간(초). 보통 호출어에 바로 이어 말한다.
LISTEN_SEC = 2.0
# 호출어만 부르고 멈췄을 때 되묻는 말과, 되물은 뒤 대답을 기다리는 시간(초)
REPROMPT = "네, 말씀하세요."
REPLY_SEC = 5.0
# 되묻기 대답("왼쪽 거")을 받아쓸 때 STT 에 주는 힌트. 짧은 대답이 덜 틀린다(voice/stt.py).
# 후보 이름(한글)은 v1 대상 결정이 되묻는 문장을 만들 때 같이 더한다.
REPLY_HINTS = ("왼쪽", "오른쪽", "가운데")
# 안내 문장. 테스트도 이 상수로 "이 문장으로 끝났는가"를 본다.
SORRY = "죄송해요, 지금은 대답을 못 드려요. 잠시 뒤에 다시 물어봐 주세요."
RECAPTURE_GIVE_UP = "화면이 흐려서 잘 안 보여요. 다시 바라보며 불러 주세요."  # 스토리보드 S-06
CLARIFY_GIVE_UP = "어느 것을 말씀하시는지 모르겠어요. 다시 가리켜 주세요."  # 가리킴 흐름도
INGREDIENT_UNAVAILABLE = "지금은 성분을 확인할 수 없어요. 포장의 원재료명을 직접 확인해 주세요."
# 미리 정한 안내 문장들. 새 내용이 없어서 대화 기억에 넣지 않는다.
CANNED = frozenset({SORRY, RECAPTURE_GIVE_UP, CLARIFY_GIVE_UP, INGREDIENT_UNAVAILABLE})


@dataclass
class GraphDeps:
    """노드가 부르는 부품. 실행 스크립트는 실제 것을, 테스트는 Fake 를 넣는다.

    stt·tts·vlm·detector 는 실제 클래스와 Fake 가 공통 부모 없이 메서드 모양만 같아서
    타입을 Any 로 두고, 쓰는 메서드를 옆에 적는다. None 으로 둔 부품은 그 일을 건너뛴다.
    """

    stt: Any  # listen(), last_raw_text, last_latency_ms
    tts: Any  # speak(), last_interrupted, last_latency_ms
    vlm: Any  # ask(frame, question, memories), last_latency_ms
    # 없으면 사진 없이 묻는다. 가리킴·재촬영도 하지 않는다.
    frames: FrameBuffer | None = None
    # detect(image) -> list[Detection]. resolve 와 둘 다 있어야 가리킴 길이 열린다.
    detector: Any = None
    # resolve_target 과 같은 키워드를 받아 TargetResolution 을 돌려주는 함수
    resolve: Callable[..., TargetResolution] | None = None
    # 없으면 성분 질문에 "확인할 수 없어요"로 답한다. VLM 으로 넘기지 않는다.
    checker: IngredientChecker | None = None
    store: MemoryStore | None = None  # 없으면 턴을 기록하지 않고, 알레르기 목록도 비어 있다
    memory: VectorMemory | None = None  # 없으면 기억 검색 결과가 빈 목록이다
    dialog: DialogMemory | None = None  # 없으면 질문·대답을 기억에 넣지 않는다
    collector: ResponseCollector | None = None  # 없으면 "잠깐만요" 없이 답을 기다린다
    # 꺼져 있으면 기억을 묻는 질문도 GENERAL 로 보낸다(config 의 기능 플래그)
    recall_enabled: bool = config.ENABLE_RECALL
    # 안경 카메라로 기준을 재기 전까지는 흐림을 보지 않는다(None). quality.py 의 100 은
    # 출발점일 뿐이라, 켜 두면 어두운 방의 멀쩡한 사진도 다시 고른다.
    # 사진이 아예 없을 때는 이 값과 상관없이 다시 고른다.
    sharpness_threshold: float | None = None


def can_point(state: JarviseoState, deps: GraphDeps) -> bool:
    """가리킴 길을 탈 수 있는지. 검출기·대상 결정·사진이 다 있어야 한다."""
    return deps.detector is not None and deps.resolve is not None and state.get("frame") is not None


# -- 노드 ----------------------------------------------------------------------


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
    # 호출어를 잡은 시각부터 거슬러 읽어야 "자비서, 저거 뭐야"의 "저거"가 안 잘린다.
    utterance = stt.listen(on_speech_start=pick, timeout=LISTEN_SEC, since=wake_at)
    if utterance is None and wake_at is not None:
        # 호출어만 부르고 멈췄다. 되묻고 한 번 더 듣는다.
        deps.tts.speak(REPROMPT)
        utterance = stt.listen(on_speech_start=pick, timeout=REPLY_SEC)
    if utterance is None:
        return {}

    update: dict = {"utterance": utterance, "stt_raw_text": stt.last_raw_text}
    if stt.last_latency_ms is not None:
        update["latency_ms"] = {"stt": stt.last_latency_ms}
    if frames is not None:
        # 되물었으면 두 번째 질문 때 고른 사진. 그 구간에 사진이 없었으면 버퍼 전체에서 고른다.
        update["frame"] = picked[-1] if picked else frames.pick_best()
    return update


def recapture(state: JarviseoState, deps: GraphDeps) -> dict:
    count = state.get("recapture_count", 0)
    if count >= MAX_RECAPTURE:
        return {"response": _say(state, RECAPTURE_GIVE_UP)}
    # 실행 중에는 카메라 스레드가 버퍼를 계속 채우니, 다시 고르면 그사이 들어온 사진도 후보다.
    return {"frame": deps.frames.pick_best(), "recapture_count": count + 1}


def route(state: JarviseoState, deps: GraphDeps) -> dict:
    intent = classify_intent(state["utterance"].text)
    if intent is Intent.RECALL and not deps.recall_enabled:
        intent = Intent.GENERAL
    # 가리킬 사진이나 부품이 없으면 지시어 질문은 전체 사진(또는 글자만)으로 답한다.
    # 성분 질문은 그래도 VLM 으로 보내지 않는다(edges.after_route 가 check_ingredient 로 보낸다).
    if intent is Intent.POINTING and not can_point(state, deps):
        intent = Intent.GENERAL
    return {"intent": intent}


def recall(state: JarviseoState, deps: GraphDeps) -> dict:
    memory, user_id = deps.memory, state.get("user_id")
    if memory is None or user_id is None:
        return {"memories": []}
    update: dict = {"memories": memory.search(user_id, state["utterance"].text)}
    if memory.last_latency_ms is not None:
        update["latency_ms"] = {"recall": memory.last_latency_ms}
    return update


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


def clarify(state: JarviseoState, deps: GraphDeps) -> dict:
    target, count = state["target"], state.get("clarify_count", 0)
    if not target.candidates or count >= MAX_CLARIFY:
        return {"response": _say(state, CLARIFY_GIVE_UP)}
    deps.tts.speak(target.clarify_question or CLARIFY_GIVE_UP)
    # 사진을 새로 고르지 않는다(on_speech_start 없음). 대답하는 사이 고개를 돌렸어도
    # 처음 가리킨 장면에서 다시 골라야 한다.
    # 대답은 "컵"처럼 한 글자일 수 있다. 기본(두 글자 미만은 버림)대로면 대답이 없던 것이 된다.
    reply = deps.stt.listen(timeout=REPLY_SEC, min_letters=1, hints=REPLY_HINTS)
    if reply is None:
        return {"response": _say(state, CLARIFY_GIVE_UP)}
    # 한 개짜리 리스트로 줘야 리듀서(operator.add)가 앞의 대답 뒤에 이어 붙인다.
    return {"clarify_replies": [reply.text], "clarify_count": count + 1}


def crop_target(state: JarviseoState, deps: GraphDeps) -> dict:
    bbox = state["target"].chosen.detection.bbox
    return {"crop": cut_target(state["frame"].image, bbox)}


def check_ingredient(state: JarviseoState, deps: GraphDeps) -> dict:
    """성분 판정. VLM 을 부르지 않고, 규칙이 만든 판정 문장을 그대로 읽는다.

    VLM 이 포장 사진만 보고 "땅콩은 없어요"를 지어내면 사용자가 그 말을 믿고 먹는다.
    재촬영 안내(흐림·잘림)가 와도 그 문장을 읽고 이번 턴은 끝낸다.
    """
    crop, frame = state.get("crop"), state.get("frame")
    # 가리킴 부품이 없어 크롭을 못 만들었으면 사진 전체를 대상으로 본다.
    image = crop if crop is not None else (frame.image if frame is not None else None)
    if deps.checker is None or image is None:
        return {"response": _say(state, INGREDIENT_UNAVAILABLE)}
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


def generate(state: JarviseoState, deps: GraphDeps) -> dict:
    vlm, collector = deps.vlm, deps.collector
    question, frame = state["utterance"].text, state.get("frame")
    memories = state.get("memories")  # 키가 없으면 None: 기억을 안 찾은 질문
    # 대상을 골랐으면 그 크롭만 보낸다. 사진만 바꾼 Frame 이라 VLMClient 는 그대로 쓴다.
    # 선명도는 원본 사진의 값이라 지운다. numpy 배열은 `if crop:` 으로 쓰면 에러라
    # is not None 으로 본다.
    crop = state.get("crop")
    sent = replace(frame, image=crop, sharpness=None) if crop is not None else frame

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


def log(state: JarviseoState, deps: GraphDeps) -> dict:
    store, session_id = deps.store, state.get("session_id")
    if store is None or session_id is None:
        return {}
    response = replace(state["response"], latency_ms=dict(state.get("latency_ms", {})))
    turn_id = store.log_turn(
        session_id,
        state["utterance"],
        response,
        trigger_type=state.get("trigger_type", "WAKEWORD"),
        stt_raw_text=state.get("stt_raw_text"),
        interrupted=state.get("interrupted", False),
        filler_sent=state.get("filler_sent", False),
    )
    if "ingredient" in state:
        store.log_ingredient(turn_id, state["ingredient"])
    dialog, user_id = deps.dialog, state.get("user_id")
    if dialog is not None and user_id is not None and _worth_remembering(response):
        # 바로 돌아온다. 임베딩·저장은 뒤에서 한다
        dialog.remember(user_id, turn_id, state["utterance"].text, response.text)
    return {"turn_id": turn_id}


def _worth_remembering(response: AssistantResponse) -> bool:
    """대화 기억에 넣을 턴인지.

    답을 버렸거나(빈 대답) 미리 정한 안내(사과·재촬영·되묻기 포기·성분 확인 불가)만 읽었으면
    기억할 내용이 없다. 기억을 물은 턴(RECALL)도 넣지 않는다. 대답이 옛 기록을 다시 말한 것이라
    새 내용이 없고, 넣으면 다음에 비슷한 질문을 할 때 원래 기록보다 이 턴이 먼저 찾힌다.
    """
    return (
        bool(response.text) and response.text not in CANNED and response.intent is not Intent.RECALL
    )


def _say(state: JarviseoState, text: str) -> AssistantResponse:
    """미리 정한 안내를 답으로 만든다(재촬영·되묻기 포기, 성분 확인 불가).

    재촬영 포기는 의도 분류 전이라 의도가 없을 수 있다.
    """
    target = state.get("target")
    return AssistantResponse(
        text=text,
        intent=state.get("intent", Intent.GENERAL),
        used_frame=state.get("frame"),
        target=target,
        # 되묻다 끝난 턴은 아무도 대상을 고르지 않았다. log_turn 이 resolved_by 를 비운다.
        needs_clarify=target is not None and target.needs_clarify,
    )
