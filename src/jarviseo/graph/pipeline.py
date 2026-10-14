"""LangGraph 그래프 구성.  담당: 최홍묵 (D12 최소 그래프: 문태현)

노드와 분기를 여기서 엮는다.
개별 노드는 각 모듈의 함수를 부르기만 하고, 로직을 여기 넣지 않는다.
그래야 팀원이 자기 모듈만 고쳐도 되고 이 파일에서 충돌이 나지 않는다.

질문 한 번이 graph.invoke() 한 번이다.

    listen ──(질문 없음)──────────────────────────────────────────▶ END
      │
      ▼
    route ──(RECALL)──▶ recall ──┐
      │                          ▼
      └──(GENERAL)──────────▶ generate ──(다시 불러 답을 버림)──▶ log ──▶ END
                                 │                               ▲
                                 └──────────▶ speak ─────────────┘

    listen   : 받아쓴다. 말을 시작한 순간 그 시각 근처에서 가장 선명한 사진을 고른다
    route    : 의도를 정한다 (router.py, 지금은 RECALL / GENERAL)
    recall   : 기억을 찾는다 (memory/vector.py)
    generate : VLM 답을 받는다. 늦으면 "잠깐만요"를 먼저 읽는다 (voice/collector.py)
    speak    : 답을 읽는다 (voice/tts.py). 도중에 "자비서"가 들리면 바로 끊긴다
    log      : 턴을 DB 에 남기고, 장면 설명 저장을 뒤에 맡긴다 (memory/store.py, scene.py)

호출어를 기다리는 일과 끼어들기는 그래프 밖(scripts/assistant.py)에서 한다.
호출어가 들리면 graph.invoke({"wake_at": 감지 시각, ...}) 로 질문 한 번을 돌린다.

아직 없는 것 (이 위에 붙일 자리)
    detect · resolve · clarify : POINTING 이 붙으면 route 뒤에 (가리킴 담당)
    ocr                        : INGREDIENT 가 붙으면 route 뒤에 (성분 담당)
    recapture                  : listen 이 고른 사진이 흐리면(is_too_blurry) 다시 고르기
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from jarviseo import config
from jarviseo.graph.router import route as decide_intent
from jarviseo.graph.state import JarviseoState
from jarviseo.types import AssistantResponse, Frame, Intent

if TYPE_CHECKING:
    from jarviseo.capture import FrameBuffer
    from jarviseo.memory import MemoryStore, SceneMemory, VectorMemory
    from jarviseo.vlm import FakeVLMClient, VLMClient
    from jarviseo.voice import (
        FakeSpeechToText,
        FakeTextToSpeech,
        ResponseCollector,
        SpeechToText,
        TextToSpeech,
    )

__all__ = ["Parts", "build_graph"]

# 호출어 뒤 질문을 기다리는 시간(초). 보통 호출어에 바로 이어 말한다.
LISTEN_SEC = 2.0
# 호출어만 부르고 멈췄을 때 되묻는 말과, 되물은 뒤 기다리는 시간(초)
REPROMPT = "네, 말씀하세요."
REPROMPT_SEC = 5.0
# VLM 이 답을 못 줬을 때 대신 읽는 말
SORRY = "죄송해요, 지금은 대답을 못 드려요. 잠시 뒤에 다시 물어봐 주세요."


@dataclass
class Parts:
    """노드가 부르는 부품. 실행 스크립트는 실제 것을, 테스트는 Fake 를 넣는다.

    없어도 되는 것은 None 으로 둔다. frames 가 없으면 사진 없이 묻고, memory 가 없으면
    기억 검색을 건너뛰고, store 가 없으면 기록을 남기지 않는다.
    """

    stt: SpeechToText | FakeSpeechToText
    tts: TextToSpeech | FakeTextToSpeech
    vlm: VLMClient | FakeVLMClient
    frames: FrameBuffer | None = None
    store: MemoryStore | None = None
    memory: VectorMemory | None = None
    scenes: SceneMemory | None = None
    collector: ResponseCollector | None = None
    # 꺼져 있으면 기억을 묻는 질문도 GENERAL 로 보낸다(config 의 기능 플래그)
    recall_enabled: bool = config.ENABLE_RECALL


def build_graph(parts: Parts):  # -> CompiledStateGraph
    """그래프를 만들어 컴파일해서 반환한다.

    graph.invoke({"session_id": ..., "user_id": ..., "wake_at": ..., "trigger_type": ...})
    처럼 턴 정보를 넣어 부르면, 끝난 뒤의 상태(답·지연·turn_id 등)를 돌려준다.
    """
    nodes = _Nodes(parts)
    graph = StateGraph(JarviseoState)
    graph.add_node("listen", nodes.listen)
    graph.add_node("route", nodes.route)
    graph.add_node("recall", nodes.recall)
    graph.add_node("generate", nodes.generate)
    graph.add_node("speak", nodes.speak)
    graph.add_node("log", nodes.log)

    graph.add_edge(START, "listen")
    graph.add_conditional_edges("listen", _after_listen, ["route", END])
    graph.add_conditional_edges("route", _after_route, ["recall", "generate"])
    graph.add_edge("recall", "generate")
    graph.add_conditional_edges("generate", _after_generate, ["speak", "log"])
    graph.add_edge("speak", "log")
    graph.add_edge("log", END)
    return graph.compile()


def _after_listen(state: JarviseoState) -> str:
    """질문을 못 받았으면 끝낸다. 남길 기록도 없다."""
    return "route" if state.get("utterance") is not None else END


def _after_route(state: JarviseoState) -> str:
    return "recall" if state["intent"] is Intent.RECALL else "generate"


def _after_generate(state: JarviseoState) -> str:
    """답을 기다리는 동안 다시 불러서 답을 버렸으면, 읽지 않고 기록만 남긴다."""
    return "log" if state.get("interrupted") else "speak"


class _Nodes:
    """노드 함수 묶음. 노드는 상태를 받아 자기가 채운 키만 돌려준다."""

    def __init__(self, parts: Parts) -> None:
        self.parts = parts

    def listen(self, state: JarviseoState) -> dict:
        stt, frames = self.parts.stt, self.parts.frames
        picked: list[Frame] = []

        def pick(started_at: float) -> None:
            # 말을 시작한 순간에 고른다. 받아쓰기가 끝난 뒤(1초 넘게 뒤)에는
            # 버퍼(약 1초)에 그 시각 사진이 남아 있지 않다.
            frame = frames.pick_best(around=started_at) if frames is not None else None
            if frame is not None:
                picked.append(frame)

        wake_at = state.get("wake_at")
        utterance = stt.listen(on_speech_start=pick, timeout=LISTEN_SEC, since=wake_at)
        if utterance is None and wake_at is not None:
            # 호출어만 부르고 멈췄다. 되묻고 다시 듣는다(읽은 뒤 0.3초는 게이트가 막는다).
            self.parts.tts.speak(REPROMPT)
            utterance = stt.listen(on_speech_start=pick, timeout=REPROMPT_SEC)
        if utterance is None:
            return {}

        update: dict = {"utterance": utterance, "stt_raw_text": stt.last_raw_text}
        if stt.last_latency_ms is not None:
            update["latency_ms"] = {"stt": stt.last_latency_ms}
        if picked:
            update["frame"] = picked[-1]  # 되물었으면 두 번째 질문의 사진
        return update

    def route(self, state: JarviseoState) -> dict:
        started = time.perf_counter()
        intent = decide_intent(state["utterance"].text)
        if intent is Intent.RECALL and not self.parts.recall_enabled:
            intent = Intent.GENERAL
        return {"intent": intent, "latency_ms": {"route": _ms(started)}}

    def recall(self, state: JarviseoState) -> dict:
        memory, user_id = self.parts.memory, state.get("user_id")
        if memory is None or user_id is None:
            return {"memories": []}
        update: dict = {"memories": memory.search(user_id, state["utterance"].text)}
        if memory.last_latency_ms is not None:
            update["latency_ms"] = {"recall": memory.last_latency_ms}
        return update

    def generate(self, state: JarviseoState) -> dict:
        vlm, collector = self.parts.vlm, self.parts.collector
        question, frame = state["utterance"].text, state.get("frame")
        memories = state.get("memories")  # 키가 없으면 None: 기억을 안 찾은 질문

        def ask() -> str | None:
            return vlm.ask(frame, question, memories)

        answer = collector.collect(ask) if collector is not None else ask()
        dropped = collector is not None and collector.dropped
        response = AssistantResponse(
            text="" if dropped else (answer or SORRY),
            intent=state["intent"],
            used_frame=frame,
            memories=memories or [],
        )
        update: dict = {
            "response": response,
            "filler_sent": collector is not None and collector.filler_sent,
            "interrupted": dropped,
        }
        # 버린 답은 아직 받는 중이라 last_latency_ms 가 이번 값이 아니다
        if not dropped and vlm.last_latency_ms is not None:
            update["latency_ms"] = {"vlm": vlm.last_latency_ms}
        return update

    def speak(self, state: JarviseoState) -> dict:
        tts = self.parts.tts
        asked = time.monotonic()
        tts.speak(state["response"].text)
        update: dict = {"interrupted": tts.last_interrupted}
        if tts.last_latency_ms is not None:
            # 말이 끝나고 대답 첫 소리가 나오기까지. "잠깐만요"를 읽었어도 대답 기준으로 잰다.
            total = (asked - state["utterance"].ended_at) * 1000 + tts.last_latency_ms
            update["latency_ms"] = {"tts": tts.last_latency_ms, "total": total}
        return update

    def log(self, state: JarviseoState) -> dict:
        store, session_id = self.parts.store, state.get("session_id")
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
        scenes, frame, user_id = self.parts.scenes, state.get("frame"), state.get("user_id")
        if scenes is not None and frame is not None and user_id is not None:
            scenes.remember(user_id, turn_id, frame)  # 바로 돌아온다. 저장은 뒤에서 한다
        return {"turn_id": turn_id}


def _ms(since: float) -> float:
    """perf_counter 로 잰 since 부터 지금까지(ms)."""
    return (time.perf_counter() - since) * 1000
