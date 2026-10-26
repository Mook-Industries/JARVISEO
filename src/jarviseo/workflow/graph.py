"""LangGraph 그래프 조립.  담당: 최홍묵

이 파일은 노드(nodes/)를 잇기만 한다. 질문 한 번이 graph.invoke() 한 번이다.
분기 함수는 "무슨 일이 있었나"(결과 이름)만 돌려주고, 어느 노드로 갈지는 아래 매핑이 정한다.

    listen ─(no_question · wake_rejected: 호출어 2단계 확인에서 걸러짐)─▶ END
      │ └─(bad_frame)─▶ recapture ↺(still_bad) ─(gave_up)──────────────────────▶ speak
      ▼                        │(good)
    route ◀────────────────────┘
      ├─(recall)───────▶ recall ───────────────────────────────────────────────▶ generate
      ├─(needs_target)─▶ detect ─▶ resolve ⇄(ambiguous/replied) clarify ─(gave_up)▶ speak
      │                              └─(chosen)─▶ crop_target
      │                                             ├─(ingredient)─▶ check_ingredient
      │                                             └─(answer)─────▶ generate
      ├─(ingredient: 가리킬 부품 없음)───────────────────────────────▶ check_ingredient
      └─(answer)──────────────────────────────────────────────────────▶ generate

    check_ingredient ─▶ speak ─▶ log ─▶ END
    generate ─(answered)─▶ speak ─▶ log ─▶ END,  generate ─(dropped: 다시 불러 답을 버림)─▶ log
    clarify ─(called_again: 되묻는 중이나 대답을 기다리는 중에 다시 불림)─▶ log

resolve 의 chosen 에는 손 우선 크롭도 든다(가리키는 손이 있는데 손가락 선 근처에 후보가 없음).
그때 crop_target 은 고른 박스 대신 손끝 방향 사각형을 자른다.
노드가 하는 일은 nodes/ 의 각 파일 머리말에 있다.
호출어 기다리기와 끼어들기는 그래프 밖(scripts/assistant.py)에서 한다.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from jarviseo.workflow.deps import GraphDeps
from jarviseo.workflow.nodes import answer, capture, ingredient, intent, memory, pointing, voice
from jarviseo.workflow.state import JarviseoState

__all__ = ["build_graph"]

# langgraph 1.2 의 기본 재귀 한도(10007)로는 루프 버그가 있어도 사실상 안 멈춘다.
# 가장 긴 경로(재촬영 2번 + 되묻기 2번)가 14단계라 30 을 넘으면 버그다.
RECURSION_LIMIT = 30


def build_graph(deps: GraphDeps):  # -> CompiledStateGraph
    """그래프를 만들어 컴파일해서 돌려준다.

    graph.invoke({"session_id": ..., "user_id": ..., "wake_at": ..., "trigger_type": ...})
    처럼 턴 정보를 넣어 부르면, 끝난 뒤의 상태(답·지연·turn_id 등)를 돌려준다.
    부품은 상태가 아니라 partial 로 노드에 묶는다. 상태에는 이번 질문에서 바뀌는 값만 둔다.
    """

    def bind(fn: Callable[..., Any]) -> Callable[[JarviseoState], Any]:
        """(state, deps) 함수를 LangGraph 가 부르는 (state) 함수로 만든다."""
        return partial(fn, deps=deps)

    graph = StateGraph(JarviseoState)
    graph.add_node("listen", bind(voice.listen))
    graph.add_node("recapture", bind(capture.recapture))
    graph.add_node("route", _timed("route", bind(intent.route)))
    graph.add_node("recall", bind(memory.recall))
    graph.add_node("detect", _timed("detect", bind(pointing.detect)))
    graph.add_node("resolve", _timed("resolve", bind(pointing.resolve)))
    graph.add_node("clarify", bind(pointing.clarify))
    # 노드 이름은 상태 키(crop·ingredient)와 다르게 짓는다. langgraph 1.2 는 같아도 받아 주지만
    # 그림·로그에서 노드인지 값인지 헷갈린다. 그래서 crop_target · check_ingredient 다.
    graph.add_node("crop_target", _timed("crop", bind(pointing.crop_target)))
    graph.add_node("check_ingredient", _timed("ingredient", bind(ingredient.check_ingredient)))
    graph.add_node("generate", bind(answer.generate))
    graph.add_node("speak", bind(voice.speak))
    graph.add_node("log", bind(memory.log))

    graph.add_edge(START, "listen")
    # 결과 이름 → 다음 노드. 매핑에는 분기 함수가 돌려줄 수 있는 값이 전부 있어야 한다.
    graph.add_conditional_edges(
        "listen",
        bind(voice.after_listen),
        {"wake_rejected": END, "no_question": END, "bad_frame": "recapture", "heard": "route"},
    )
    graph.add_conditional_edges(
        "recapture",
        bind(capture.after_recapture),
        {"gave_up": "speak", "still_bad": "recapture", "good": "route"},
    )
    graph.add_conditional_edges(
        "route",
        bind(intent.after_route),
        {
            "recall": "recall",
            "needs_target": "detect",
            "ingredient": "check_ingredient",
            "answer": "generate",
        },
    )
    graph.add_edge("recall", "generate")
    graph.add_edge("detect", "resolve")
    graph.add_conditional_edges(
        "resolve", pointing.after_resolve, {"ambiguous": "clarify", "chosen": "crop_target"}
    )
    graph.add_conditional_edges(
        "clarify",
        pointing.after_clarify,
        {"called_again": "log", "gave_up": "speak", "replied": "resolve"},
    )
    graph.add_conditional_edges(
        "crop_target", intent.after_crop, {"ingredient": "check_ingredient", "answer": "generate"}
    )
    graph.add_edge("check_ingredient", "speak")
    graph.add_conditional_edges(
        "generate", answer.after_generate, {"dropped": "log", "answered": "speak"}
    )
    graph.add_edge("speak", "log")
    graph.add_edge("log", END)
    return graph.compile().with_config(recursion_limit=RECURSION_LIMIT)


def _timed(stage: str, node: Callable[[JarviseoState], dict]) -> Callable[[JarviseoState], dict]:
    """노드에 걸린 시간을 재서 {"latency_ms": {stage: ms}} 를 결과에 붙인다.

    사람이 말하거나 듣는 시간이 안 섞이는 노드(route·detect·resolve·crop·ingredient)에만
    씌운다. 듣기·기억 검색·답 생성·읽기는 벽시계 시간에 말하는 시간이나 "잠깐만요" 재생이
    섞여서, 부품이 직접 잰 값(last_latency_ms)을 노드가 넣는다.
    """

    def timed(state: JarviseoState) -> dict:
        started = time.perf_counter()
        update = node(state)
        return {**update, "latency_ms": {stage: (time.perf_counter() - started) * 1000}}

    return timed
