"""LangGraph 그래프 조립.  담당: 최홍묵

이 파일은 노드(nodes.py)와 분기(edges.py)를 잇기만 한다. 질문 한 번이 graph.invoke() 한 번이다.

    listen ─(질문 없음)─▶ END
      │ └─(사진 없음·흐림)─▶ recapture ↺ ─(포기)─────────────────────▶ speak
      ▼                         │
    route ◀─────────────────────┘
      ├─(RECALL)───────────────▶ recall ────────────────────────────▶ generate
      ├─(POINTING·INGREDIENT)──▶ detect ─▶ resolve ⇄ clarify ─(포기)▶ speak
      │                                      └─▶ crop_target
      │                                            ├─(INGREDIENT)───▶ check_ingredient
      │                                            └─(그 밖)────────▶ generate
      ├─(INGREDIENT, 가리킬 부품 없음)────────────────────────────────▶ check_ingredient
      └─(GENERAL)─────────────────────────────────────────────────────▶ generate

    check_ingredient ─▶ speak ─▶ log ─▶ END
    generate ─▶ speak ─▶ log ─▶ END  (기다리는 동안 다시 불러 답을 버렸으면 generate ─▶ log)

    listen           : 받아쓰고, 말을 시작한 순간 그 시각 근처에서 가장 선명한 사진을 고른다
    recapture        : 버퍼에서 다시 고른다. 2번 넘으면 안내하고 끝낸다
    route            : 의도를 정한다 (router.py, 키워드 규칙)
    recall           : 기억을 찾는다 (memory/vector.py)
    detect           : 물체 후보를 찾는다 (pointing/detector.py)
    resolve          : 후보 중 가리킨 대상을 고른다 (pointing/resolver.py)
    clarify          : 1·2위가 비슷하면 되묻고 대답을 듣는다. 사진은 다시 고르지 않는다
    crop_target      : 고른 대상만 잘라낸다 (pointing/crop.py)
    check_ingredient : 성분을 판정한다. VLM 을 부르지 않는다 (nutrition/service.py)
    generate         : VLM 답을 받는다. 늦으면 "잠깐만요"를 먼저 읽는다 (voice/collector.py)
    speak            : 답을 읽는다 (voice/tts.py)
    log              : 턴·성분 판정을 DB 에 남기고 질문·대답을 기억에 맡긴다 (store.py, dialog.py)

호출어 기다리기와 끼어들기는 그래프 밖(scripts/assistant.py)에서 한다.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from jarviseo.graph import edges, nodes
from jarviseo.graph.nodes import GraphDeps
from jarviseo.graph.state import JarviseoState

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
    graph.add_node("listen", bind(nodes.listen))
    graph.add_node("recapture", bind(nodes.recapture))
    graph.add_node("route", _timed("route", bind(nodes.route)))
    graph.add_node("recall", bind(nodes.recall))
    graph.add_node("detect", _timed("detect", bind(nodes.detect)))
    graph.add_node("resolve", _timed("resolve", bind(nodes.resolve)))
    graph.add_node("clarify", bind(nodes.clarify))
    # 노드 이름은 상태 키(crop·ingredient)와 다르게 짓는다. langgraph 1.2 는 같아도 받아 주지만
    # 그림·로그에서 노드인지 값인지 헷갈린다. 그래서 crop_target · check_ingredient 다.
    graph.add_node("crop_target", _timed("crop", bind(nodes.crop_target)))
    graph.add_node("check_ingredient", _timed("ingredient", bind(nodes.check_ingredient)))
    graph.add_node("generate", bind(nodes.generate))
    graph.add_node("speak", bind(nodes.speak))
    graph.add_node("log", bind(nodes.log))

    graph.add_edge(START, "listen")
    # 매핑에는 분기 함수가 돌려줄 수 있는 값이 전부 있어야 한다. 빠지면 실행 중에 에러가 난다.
    graph.add_conditional_edges(
        "listen",
        bind(edges.after_listen),
        {"end": END, "recapture": "recapture", "route": "route"},
    )
    graph.add_conditional_edges(
        "recapture",
        bind(edges.after_recapture),
        {"speak": "speak", "recapture": "recapture", "route": "route"},
    )
    graph.add_conditional_edges(
        "route",
        bind(edges.after_route),
        {
            "recall": "recall",
            "detect": "detect",
            "check_ingredient": "check_ingredient",
            "generate": "generate",
        },
    )
    graph.add_edge("recall", "generate")
    graph.add_edge("detect", "resolve")
    graph.add_conditional_edges(
        "resolve", edges.after_resolve, {"clarify": "clarify", "crop_target": "crop_target"}
    )
    graph.add_conditional_edges(
        "clarify", edges.after_clarify, {"speak": "speak", "resolve": "resolve"}
    )
    graph.add_conditional_edges(
        "crop_target",
        edges.after_crop,
        {"check_ingredient": "check_ingredient", "generate": "generate"},
    )
    graph.add_edge("check_ingredient", "speak")
    graph.add_conditional_edges("generate", edges.after_generate, {"speak": "speak", "log": "log"})
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
