"""기억 검색 · 기록 노드.  담당: 최홍묵 (부르는 모듈: memory/ 문태현)

recall : 기억을 찾는다 (memory/vector.py)
log    : 턴·성분 판정을 DB 에 남기고 질문·대답을 기억에 맡긴다 (memory/store.py, dialog.py)
"""

from __future__ import annotations

from dataclasses import replace

from jarviseo.types import AssistantResponse, Intent
from jarviseo.workflow.deps import GraphDeps
from jarviseo.workflow.follow_up import asked
from jarviseo.workflow.messages import CANNED
from jarviseo.workflow.state import JarviseoState

__all__ = ["log", "recall"]


def recall(state: JarviseoState, deps: GraphDeps) -> dict:
    memory, user_id = deps.memory, state.get("user_id")
    if memory is None or user_id is None:
        return {"memories": []}
    update: dict = {"memories": memory.search(user_id, asked(state))}
    if memory.last_latency_ms is not None:
        update["latency_ms"] = {"recall": memory.last_latency_ms}
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
        # 바로 돌아온다. 임베딩·저장은 뒤에서 한다. 이어 들은 대답("다시 확인해줘")은 앞 질문을
        # 붙인 글로 남겨야 나중에 무엇을 물었는지 찾힌다.
        dialog.remember(user_id, turn_id, asked(state), response.text)
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
