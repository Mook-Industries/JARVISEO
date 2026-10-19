"""그래프 상태의 리듀서가 노드 결과를 덮어쓰지 않고 합치는지 확인한다."""

from langgraph.graph import END, START, StateGraph

from jarviseo.workflow.state import JarviseoState, merge_latency


def test_different_stages_are_kept_side_by_side():
    """다른 단계는 나란히 남는다."""
    assert merge_latency({"stt": 800.0}, {"vlm": 1200.0}) == {"stt": 800.0, "vlm": 1200.0}


def test_repeated_stage_adds_time():
    """같은 단계가 또 돌면 시간을 더한다."""
    # 되묻고 다시 고르면 resolve 가 두 번 돈다. 그 단계에 쓴 시간은 둘을 더한 값이다.
    assert merge_latency({"resolve": 3.0}, {"resolve": 2.0}) == {"resolve": 5.0}


def test_original_dict_is_not_mutated():
    """원래 딕셔너리는 바꾸지 않는다."""
    old = {"stt": 800.0}
    merge_latency(old, {"stt": 1.0})
    assert old == {"stt": 800.0}


def test_latency_and_replies_merge_in_graph():
    """그래프 안에서도 지연과 대답이 합쳐진다."""
    graph = StateGraph(JarviseoState)
    graph.add_node("a", lambda state: {"latency_ms": {"stt": 800.0}, "clarify_replies": ["왼쪽"]})
    graph.add_node("b", lambda state: {"latency_ms": {"stt": 200.0}, "clarify_replies": ["거"]})
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)

    result = graph.compile().invoke({"session_id": 1})

    # 리듀서가 없으면 마지막 노드의 값만 남는다
    assert result["latency_ms"] == {"stt": 1000.0}
    assert result["clarify_replies"] == ["왼쪽", "거"]
    assert result["session_id"] == 1  # 노드가 안 건드린 키는 그대로 남는다
