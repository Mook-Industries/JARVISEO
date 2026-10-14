"""그래프 상태의 키가 노드 사이에서 맞게 넘어가는지 확인한다."""

from langgraph.graph import END, START, StateGraph

from jarviseo.graph.state import JarviseoState


def test_노드마다_돌려준_지연은_합쳐진다():
    graph = StateGraph(JarviseoState)
    graph.add_node("stt", lambda state: {"latency_ms": {"stt": 800.0}})
    graph.add_node("tts", lambda state: {"latency_ms": {"tts": 950.0}})
    graph.add_edge(START, "stt")
    graph.add_edge("stt", "tts")
    graph.add_edge("tts", END)

    result = graph.compile().invoke({"session_id": 1})

    # 합치지 않으면 마지막 노드의 {"tts": 950.0} 만 남는다
    assert result["latency_ms"] == {"stt": 800.0, "tts": 950.0}
    assert result["session_id"] == 1  # 노드가 안 건드린 키는 그대로 남는다
