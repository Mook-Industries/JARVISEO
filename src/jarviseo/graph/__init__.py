"""LangGraph 오케스트레이션 — 담당: 최홍묵.

질문 한 번을 그래프 한 번으로 돌린다.

    듣기(STT + 말을 시작한 순간의 사진) -> 의도 분류(키워드 규칙)
    -> 기억 검색 | 물체 검출 -> 대상 결정 -> 크롭 -> (성분 판정 | VLM 답) -> 읽기 -> 기록

되돌아가는 분기가 두 개 있다.
- recapture : 사진이 없거나 흐리면 버퍼에서 다시 고른다.
- clarify   : 지시 대상이 모호하면 되묻고 대답을 들어 다시 고른다.

이 두 분기가 있다는 것이 단순 파이프라인과의 차이이고, LangGraph 를 쓰는 이유이기도 하다.
"""

from jarviseo.graph.nodes import GraphDeps
from jarviseo.graph.pipeline import build_graph
from jarviseo.graph.state import JarviseoState

__all__ = ["GraphDeps", "JarviseoState", "build_graph"]
