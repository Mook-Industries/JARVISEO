"""LangGraph 오케스트레이션 — 담당: A (팀장).

전체 흐름을 그래프로 묶는다.

    호출어 감지 -> 발화 시작 시점 프레임 캡처 -> 로컬 STT -> 의도 라우팅
    -> (병렬) 객체 검출 / OCR / 기억 검색 -> 지시 대상 특정
    -> VLM 응답 생성 -> TTS

되돌아가는 분기가 두 개 있다.
- recapture : 프레임이 흐리면 다시 찍는다.
- clarify   : 지시 대상이 모호하면 사용자에게 되묻는다.

이 두 분기가 있다는 것이 단순 파이프라인과의 차이이고,
LangGraph 를 쓰는 이유이기도 하다.
"""

from jarviseo.graph.pipeline import build_graph
from jarviseo.graph.state import JarviseoState

__all__ = ["build_graph", "JarviseoState"]
