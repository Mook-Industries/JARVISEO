"""LangGraph 그래프 구성.  담당: 최홍묵

노드와 분기를 여기서 엮는다.
개별 노드는 각 모듈의 함수를 부르기만 하고, 로직을 여기 넣지 않는다.
그래야 팀원이 자기 모듈만 고쳐도 되고 이 파일에서 충돌이 나지 않는다.
"""

from __future__ import annotations

from jarviseo.graph.state import JarviseoState

__all__ = ["build_graph"]


def build_graph():  # -> CompiledGraph
    """그래프를 만들어 컴파일해서 반환한다.

    노드 구성 (W1 스파이크에서는 capture -> stt -> vlm -> tts 최소 4개만)

        capture   : FrameBuffer 에서 발화 시작 시각 기준 프레임을 고른다
        stt       : 발화를 받아쓴다
        route     : 의도를 판단한다 (POINTING / INGREDIENT / ... / GENERAL)
        detect    : 객체·손끝 검출          (POINTING 일 때)
        ocr       : 성분표 검출 + OCR       (INGREDIENT 일 때)
        recall    : 벡터 검색               (BELONGING / RECALL 일 때)
        resolve   : 지시 대상 특정
        generate  : VLM 응답 생성
        speak     : TTS

    조건부 분기

        capture -> recapture : 프레임이 흐리면(is_too_blurry) 되돌아간다.
                               단 recapture_count < MAX_RECAPTURE 일 때만.
        resolve -> clarify   : margin 이 임계값 미만이면 되묻는다.
                               단 clarify_count < MAX_CLARIFY 일 때만.
                               상한을 넘으면 1위 후보로 그냥 답한다.
    """
    raise NotImplementedError("W1 스파이크: 노드 2개짜리 최소 그래프부터 돌려볼 것")


def _node_capture(state: JarviseoState) -> JarviseoState:
    raise NotImplementedError


def _node_resolve(state: JarviseoState) -> JarviseoState:
    raise NotImplementedError


def _route_after_capture(state: JarviseoState) -> str:
    """ "recapture" 또는 "continue" 를 반환한다."""
    raise NotImplementedError


def _route_after_resolve(state: JarviseoState) -> str:
    """ "clarify" 또는 "generate" 를 반환한다."""
    raise NotImplementedError
