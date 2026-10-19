"""그래프 노드가 부르는 부품 묶음.  담당: 최홍묵

오래 사는 객체(STT·DB 등)는 상태가 아니라 여기에 둔다. 상태에는 이번 질문에서 바뀌는 값만 둔다.
graph.build_graph 가 functools.partial 로 노드·분기 함수에 묶어 넣는다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jarviseo import config
from jarviseo.capture import FrameBuffer
from jarviseo.types import TargetResolution
from jarviseo.workflow.state import JarviseoState

if TYPE_CHECKING:
    from jarviseo.memory import DialogMemory, MemoryStore, VectorMemory
    from jarviseo.nutrition.service import IngredientChecker
    from jarviseo.voice import ResponseCollector

__all__ = ["GraphDeps", "can_point"]


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
    # VLM 답을 이만큼(초)만 기다리고, 넘으면 안내 문장으로 턴을 끝낸다. 테스트는 짧게 바꿔 끼운다.
    vlm_deadline_sec: float = config.VLM_DEADLINE_SEC


def can_point(state: JarviseoState, deps: GraphDeps) -> bool:
    """가리킴 길을 탈 수 있는지. 검출기·대상 결정·사진이 다 있어야 한다."""
    return deps.detector is not None and deps.resolve is not None and state.get("frame") is not None
