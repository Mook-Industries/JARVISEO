"""그래프를 흐르는 상태.  담당: A (팀장)

노드들이 이 딕셔너리 하나를 계속 넘겨받으며 채워나간다.
types.py 가 '모듈 사이의 계약'이라면, 이 파일은 '노드 사이의 계약'이다.

필드를 추가할 때는 types.py 와 같은 규칙을 따른다.
커밋 메시지 제목에 [contract] 를 붙이고 팀에 공지한다.
"""

from __future__ import annotations

from typing import TypedDict

from jarviseo.types import (
    AllergenJudgement,
    AssistantResponse,
    Detection,
    Frame,
    Intent,
    MemoryHit,
    TargetResolution,
    Utterance,
)


class JarviseoState(TypedDict, total=False):
    """그래프 전체를 흐르는 상태.

    total=False 이므로 모든 키가 선택적이다.
    노드는 자기가 채울 키만 반환하면 된다.
    """

    # --- 입력 ---
    utterance: Utterance
    frame: Frame

    # --- 라우팅 ---
    intent: Intent

    # --- 검출 결과 ---
    detections: list[Detection]
    fingertip: Detection | None
    gaze_vector: tuple[float, float] | None

    # --- 특화 모듈 결과 ---
    target: TargetResolution | None
    allergen: AllergenJudgement | None
    memories: list[MemoryHit]

    # --- 출력 ---
    response: AssistantResponse

    # --- 제어 ---
    # 프레임이 흐려서 다시 찍은 횟수. 무한 루프를 막으려고 센다.
    recapture_count: int
    # 되물은 횟수. 두 번까지만 되묻고 그 뒤에는 1위 후보로 그냥 답한다.
    clarify_count: int
    # 단계별 소요 시간(ms). 대시보드 지연 그래프의 원본.
    latency_ms: dict[str, float]

    error: str | None


# 되풀이 상한. 이걸 안 걸면 흐린 장면 앞에서 영원히 다시 찍는다.
MAX_RECAPTURE = 3
MAX_CLARIFY = 2
