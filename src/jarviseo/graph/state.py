"""그래프를 흐르는 상태.  담당: 최홍묵

노드들이 이 딕셔너리 하나를 계속 넘겨받으며 채워나간다.
types.py 가 '모듈 사이의 계약'이라면, 이 파일은 '노드 사이의 계약'이다.

필드를 추가할 때는 types.py 와 같은 규칙을 따른다.
커밋 타입 뒤에 ! 를 붙이고(feat(state)!: ...) 팀에 공지한다.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

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

    # --- 턴 (그래프를 부르는 쪽이 넣는다) ---
    session_id: int
    user_id: int
    # WAKEWORD(호출어) / BARGE_IN(재생 중 끼어들기). session_turn.trigger_type 에 남는다.
    trigger_type: str
    # 호출어가 끝난 시각(monotonic). listen(since=...) 이 이보다 조금 앞부터 읽어서
    # 호출어에 바로 이어 말한 질문이 잘리지 않는다. 없으면 지금부터 듣는다.
    wake_at: float

    # --- 라우팅 ---
    intent: Intent

    # --- 검출 결과 ---
    detections: list[Detection]
    fingertip: Detection | None
    gaze_vector: tuple[float, float] | None

    # --- 특화 모듈 결과 ---
    target: TargetResolution | None
    allergen: AllergenJudgement | None
    # 기억 검색(RECALL) 결과. 키가 없으면 검색을 안 한 질문, 빈 목록이면 찾았는데 없던 것이다.
    memories: list[MemoryHit]

    # --- 출력 ---
    response: AssistantResponse

    # --- 턴 기록 (log 노드가 log_turn 에 넘긴다) ---
    # 호출어·필러를 빼기 전 받아쓴 원문 (turn_voice.stt_raw_text)
    stt_raw_text: str | None
    # 답이 늦어 "잠깐만요"를 먼저 읽었으면 True (turn_voice.is_filler_sent)
    filler_sent: bool
    # 읽던 답이 끼어들기로 끊겼거나, 답을 기다리는 동안 다시 불러서 답을 버렸으면 True
    # (turn_voice.is_interrupted). 그러면 다음 턴은 BARGE_IN 이다.
    interrupted: bool
    turn_id: int

    # --- 제어 ---
    # 프레임이 흐려서 다시 찍은 횟수. 무한 루프를 막으려고 센다.
    recapture_count: int
    # 되물은 횟수. 두 번까지만 되묻고 그 뒤에는 1위 후보로 그냥 답한다.
    clarify_count: int
    # 단계별 소요 시간(ms). 대시보드 지연 그래프의 원본.
    # 노드마다 자기 단계만 {"stt": 812.0} 처럼 돌려주면 LangGraph 가 operator.or_ 로
    # 앞의 값과 합친다. 이게 없으면 나중 노드가 앞 노드의 값을 통째로 덮어쓴다.
    latency_ms: Annotated[dict[str, float], operator.or_]

    error: str | None


# 되풀이 상한. 이걸 안 걸면 흐린 장면 앞에서 영원히 다시 찍는다.
# 값은 AI 워크플로(가리킴 모델)의 "재촬영 요청 최대 2회"를 따른다.
MAX_RECAPTURE = 2
MAX_CLARIFY = 2
