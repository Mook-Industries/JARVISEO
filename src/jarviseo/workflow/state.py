"""그래프를 흐르는 상태.  담당: 최홍묵

노드들이 이 딕셔너리 하나를 계속 넘겨받으며 채워 나간다.
types.py 가 '모듈 사이의 계약'이라면, 이 파일은 '노드 사이의 계약'이다.

리듀서가 붙은 키(latency_ms, clarify_replies)는 노드가 돌려준 값이 덮어쓰지 않고 합쳐진다.
필드를 바꾸거나 지울 때는 types.py 와 같은 규칙을 따른다.
커밋 타입 뒤에 ! 를 붙이고(feat(state)!: ...) 팀에 공지한다. 새 필드를 더하기만 하면 기존 노드가
깨지지 않으므로 ! 없이 올리고 공지만 한다.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

import numpy as np

from jarviseo.nutrition.service import IngredientCheck
from jarviseo.pointing.hands import HandPose
from jarviseo.types import (
    AssistantResponse,
    BBox,
    Detection,
    Frame,
    Intent,
    MemoryHit,
    TargetResolution,
    Utterance,
)

__all__ = ["MAX_CLARIFY", "MAX_RECAPTURE", "JarviseoState", "merge_latency"]

# 되풀이 상한. AI 워크플로(가리킴 모델)의 "재촬영 요청 최대 2회"를 따르고, 되묻기도 같게 둔다.
# 이걸 안 걸면 흐린 장면 앞에서 영원히 다시 고른다.
MAX_RECAPTURE = 2
MAX_CLARIFY = 2


def merge_latency(old: dict[str, float], new: dict[str, float]) -> dict[str, float]:
    """단계별 시간(ms)을 합친다. 같은 단계가 또 돌면(되묻고 다시 고르기) 시간을 더한다."""
    merged = dict(old)
    for stage, ms in new.items():
        merged[stage] = merged.get(stage, 0.0) + ms
    return merged


class JarviseoState(TypedDict, total=False):
    """그래프 전체를 흐르는 상태.

    total=False 이므로 모든 키가 선택적이다. 노드는 자기가 채운 키만 돌려준다.
    """

    # --- 턴 정보 (그래프를 부르는 쪽이 넣는다) ---
    session_id: int
    user_id: int
    # WAKEWORD(호출어) / BARGE_IN(읽는 중 끼어들기). session_turn.trigger_type 에 남는다.
    trigger_type: str
    # 호출어를 잡은 시각(monotonic). 듣기가 여기서부터 거슬러 읽는다.
    wake_at: float

    # --- 듣기 ---
    utterance: Utterance
    # 호출어·필러를 빼기 전 받아쓴 원문 (turn_voice.stt_raw_text)
    stt_raw_text: str | None
    frame: Frame | None
    # 호출어 2단계 확인(voice/wakecheck.py)에서 호출어가 아니라고 본 턴이면 True.
    # 되묻기·대답·기록 없이 끝난다. 확인에서 받아쓴 글자는 통과한 턴에도 wake_check_text 에
    # 남는다(받아쓰지 못했으면 None).
    wake_rejected: bool
    wake_check_text: str | None

    # --- 의도 · 기억 ---
    intent: Intent
    # 기억 검색 결과. 키가 없으면 검색을 안 한 질문, 빈 목록이면 찾았는데 없던 것이다.
    memories: list[MemoryHit]

    # --- 가리킴 ---
    detections: list[Detection]
    # 가리키는 손. detect 가 YOLO 와 같이 한 번 찾는다. 없으면 None(손 단서 기권).
    # 글자로 GENERAL 인 질문은 route 가 먼저 찾아, 가리키는 손이면 가리킴으로 보내며 넣는다(#217).
    hand: HandPose | None
    # 같은 세션에서 바로 전 가리킴 턴(60초 안)에 고른 대상의 검출 이름("cup"). 이어 묻기 단서(v5)가
    # "그거 씻어야 돼?"의 "그거"를 이 대상으로 읽는다. detect 가 턴마다 한 번 DB 에서 채운다.
    # 턴마다 그래프를 새로 불러 상태가 남지 않아서다. 없으면 None(이어 묻기 단서 기권).
    prev_target_label: str | None
    target: TargetResolution
    # 손 우선 크롭 사각형(픽셀). 가리키는 손이 있는데 손가락 선 근처에 후보가 없으면 resolve 가
    # 채우고, crop_target 이 고른 박스 대신 이 사각형을 자른다. None 이면 고른 박스로 자른다.
    # TargetResolution(types.py)은 세 사람의 계약이라 "박스 없는 선택"을 여기에 따로 둔다.
    hand_crop: BBox | None
    # 고른 대상만 잘라낸 사진. VLM 과 성분 판정에는 전체 프레임 대신 이것이 간다.
    crop: np.ndarray
    # 성분 판정 결과. log 노드가 store.log_ingredient 로 turn_ingredient 에 남긴다.
    ingredient: IngredientCheck

    # --- 답 ---
    response: AssistantResponse
    # 답이 늦어 "잠깐만요"를 먼저 읽었으면 True (turn_voice.is_filler_sent)
    filler_sent: bool
    # 읽던 답이 끼어들기로 끊겼거나, 기다리는 동안 다시 불러 답을 버렸거나,
    # 되묻는 중이나 되묻기 대답을 기다리는 중에 다시 불렀으면 True.
    # 그러면 다음 턴은 BARGE_IN 이다 (turn_voice.is_interrupted).
    interrupted: bool
    turn_id: int

    # --- 되풀이 제어 ---
    recapture_count: int
    clarify_count: int
    # 되묻기 대답 글자. 처음 질문(utterance)은 그대로 두고 대답만 쌓는다.
    clarify_replies: Annotated[list[str], operator.add]
    # 단계별 시간(ms). 노드는 자기 단계만 {"detect": 41.0} 처럼 돌려주고 merge_latency 가 합친다.
    # 이름은 memory/store.py 의 STAGE_COLUMNS 키에 맞춰서 log_turn 이 그대로 DB 열에 적는다.
    latency_ms: Annotated[dict[str, float], merge_latency]
