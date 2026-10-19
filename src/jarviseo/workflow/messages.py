"""그래프가 미리 정해 둔 안내 문장.  담당: 최홍묵

여러 노드가 같은 문장을 쓰고, 테스트도 이 상수로 "이 문장으로 끝났는가"를 본다.
"""

from __future__ import annotations

from jarviseo.types import AssistantResponse, Intent
from jarviseo.workflow.state import JarviseoState

__all__ = [
    "CANNED",
    "CLARIFY_GIVE_UP",
    "INGREDIENT_UNAVAILABLE",
    "RECAPTURE_GIVE_UP",
    "REPROMPT",
    "SORRY",
    "VLM_TOO_SLOW",
    "canned_response",
]

# 호출어만 부르고 멈췄을 때 되묻는 말
REPROMPT = "네, 말씀하세요."
SORRY = "죄송해요, 지금은 대답을 못 드려요. 잠시 뒤에 다시 물어봐 주세요."
# VLM 이 GraphDeps.vlm_deadline_sec 안에 답하지 않았을 때. 실패(SORRY)와 기록에서 나눠 센다
VLM_TOO_SLOW = "답이 늦어지고 있어요. 잠시 뒤에 다시 물어봐 주세요."
RECAPTURE_GIVE_UP = "화면이 흐려서 잘 안 보여요. 다시 바라보며 불러 주세요."  # 스토리보드 S-06
CLARIFY_GIVE_UP = "어느 것을 말씀하시는지 모르겠어요. 다시 가리켜 주세요."  # 가리킴 흐름도
INGREDIENT_UNAVAILABLE = "지금은 성분을 확인할 수 없어요. 포장의 원재료명을 직접 확인해 주세요."
# 새 내용이 없어서 대화 기억에 넣지 않는 문장들
CANNED = frozenset(
    {SORRY, VLM_TOO_SLOW, RECAPTURE_GIVE_UP, CLARIFY_GIVE_UP, INGREDIENT_UNAVAILABLE}
)


def canned_response(state: JarviseoState, text: str) -> AssistantResponse:
    """미리 정한 안내를 답으로 만든다(재촬영·되묻기 포기, 성분 확인 불가).

    재촬영 포기는 의도 분류 전이라 의도가 없을 수 있다.
    """
    target = state.get("target")
    return AssistantResponse(
        text=text,
        intent=state.get("intent", Intent.GENERAL),
        used_frame=state.get("frame"),
        target=target,
        # 되묻다 끝난 턴은 아무도 대상을 고르지 않았다. log_turn 이 resolved_by 를 비운다.
        needs_clarify=target is not None and target.needs_clarify,
    )
