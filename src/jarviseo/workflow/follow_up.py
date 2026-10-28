"""되물은 뒤 호출어 없이 이어 듣기.  담당: 최홍묵 (부르는 쪽: assistant.py · api/ 문태현)

"이거 먹어도 돼?" → "원재료명 부분을 보여 주세요." 처럼 자비서가 사용자에게 무언가를 다시 해 달라고
하면, 예전에는 대답을 읽고 바로 호출어 대기로 돌아갔다. 사용자는 "자비서"를 다시 불러야 했고, 그
턴은 앞 질문을 몰라서 "다시 확인해줘"가 일반 질문(VLM)으로 갔다.

이제 듣기 루프(scripts/assistant.py · api/voice_loop.py · api/browser_voice.py)가 턴이 끝날
때마다 follow_up_turn(state) 를 묻는다. 턴 정보가 나오면 시작음을 내고 호출어 없이
FOLLOW_UP_SEC 동안 듣는 턴을 하나 더 돌린다(trigger_type CLARIFY_REPLY, ERD 의 "되묻기에 대한
대답" 턴).

    질문 "이거 먹어도 돼?"  →  대답 "원재료명 부분을 보여 주세요."  (되묻는 말 → 이어 듣기)
    이어 "다시 확인해줘"    →  question "이거 먹어도 돼? 다시 확인해줘"
                               → 성분 질문으로 새 장면을 다시 판정

이어 들은 말이 짧은 대답("여기", "왼쪽 거", "이제 보여?")이면 앞 질문 뒤에 붙여 묻는다(question).
기록 · 화면에는 사용자가 실제로 한 말(utterance)이 남는다. 새 질문("저건 뭐야?")이면 그대로 묻는다.
"""

from __future__ import annotations

import re

from jarviseo.types import Intent
from jarviseo.workflow.messages import CLARIFY_GIVE_UP, RECAPTURE_GIVE_UP
from jarviseo.workflow.router import classify_intent
from jarviseo.workflow.state import JarviseoState

__all__ = [
    "FOLLOW_UP_SEC",
    "FOLLOW_UP_TRIGGER",
    "MAX_FOLLOW_UP",
    "asked",
    "expects_reply",
    "follow_up_question",
    "follow_up_turn",
]

# 이어 듣는 턴의 trigger_type. ERD(session_turn.trigger_type)의 "되묻기에 대한 대답"이다.
FOLLOW_UP_TRIGGER = "CLARIFY_REPLY"
# 되물은 뒤 말을 시작하기를 기다리는 시간(초). 포장을 돌려 카메라에 대는 동안 말을 늦게 꺼낸다.
FOLLOW_UP_SEC = 6.0
# 잇달아 이어 듣는 최대 횟수. 성분표를 계속 못 읽어도 세 번째에는 호출어 대기로 돌아간다.
MAX_FOLLOW_UP = 2

# 사용자에게 다시 말하거나 · 보여 주거나 · 가리켜 달라는 말(VLM 대답과 성분 재촬영 안내)
_ASKS_BACK = re.compile(
    r"알려\s?주시면|말씀해\s?주|물으시는지|가리켜\s?주|다시 보여|보여\s?주세요|다시 찍어|"
    r"다시 말|무슨 말씀|못 알아"
)
# 이어 들은 말에 이런 물음말이 있으면 앞 질문에 붙이지 않는 새 질문이다
_OWN_QUESTION = re.compile(r"뭐|무엇|어디|언제|누구|누가|몇|얼마|어때|어떻|왜")


def asked(state: JarviseoState) -> str:
    """이 턴에 실제로 묻는 글. 이어 들은 짧은 대답이면 앞 질문을 붙인 글(question)이다."""
    return state.get("question") or state["utterance"].text


def expects_reply(state: JarviseoState) -> bool:
    """자비서가 읽은 대답이 사용자에게 무언가를 다시 해 달라는 말인지."""
    response = state.get("response")
    if state.get("utterance") is None or response is None or not response.text:
        return False
    check = state.get("ingredient")
    if check is not None and check.retake:
        return True  # 성분표를 못 읽어 다시 보여 달라고 했다
    if response.text in (RECAPTURE_GIVE_UP, CLARIFY_GIVE_UP):
        return True
    # VLM 이 되물었다("어떤 물건을 말씀하시는지 알려 주시면…", "…무엇을 찾으세요?")
    return bool(_ASKS_BACK.search(response.text)) or response.text.rstrip().endswith("?")


def follow_up_turn(state: JarviseoState) -> dict | None:
    """state 턴 뒤에 호출어 없이 이어 들을 턴 정보. 이어 듣지 않으면 None.

    읽던 대답이 끊겼으면(끼어들기) 그 호출이 다음 턴이라 이어 듣지 않는다.
    """
    count = state.get("follow_up_count", 0)
    if count >= MAX_FOLLOW_UP or state.get("interrupted") or not expects_reply(state):
        return None
    return {
        "session_id": state.get("session_id"),
        "user_id": state.get("user_id"),
        "trigger_type": FOLLOW_UP_TRIGGER,
        "follow_up_of": asked(state),
        "follow_up_count": count + 1,
    }


def follow_up_question(previous: str, reply: str) -> str:
    """이어 들은 말로 물을 글. 짧은 대답이면 앞 질문 뒤에 붙이고, 새 질문이면 그대로 둔다.

    성분 · 기억 질문이거나 물음말("뭐", "어디" …)이 있으면 새 질문이다. "다시 확인해줘", "여기",
    "왼쪽 거", "이제 보여?" 는 앞 질문에 대한 대답이다.
    """
    if classify_intent(reply) in (Intent.INGREDIENT, Intent.RECALL) or _OWN_QUESTION.search(reply):
        return reply
    return f"{previous} {reply}"
