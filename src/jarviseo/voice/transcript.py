"""받아쓴 글자 정리 (Transcript Check).  담당: 문태현

받아쓴 원문에서 호출어와 필러를 빼고 공백을 정리한다. 라우터는 정리된 질문만 본다.
원문은 ``turn_voice.stt_raw_text`` 용으로 따로 남긴다.
"""

from __future__ import annotations

import re

__all__ = ["MIN_LETTERS", "clean", "is_question"]

# 호출어. 힌트를 줘도 "자비스"처럼 받아써질 수 있어서 비슷한 표기도 같이 잡는다.
# "자비서야" 같은 부르는 말은 호출어로 보고, "자비스럽게"처럼 뒤에 말이 이어지면 아니다.
_WAKE = re.compile(r"(?:[자재]\s?비\s?[서스써](?:야|아|님)?|jarvis)(?![가-힣])", re.IGNORECASE)
# 말 사이에 끼는 소리. 따로 떨어진 토막일 때만 지운다("아까"의 "아"는 그대로 둔다).
_FILLER = re.compile(r"(?<!\w)(?:음+|으+음|어+|아+|에+|흠+)(?!\w)[.,!?…~]*")
_LETTER = re.compile(r"[가-힣A-Za-z0-9]")

# 질문으로 치려면 글자(한글·영문·숫자)가 이만큼은 있어야 한다. 잡음이 한 글자로
# 받아써진 것을 거른다. 되묻기는 "왼쪽 컵이요, 오른쪽 컵이요?"처럼 골라 답하게 묻으므로
# "네" 같은 한 글자 대답은 기대하지 않는다.
MIN_LETTERS = 2


def clean(text: str) -> str:
    """호출어와 필러를 빼고 공백을 정리한다.

    호출어가 있으면 그 뒤가 질문이다. "아, 들려? 자비스 저거 뭐야?" → "저거 뭐야?"
    뒤에 남는 말이 없으면("저거 뭐야, 자비서?") 호출어 앞을 쓴다.
    """
    hits = list(_WAKE.finditer(text))
    if hits:
        after = text[hits[-1].end() :]
        text = after if _LETTER.search(after) else text[: hits[0].start()]
    text = re.sub(r"\s+", " ", _FILLER.sub(" ", text)).strip()
    return text.lstrip(",.!?…~ ").rstrip(", ")


def is_question(text: str) -> bool:
    """정리한 글자가 질문으로 쓸 만한지. 비었거나 너무 짧으면 False."""
    return len(_LETTER.findall(text)) >= MIN_LETTERS
