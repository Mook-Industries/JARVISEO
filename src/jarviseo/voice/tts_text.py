"""TTS 에 넣을 글자 다듬기.  담당: 문태현

답변 문장을 소리 내 읽기 좋게 바꾼다. 숫자 뒤 단위는 우리말로 풀고("500mg" → "500밀리그램"),
천 단위 쉼표와 범위 물결표는 읽는 말로 바꾸고, 마크다운 기호는 뺀다.
숫자 자체는 TTS 가 잘 읽으므로 그대로 둔다.
"""

from __future__ import annotations

import re

__all__ = ["normalize"]

# 성분표·안내에 자주 나오는 단위. 숫자 바로 뒤에 올 때만 바꾼다("gram" 같은 영어는 그대로).
_UNITS = {
    "kcal": "킬로칼로리",
    "mg": "밀리그램",
    "kg": "킬로그램",
    "g": "그램",
    "ml": "밀리리터",
    "mL": "밀리리터",
    "L": "리터",
    "l": "리터",
    "km": "킬로미터",
    "cm": "센티미터",
    "mm": "밀리미터",
    "m": "미터",
    "%": "퍼센트",
    "°C": "도",
    "℃": "도",
}
_UNIT = re.compile(
    r"(\d)\s?(" + "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True)) + r")(?![A-Za-z])"
)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")  # 1,200 → 1200
_RANGE = re.compile(r"(\d)\s?~\s?(\d)")  # 3~5 → 3에서 5
_MARKDOWN = re.compile(r"[*_`#>]+|^[ \t]*[-•][ \t]+", re.MULTILINE)


def normalize(text: str) -> str:
    """단위·쉼표·물결표를 읽는 말로 바꾸고 마크다운 기호를 뺀다. 줄바꿈은 남긴다."""
    text = _MARKDOWN.sub("", text)
    text = _THOUSANDS.sub("", text)
    text = _RANGE.sub(r"\1에서 \2", text)
    text = _UNIT.sub(lambda m: m.group(1) + _UNITS[m.group(2)], text)
    return re.sub(r"[ \t]+", " ", text).strip()
