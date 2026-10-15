"""의도 라우터 (키워드 규칙).  담당: 최홍묵

받아쓴 질문을 보고 그래프의 어느 길로 보낼지 정한다. 모델을 부르지 않아 0ms 에 끝난다.
음성 워크플로의 Router Agent 규칙(키워드 사전 매칭)을 따른다.

위에서부터 먼저 걸리는 쪽으로 간다.
- INGREDIENT : 먹어도·성분·알레르기. "이거 먹어도 돼?"에 지시어도 있어서 가리킴보다 먼저 본다
- RECALL     : 지난 일을 묻는 말. "아까 본 그거"에 지시어도 있어서 가리킴보다 먼저 본다
- POINTING   : 지시어(저거·이거·그거)
- GENERAL    : 그 밖의 전부
"""

from __future__ import annotations

from jarviseo.types import Intent

__all__ = ["classify_intent"]

# 띄어쓰기를 지운 글자와 맞춰 보므로 낱말도 띄어쓰기 없이 적는다.
_INGREDIENT = ("먹어도", "성분", "알레르기", "원재료", "들어갔")
# 지난 일만 넣는다. "어디 있어"(지금 찾기)는 10/7 소지품 기능을 빼면서 GENERAL 로 정했다.
# 기억 질문 평가셋(data/eval/recall_set.csv, 질문 30개)으로 잰다: scripts/eval_recall.py
_RECALL = (
    "아까",
    "조금전",
    "전에본",
    "예전에",
    "지난번",
    "어제",
    "봤던",
    "봤었",
    "본거",
    "본게",
    "기억나",
    "어디뒀",
    "뒀지",
    "두었지",
    "놨지",
    "놓았지",
    "있었지",
    "있었더라",
)
# "뭐야"는 넣지 않는다. 넣으면 "오늘 날씨 뭐야?"까지 가리킴으로 빠진다.
_POINTING = ("저거", "이거", "그거", "저것", "이것", "그것", "저게", "이게", "그게")


def classify_intent(text: str) -> Intent:
    """질문 글자로 의도를 정한다."""
    # STT 는 "어디 뒀지"를 "어디뒀지"로 붙여 줄 때가 있어서 띄어쓰기를 지우고 본다.
    compact = "".join(text.split())
    if any(word in compact for word in _INGREDIENT):
        return Intent.INGREDIENT
    if any(word in compact for word in _RECALL):
        return Intent.RECALL
    if any(word in compact for word in _POINTING):
        return Intent.POINTING
    return Intent.GENERAL
