"""의도 라우터 (규칙).  담당: 최홍묵 (D12 최소판: 문태현)

받아쓴 질문을 보고 어느 길로 보낼지 정한다. 모델을 부르지 않고 낱말로만 정해서 0ms 에 끝난다.

지금은 두 갈래뿐이다.
- RECALL  : 전에 본 것을 묻는 말("아까", "어디 뒀지")이면 기억을 찾은 뒤 VLM 에 넘긴다.
- GENERAL : 그 밖의 전부. 지금 보이는 사진으로 VLM 이 답한다.

POINTING("저거 뭐야")·INGREDIENT("뭐 들어갔어")는 그 노드가 그래프에 붙을 때 여기에 규칙을
더한다. 갈 곳이 없는 지금 보내면 답이 안 나가므로 GENERAL 로 둔다. 그래도 VLM 이 사진을 보고 답한다.
"내 가방 어디 있어?"처럼 지금 찾는 말도 GENERAL 이다(10/7 소지품 기능을 빼면서 정함).
"""

from __future__ import annotations

from jarviseo.types import Intent

__all__ = ["RECALL_CUES", "route"]

# 하나라도 들어 있으면 기억을 묻는 질문으로 본다. 지금 일("어디 있어")이 아니라
# 지난 일("어디 있었지", "어디 뒀지")을 묻는 말만 넣는다.
RECALL_CUES = (
    "아까",
    "조금 전",
    "예전에",
    "지난번",
    "어제",
    "봤던",
    "봤었",
    "본 거",
    "본 게",
    "기억나",
    "뒀지",
    "두었지",
    "놨지",
    "놓았지",
    "있었지",
    "있었더라",
)


def route(text: str) -> Intent:
    """질문 글자로 의도를 정한다."""
    if any(cue in text for cue in RECALL_CUES):
        return Intent.RECALL
    return Intent.GENERAL
