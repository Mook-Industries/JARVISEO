"""의도 라우터 (키워드 규칙).  담당: 최홍묵

받아쓴 질문을 보고 그래프의 어느 길로 보낼지 정한다. 모델을 부르지 않아 0ms 에 끝난다.
음성 워크플로의 Router Agent 규칙(키워드 사전 매칭)을 따른다.

위에서부터 먼저 걸리는 쪽으로 간다.
- INGREDIENT : 먹어도·성분·알레르기·알러지. "이거 먹어도 돼?"에 지시어도 있어서 가리킴보다 먼저 본다
- RECALL     : 지난 일을 묻는 말. "아까 본 그거"에 지시어도 있어서 가리킴보다 먼저 본다
- POINTING   : 물건 하나를 집는 말 — 지시어(이거·이건·쟤), "저기"(저기요는 빼고),
               위치 말 바로 뒤의 거(왼쪽꺼·위에 있는 거), "가리키", 아무것도 안 붙은 "뭐야?"
- GENERAL    : 그 밖의 전부. 찾기("어떤 게 X야?")·장면("뭐가 보여?")·영역("왼쪽에는 뭐가 있지?")
               질문은 VLM 이 전체 사진을 봐야 풀려서 일부러 여기 둔다

글자로 GENERAL 이어도 가리키는 손이 보이면 intent 노드가 가리킴으로 보낸다(#217). 사진을 봐야
하는 판단이라 여기서는 하지 않고, 그때 그대로 둘 장면 · 찾기 · 영역 질문만 asks_whole_scene 으로
가려 준다.
"""

from __future__ import annotations

import re

from jarviseo.types import Intent

__all__ = ["asks_whole_scene", "classify_intent"]

# 띄어쓰기를 지운 글자와 맞춰 보므로 낱말도 띄어쓰기 없이 적는다.
# "들어있"은 "나트륨 얼마나 들어 있어?"도 성분 길로 보낸다.
# 알레르기 질문이 VLM 으로 새는 것보다 낫다(VLM 은 알레르기를 지어낼 수 있다).
_INGREDIENT = ("먹어도", "성분", "알레르기", "알러지", "원재료", "들어갔", "들어있")
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
# "뭐야"는 넣지 않는다. 넣으면 "오늘 날씨 뭐야?"까지 가리킴으로 빠진다(단독 "뭐야?"는 아래 _BARE).
# "그건"은 "그건 그렇고"에 걸려 넣지 않는다.
_POINTING = (
    "저거",
    "이거",
    "그거",
    "저것",
    "이것",
    "그것",
    "저게",
    "이게",
    "그게",
    "이건",
    "저건",
    "이걸",
    "저걸",
    "쟤",
    "가리키",
)
# 위치 말 바로 뒤의 거·꺼·것 — 물건 하나를 집는다. "앞으로 할 거"·"위험한 거"는 안 걸린다.
_POSITION_ONE = re.compile(r"(왼쪽|오른쪽|가운데|위|아래|옆|앞|뒤)(에|에있는|쪽|에놓인)?(꺼|거|것)")
# "저기 있는 거"는 가리키지만 "저기요"는 부르는 말이다.
_THERE = re.compile(r"저기(?!요)")
# 앞에 아무 말도 없는 질문은 보고 있는 물건을 묻는다.
_BARE = ("뭐야", "뭐지", "뭐예요", "뭐에요", "뭐냐")
# 사진 전체를 봐야 풀리는 말. 손이 보여도 GENERAL 로 둔다(#217).
# 찾기 · 비교("어떤 게 X야?" · "가장 큰 물건") — 여러 물건을 견줘야 해서 손끝 크롭에 다 안 들어간다.
# 장면("뭐가 보여?" · "여기 어디야?") — 물건 하나가 아니라 둘러싼 장면을 묻는다.
_WHOLE_SCENE = (
    "어떤게",
    "어떤거",
    "어떤것",
    "어느게",
    "어느거",
    "어느것",
    "둘중",
    "가장",
    "제일",
    "뭐가보여",
    "뭐가있어",
    "뭐가있지",
    "뭐있어",
    "여기어디",
)
# 영역 — 위치 말 뒤에 "에는 · 엔"이 붙으면 그 자리에 있는 것들을 묻는다("왼쪽에는 뭐가 있지?").
_AREA = re.compile(r"(왼쪽|오른쪽|가운데|위|아래|옆|앞|뒤)(칸)?(에는|엔)")


def classify_intent(text: str) -> Intent:
    """질문 글자로 의도를 정한다."""
    # STT 는 "어디 뒀지"를 "어디뒀지"로 붙여 줄 때가 있어서 띄어쓰기를 지우고 본다.
    compact = "".join(text.split())
    if any(word in compact for word in _INGREDIENT):
        return Intent.INGREDIENT
    if any(word in compact for word in _RECALL):
        return Intent.RECALL
    if _points_at_one(compact):
        return Intent.POINTING
    return Intent.GENERAL


def asks_whole_scene(text: str) -> bool:
    """찾기 · 비교 · 장면 · 영역 질문처럼 사진 전체를 봐야 풀리는 말인지."""
    compact = "".join(text.split())
    return any(word in compact for word in _WHOLE_SCENE) or bool(_AREA.search(compact))


def _points_at_one(compact: str) -> bool:
    """띄어쓰기를 지운 질문이 물건 하나를 집는 말인지."""
    if any(word in compact for word in _POINTING):
        return True
    if _POSITION_ONE.search(compact) or _THERE.search(compact):
        return True
    return compact.rstrip("?!.~") in _BARE
