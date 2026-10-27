"""기억 질문의 때 낱말("아까", "어제")을 찾을 기록의 시각 범위로 바꾼다.  담당: 문태현

임베딩 검색은 뜻만 보고 때는 보지 않는다. "아까 본 지갑 기억나?"에 6일 전 테스트 때 본
지갑 기록이 1등으로 찾혀 "네, 10월 14일에 … 기억납니다"라고 답했다(#132).
그래서 질문에 때 낱말이 있으면 그 범위 안의 기록만 찾는다.

    아까 · 방금 · 조금 전 · 좀 전   오늘 0시부터 (0시를 갓 넘겼으면 3시간 전부터)
    오늘                           오늘 0시부터
    어제 · 어젯밤                   어제 0시 ~ 오늘 0시
    그저께 · 그제                   그저께 0시 ~ 어제 0시

하루는 이 컴퓨터의 시간대(now 를 넘기면 그 시간대)로 나눈다.
돌려주는 시각은 UTC 다(observation.observed_at 과 같다).
때 낱말이 없으면 None 이다. 그때는 전부 찾는다.

    time_range("아까 본 지갑 기억나?")   # (오늘 0시, None)
    time_range("지갑 어디 뒀더라?")      # None

지난 대화 자체를 물으면("방금 물어봤던 거 뭐지?", "아까 뭐라고 했지?") 물건 이름이 없어
뜻으로는 기록을 못 찾는다(점수 0.13~0.28). 그때는 recent_dialog_range 가 범위를 주고,
기억 검색이 그 범위의 최근 턴을 시간순으로 붙인다(memory/vector.py).

    recent_dialog_range("방금 물어봤던 거 뭐지?")    # (오늘 0시, None)
    recent_dialog_range("마지막으로 뭐라고 했지?")    # 때 낱말이 없어도 "아까"와 같은 범위
    recent_dialog_range("아까 본 지갑 기억나?")       # None — 물건을 물었다
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

__all__ = ["RECENT_MIN_HOURS", "recent_dialog_range", "time_range"]

# "아까"는 보통 몇 시간 안이다. 0시 10분에 "아까"를 물으면 오늘 기록이 거의 없으니
# 오늘 0시와 이만큼 전 중 이른 쪽부터 찾는다.
RECENT_MIN_HOURS = 3

# 앞에서부터 보고 처음 맞는 것을 쓴다. "아까워"·"아까운"은 때가 아니다
_TWO_DAYS_AGO = re.compile(r"그저께|그제(?!서야)")
_YESTERDAY = re.compile(r"어제|어젯")
_TODAY = re.compile(r"오늘")
_RECENT = re.compile(r"아까(?![워운우웠])|방금|조금\s?전|좀\s?전")
# 지난 대화 자체를 묻는 말, 또는 이름 없이 가리키는 말("방금 그거 뭐였지?").
# 질문 · 얘기 · 대답은 지난 것만("방금 질문 하나 할게"는 아니다)
_ASKS_DIALOG = re.compile(
    r"물어[보봤본]|물었|말했|말한|말하던|뭐라고|뭐라\s?했"
    r"|(질문|얘기|이야기|대답)\s?(했|한|하던)|그거|그게|그걸|그것"
)
# 때 낱말 없이 직전을 가리키는 말. "아까"와 같은 범위로 본다
_LAST = re.compile(r"마지막|직전")


def time_range(query: str, now: datetime | None = None) -> tuple[datetime, datetime | None] | None:
    """(시작, 끝) — 시작 이상, 끝 미만인 기록만 찾는다. 끝이 None 이면 지금까지.

    now 는 테스트에서 시각을 정할 때만 넘긴다(시간대가 있는 datetime). 없으면 지금 이 컴퓨터 시각.
    """
    now = now or datetime.now().astimezone()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if _TWO_DAYS_AGO.search(query):
        start, end = today - timedelta(days=2), today - timedelta(days=1)
    elif _YESTERDAY.search(query):
        start, end = today - timedelta(days=1), today
    elif _TODAY.search(query):
        start, end = today, None
    elif _RECENT.search(query):
        start, end = min(today, now - timedelta(hours=RECENT_MIN_HOURS)), None
    else:
        return None
    return start.astimezone(UTC), end.astimezone(UTC) if end is not None else None


def recent_dialog_range(
    query: str, now: datetime | None = None
) -> tuple[datetime, datetime | None] | None:
    """지난 대화 자체를 물었으면 최근 턴을 찾을 범위(time_range 와 같은 모양). 아니면 None.

    대화를 묻는 말(물어봤 · 뭐라고 · 그거 …)과 때가 같이 있어야 한다. 때는 때 낱말이나
    "마지막" · "직전"("아까"와 같은 범위)이다. 물건만 물으면("아까 본 지갑") 뜻으로 찾으면 된다.
    """
    if not _ASKS_DIALOG.search(query):
        return None
    window = time_range(query, now)
    if window is None and _LAST.search(query):
        window = time_range("아까", now)
    return window
