"""기억 질문의 때 낱말을 찾을 기록의 시각 범위로 바꾼다."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from jarviseo.memory.when import recent_dialog_range, time_range

KST = timezone(timedelta(hours=9))
NOW = datetime(2026, 10, 20, 15, 0, tzinfo=KST)
TODAY = datetime(2026, 10, 20, tzinfo=KST)


def kst(when: tuple[datetime, datetime | None] | None) -> tuple[datetime, datetime | None] | None:
    """돌려준 UTC 시각을 한국 시각으로 바꿔 읽기 쉽게 비교한다."""
    if when is None:
        return None
    start, end = when
    return start.astimezone(KST), end.astimezone(KST) if end is not None else None


@pytest.mark.parametrize(
    "query",
    ["아까 본 지갑 기억나?", "방금 본 거 뭐였지?", "조금 전에 뭐라고 했지?", "좀전에 본 컵"],
)
def test_recent_words_mean_today(query):
    """아까·방금·조금 전을 물으면 오늘 0시부터 찾는다."""
    assert kst(time_range(query, NOW)) == (TODAY, None)


def test_today_word_means_today():
    """오늘을 물으면 오늘 0시부터 찾는다."""
    assert kst(time_range("오늘 본 우유 뭐였지?", NOW)) == (TODAY, None)


def test_yesterday_word_means_yesterday_only():
    """어제를 물으면 어제 0시부터 오늘 0시 전까지만 찾는다."""
    assert kst(time_range("어제 본 가방 어디 있었지?", NOW)) == (TODAY - timedelta(days=1), TODAY)
    assert kst(time_range("어젯밤에 먹은 약", NOW)) == (TODAY - timedelta(days=1), TODAY)


def test_day_before_yesterday():
    """그저께를 물으면 그저께 하루만 찾는다."""
    assert kst(time_range("그저께 본 우산", NOW)) == (
        TODAY - timedelta(days=2),
        TODAY - timedelta(days=1),
    )


def test_recent_just_after_midnight_reaches_back_hours():
    """0시를 갓 넘겨 "아까"를 물으면 3시간 전부터 찾는다. 오늘 기록이 거의 없기 때문이다."""
    late = datetime(2026, 10, 21, 0, 30, tzinfo=KST)
    three_hours_ago = datetime(2026, 10, 20, 21, 30, tzinfo=KST)
    assert kst(time_range("아까 본 지갑", late)) == (three_hours_ago, None)


@pytest.mark.parametrize("query", ["지갑 어디 뒀더라?", "아까워서 못 버렸지", "아까운 우유"])
def test_no_time_word_searches_everything(query):
    """때 낱말이 없으면(아까워·아까운은 때가 아니다) 범위 없이 전부 찾는다."""
    assert time_range(query, NOW) is None


def test_returns_utc():
    """observation.observed_at 과 비교하도록 UTC 로 돌려준다."""
    start, _ = time_range("아까", NOW)
    assert start.tzinfo == UTC
    assert start == datetime(2026, 10, 19, 15, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "query",
    [
        "방금 물어봤던거 뭐지?",
        "아까 뭐라고 했지?",
        "좀 전에 내가 뭐 물어봤어?",
        "방금 그거 뭐였지?",
    ],
)
def test_recent_dialog_question_gets_recent_range(query):
    """지난 대화 자체를 아까 · 방금으로 물으면 "아까"와 같은 범위를 준다."""
    assert kst(recent_dialog_range(query, NOW)) == (TODAY, None)


@pytest.mark.parametrize("query", ["마지막으로 물어본 게 뭐였지?", "직전에 뭐라고 했어?"])
def test_last_word_counts_as_recent(query):
    """때 낱말 없이 마지막 · 직전으로 물어도 "아까"와 같은 범위다."""
    assert kst(recent_dialog_range(query, NOW)) == (TODAY, None)


def test_yesterday_dialog_question_gets_yesterday():
    """어제 대화를 물으면 어제 하루 범위다."""
    assert kst(recent_dialog_range("어제 내가 뭐 물어봤지?", NOW)) == (
        TODAY - timedelta(days=1),
        TODAY,
    )


@pytest.mark.parametrize(
    "query",
    ["아까 본 지갑 기억나?", "검은 지갑 어디 뒀더라?", "뭐라고 했지?", "아까워서 그거 못 버렸지"],
)
def test_object_or_timeless_question_is_not_recent_dialog(query):
    """물건을 묻거나(아까 본 지갑) 때가 없으면(뭐라고 했지?) 최근 턴을 붙이지 않는다."""
    assert recent_dialog_range(query, NOW) is None
