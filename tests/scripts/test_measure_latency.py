"""단계별 지연 리포트 — 세션 고르기 · 길 나누기 · 끊긴 턴 빼기 · 턴 수 경고.

store fixture 로 SQLite 와 실제 Postgres(테스트용 DB) 둘 다에서 돈다. 턴은 log_turn 으로
직접 만들어 넣어서, 어떤 숫자가 나와야 하는지 미리 안다.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from jarviseo.memory import MemoryStore, TurnIngredient
from jarviseo.types import AssistantResponse, Intent, Utterance

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "measure_latency.py"
_spec = importlib.util.spec_from_file_location("measure_latency", _SCRIPT)
measure_latency = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure_latency)


def _turn(
    store: MemoryStore,
    session_id: int,
    latency: dict[str, float],
    *,
    ingredient: bool = False,
    filler: bool = False,
    interrupted: bool = False,
) -> int:
    """지연이 정해진 턴 하나를 실제 기록 함수(log_turn)로 남긴다."""
    utterance = Utterance("이거 뭐야?", started_at=0.0, ended_at=1.0)
    response = AssistantResponse(text="컵이에요.", intent=Intent.GENERAL, latency_ms=latency)
    turn_id = store.log_turn(
        session_id, utterance, response, filler_sent=filler, interrupted=interrupted
    )
    if ingredient:
        with store.session() as db:
            db.add(TurnIngredient(turn_id=turn_id, verdict="contains"))
    return turn_id


@pytest.fixture
def session_id(store: MemoryStore) -> int:
    return store.start_session(store.ensure_user())


# --- 세션 고르기 ----------------------------------------------------------------


def test_latest_session_skips_empty_session(store):
    """켰다가 바로 끈 빈 세션은 건너뛰고, 턴이 있는 가장 최근 세션을 고른다."""
    user_id = store.ensure_user()
    with_turns = store.start_session(user_id)
    _turn(store, with_turns, {"total": 2000})
    store.start_session(user_id)  # 턴 없이 끝난 세션

    assert measure_latency.latest_session_with_turns(store) == with_turns


def test_latest_session_is_none_without_turns(store):
    """턴이 하나도 없으면 None 이다."""
    store.start_session(store.ensure_user())
    assert measure_latency.latest_session_with_turns(store) is None


# --- 백분위 ---------------------------------------------------------------------


def test_percentiles_use_nearest_rank():
    """p50 · p95 는 실제 있었던 값 중에서 고른다(store.latency_percentiles 와 같은 방식)."""
    values = [float(ms) for ms in range(100, 2100, 100)]  # 100 ~ 2000, 20개

    assert measure_latency.percentiles(values) == (1100.0, 2000.0)


# --- 길 나누기 ------------------------------------------------------------------


def test_total_by_path_splits_by_trace(store, session_id):
    """성분 기록이 있으면 성분, 검출 시간이 있으면 가리킴, 둘 다 없으면 나머지다."""
    _turn(store, session_id, {"detect": 100, "total": 3000})
    # 성분 질문도 가리킴을 먼저 돌아 검출 시간이 남지만, 성분으로 센다
    _turn(store, session_id, {"detect": 120, "total": 2500}, ingredient=True)
    _turn(store, session_id, {"total": 2000})

    paths = measure_latency.total_by_path(store, session_id)

    assert paths == {"pointing": [3000.0], "ingredient": [2500.0], "other": [2000.0]}


def test_interrupted_turn_is_left_out(store, session_id):
    """다시 불러 버린 턴은 대답 소리가 없어서 total 에서 뺀다."""
    _turn(store, session_id, {"total": 2000})
    # 끊긴 턴은 log_turn 이 그때까지의 단계 합(800)을 total_ms 로 채운다
    _turn(store, session_id, {"stt": 800}, interrupted=True)

    paths = measure_latency.total_by_path(store, session_id)

    assert paths["other"] == [2000.0]


def test_other_sessions_are_not_mixed(store, session_id):
    """다른 세션의 턴은 섞이지 않는다."""
    other = store.start_session(store.ensure_user())
    _turn(store, session_id, {"total": 2000})
    _turn(store, other, {"total": 9000})

    assert measure_latency.total_by_path(store, session_id)["other"] == [2000.0]


# --- 잠깐만요 비율 · 리포트 ----------------------------------------------------------


def test_filler_ratio_counts_filler_turns(store, session_id):
    """필러("잠깐만요")를 읽은 턴 수와 전체 턴 수를 센다."""
    _turn(store, session_id, {"total": 3000}, filler=True)
    _turn(store, session_id, {"total": 2000})

    assert measure_latency.filler_ratio(store, session_id) == (1, 2)


def test_report_warns_when_turns_are_few(store, session_id):
    """30턴보다 적으면 맨 위에 p95 를 믿기 어렵다고 알린다. 빈 단계는 - 로 찍는다."""
    _turn(store, session_id, {"stt": 800, "vlm": 1700, "total": 3000})

    text = measure_latency.report(store, session_id)

    assert "주의: 1턴뿐이라" in text
    assert "stt | 1 | 800 | 800" in text
    assert "detect | 0 | - | -" in text
    assert "other | 1 | 3000 | 3000" in text


def test_report_has_no_warning_with_enough_turns(store, session_id):
    """30턴 이상이면 경고가 없다."""
    for ms in range(30):
        _turn(store, session_id, {"total": 2000 + ms})

    assert "주의" not in measure_latency.report(store, session_id)


def test_main_exits_when_no_session_has_turns(tmp_path, monkeypatch):
    """턴이 있는 세션이 없으면 어느 DB 를 봤는지 알리고 끝낸다."""
    url = f"sqlite:///{(tmp_path / 'empty.db').as_posix()}"
    MemoryStore(database_url=url).init_schema()
    monkeypatch.setattr(sys, "argv", ["measure_latency.py", "--db", url])

    with pytest.raises(SystemExit, match="턴이 있는 세션이 없다"):
        measure_latency.main()
