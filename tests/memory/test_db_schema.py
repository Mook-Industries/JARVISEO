"""DB 스키마가 실제로 만들어지고 돌아가는지 확인한다.

SQLite 인메모리와 실제 Postgres 에서 한 번씩 돈다(``store`` fixture, tests/conftest.py).
Postgres 쪽은 테스트용 DB 주소가 있을 때만 돌고, CI 에서는 db job 이 맡는다.
Postgres 에서만 터지는 문제(타입 불일치, pgvector 열 등)는 그쪽에서 잡힌다.

실행: pytest tests/test_db_schema.py
"""

import time
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from jarviseo.memory.models import (
    ChatSession,
    Observation,
    SessionTurn,
    TurnCandidate,
    TurnInference,
    TurnVoice,
    User,
    UserAllergen,
)
from jarviseo.memory.store import MemoryStore
from jarviseo.types import (
    AssistantResponse,
    BBox,
    CueKind,
    CueScore,
    Detection,
    Frame,
    Intent,
    TargetCandidate,
    TargetResolution,
    Utterance,
)


def test_all_tables_are_created(store: MemoryStore):
    """테이블이 전부 만들어진다."""
    from sqlalchemy import inspect

    tables = set(inspect(store.engine).get_table_names())
    # 팀 ERD(ERDCloud JARVISEO)의 15개 테이블
    expected = {
        "users",
        "user_setting",
        "allergen",
        "ingredient_synonym",
        "user_allergen",
        "chat_session",
        "session_turn",
        "turn_voice",
        "turn_inference",
        "turn_candidate",
        "product",
        "turn_ingredient",
        "eval_run",
        "eval_sample",
        "observation",
    }
    assert tables == expected, f"빠진 것: {expected - tables} / ERD 에 없는 것: {tables - expected}"


def test_user_and_allergens(store: MemoryStore):
    """사용자와 알레르기."""
    user_id = store.ensure_user()

    # 같은 이메일로 다시 불러도 새로 만들지 않는다.
    assert store.ensure_user() == user_id

    store.set_user_allergens(user_id, ["땅콩", "우유"])
    assert set(store.get_user_allergens(user_id)) == {"땅콩", "우유"}

    # 통째로 교체된다. 빠진 것은 조회에서 사라진다.
    store.set_user_allergens(user_id, ["땅콩"])
    assert store.get_user_allergens(user_id) == ["땅콩"]


def test_allergen_is_soft_deleted(store: MemoryStore):
    """소프트 삭제. 과거 판정 기록이 "그때는 등록돼 있었다"를 설명할 수 있어야 한다."""
    user_id = store.ensure_user()
    store.set_user_allergens(user_id, ["우유"])
    store.set_user_allergens(user_id, [])

    with store.session() as db:
        rows = db.query(UserAllergen).all()
        assert len(rows) == 1  # 행은 남아 있고
        assert rows[0].deleted_at is not None  # 표시만 됐다

    # 다시 등록하면 되살아난다. 새 행이 생기지 않는다.
    store.set_user_allergens(user_id, ["우유"])
    assert store.get_user_allergens(user_id) == ["우유"]
    with store.session() as db:
        assert db.query(UserAllergen).count() == 1


def _sample_response() -> tuple[Utterance, AssistantResponse]:
    """가리킴까지 끝난 턴 하나를 만든다."""
    import numpy as np

    utterance = Utterance(text="저거 뭐야?", started_at=10.0, ended_at=11.2, confidence=0.9)
    frame = Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        timestamp=10.1,
        source_id="folder/img_001.jpg",
        sharpness=142.5,
    )
    first = TargetCandidate(
        detection=Detection(label="backpack", confidence=0.91, bbox=BBox(10, 20, 110, 220)),
        cue_scores=[
            CueScore(kind=CueKind.POINT, score=0.82),
            CueScore(kind=CueKind.CENTER, score=0.30),
        ],
        total_score=0.74,
    )
    second = TargetCandidate(
        detection=Detection(label="cup", confidence=0.66, bbox=BBox(300, 20, 360, 90)),
        cue_scores=[CueScore(kind=CueKind.CENTER, score=0.25)],
        total_score=0.41,
    )
    response = AssistantResponse(
        text="저 가방은 검은색 백팩입니다.",
        intent=Intent.POINTING,
        used_frame=frame,
        target=TargetResolution(
            candidates=[first, second],
            chosen=first,
            margin=0.33,
            needs_clarify=False,
        ),
        latency_ms={"stt": 120.0, "detect": 45.0, "vlm": 830.0, "tts": 210.0, "total": 1205.0},
    )
    return utterance, response


def test_logs_one_turn(store: MemoryStore):
    """대화 한 건을 기록한다."""
    user_id = store.ensure_user()
    session_id = store.start_session(user_id, title="데모")
    utterance, response = _sample_response()

    turn_id = store.log_turn(
        session_id,
        utterance,
        response,
        image_path="data/x.jpg",
        pointing_variant="v2",
        stt_raw_text="자비서, 저거 뭐야?",
    )

    with store.session() as db:
        turn = db.get(SessionTurn, turn_id)
        assert turn.question_text == "저거 뭐야?"
        assert turn.answer_text.startswith("저 가방")
        assert turn.trigger_type == "WAKEWORD"
        # 전체 지연은 session_turn 에 남는다
        assert turn.total_ms == 1205

        # 음성 쪽은 turn_voice 로 떨어진다
        voice = db.get(TurnVoice, turn_id)
        assert voice is not None
        # 원문은 호출어를 빼기 전 그대로, 질문은 정리된 글자로 남는다
        assert voice.stt_raw_text == "자비서, 저거 뭐야?"
        assert voice.stt_ms == 120
        assert voice.tts_ms == 210

        inf = turn.inference
        assert inf is not None
        assert inf.target_label == "backpack"
        assert inf.pointing_variant == "v2"
        # 단계 이름이 ERD 의 열 이름으로 옮겨졌는지 (detect→vision, vlm→llm)
        assert inf.vision_ms == 45
        assert inf.llm_ms == 830
        # 프레임 선택이 발화 시작에서 얼마나 벌어졌는지가 남아야 한다
        assert inf.frame_offset_ms == 100
        assert float(inf.blur_score) == pytest.approx(142.5)
        assert inf.detected_count == 2


def test_turn_without_spoken_answer_leaves_total_empty(store: MemoryStore):
    """대답 소리가 안 난 턴은 total 이 비어 있다. 단계 합으로 채우지 않는다."""
    # 답을 버린 턴·되묻다 다시 불린 턴은 speak 를 안 거쳐서 total 이 없다
    session_id = store.start_session(store.ensure_user())
    utterance, response = _sample_response()
    response = replace(response, text="", latency_ms={"stt": 120.0, "detect": 45.0})

    turn_id = store.log_turn(session_id, utterance, response, interrupted=True)

    with store.session() as db:
        assert db.get(SessionTurn, turn_id).total_ms is None
        assert db.get(TurnVoice, turn_id).stt_ms == 120  # 단계 값은 그대로 남는다


def test_candidates_keep_cue_scores(store: MemoryStore):
    """ablation 결과를 해석하려면 합산 점수만으로는 부족하다."""
    session_id = store.start_session(store.ensure_user())
    utterance, response = _sample_response()
    turn_id = store.log_turn(session_id, utterance, response)

    with store.session() as db:
        rows = (
            db.query(TurnCandidate)
            .filter(TurnCandidate.turn_id == turn_id)
            .order_by(TurnCandidate.rank)
            .all()
        )
        assert [r.label for r in rows] == ["backpack", "cup"]
        assert [r.is_chosen for r in rows] == [True, False]
        # 어느 단서가 1위를 밀어올렸는지 남아 있어야 한다
        assert rows[0].cue_scores == {"point": 0.82, "center": 0.30}
        assert rows[0].bbox == {"x1": 10, "y1": 20, "x2": 110, "y2": 220}


def test_clarified_turn_is_not_labeled(store: MemoryStore):
    """resolved_by=USER 는 정답 라벨로 쓴다. 되물은 턴에 USER 가 찍히면 라벨이 오염된다."""
    session_id = store.start_session(store.ensure_user())
    utterance, response = _sample_response()

    response.needs_clarify = True
    asked = store.log_turn(session_id, utterance, response)

    response.needs_clarify = False
    answered = store.log_turn(session_id, utterance, response, trigger_type="CLARIFY_REPLY")
    plain = store.log_turn(session_id, utterance, response)

    with store.session() as db:
        assert db.get(TurnInference, asked).resolved_by is None
        assert db.get(TurnInference, answered).resolved_by == "USER"
        assert db.get(TurnInference, plain).resolved_by == "MODEL"


def _utc(t: datetime) -> datetime:
    """SQLite 는 시간대를 빼고 돌려준다. 저장할 때는 UTC 였다."""
    return t if t.tzinfo is not None else t.replace(tzinfo=UTC)


def test_voice_logs_for_barged_in_turn_and_next_turn(store: MemoryStore):
    """끼어들기로 끊긴 턴과 다음 턴의 음성 기록."""
    session_id = store.start_session(store.ensure_user())
    now = time.monotonic()
    first = Utterance(text="오늘 일정 알려 줘", started_at=now - 5.0, ended_at=now - 3.5)
    second = Utterance(text="저거 뭐야?", started_at=now - 1.2, ended_at=now)
    response = AssistantResponse(
        text="오늘은 일정이 세 개 있어요.",
        intent=Intent.GENERAL,
        latency_ms={"stt": 900.0, "tts": 1100.0},
    )

    cut = store.log_turn(session_id, first, response, interrupted=True)
    barged = store.log_turn(session_id, second, response, trigger_type="BARGE_IN")

    with store.session() as db:
        assert db.get(SessionTurn, cut).trigger_type == "WAKEWORD"
        assert db.get(SessionTurn, barged).trigger_type == "BARGE_IN"
        cut_voice, barged_voice = db.get(TurnVoice, cut), db.get(TurnVoice, barged)
        assert cut_voice.is_interrupted is True
        assert barged_voice.is_interrupted is False
        assert (cut_voice.stt_ms, cut_voice.tts_ms) == (900, 1100)

        # monotonic 시각이 실제 시각으로 옮겨진다. 윈도우 시계 해상도(15.6ms)만큼은 흔들린다
        started, ended = _utc(cut_voice.speech_started_at), _utc(cut_voice.speech_ended_at)
        assert (ended - started).total_seconds() == pytest.approx(1.5, abs=0.05)
        assert (datetime.now(UTC) - started).total_seconds() == pytest.approx(5.0, abs=1.0)
        assert _utc(barged_voice.speech_started_at) > ended


def test_latency_percentiles(store: MemoryStore):
    """지연 백분위."""
    session_id = store.start_session(store.ensure_user())

    with store.session() as db:
        for i in range(1, 101):
            turn = SessionTurn(session_id=session_id, question_text="x", total_ms=i)
            db.add(turn)
            db.flush()
            db.add(TurnInference(turn_id=turn.turn_id, llm_ms=i))

    # 1~100ms 를 넣으면 p50 은 50 언저리, p95 는 95 언저리여야 한다.
    stats = store.latency_percentiles("vlm")
    assert stats["count"] == 100
    assert 49 <= stats["p50"] <= 52
    assert 94 <= stats["p95"] <= 97

    # session_turn 에 있는 전체 지연도 같은 방식으로 집계된다
    assert store.latency_percentiles("total", session_id=session_id)["count"] == 100

    # 기록이 없는 단계는 0 을 돌려준다. 예외를 던지면 대시보드가 깨진다.
    assert store.latency_percentiles("없는단계") == {"p50": 0.0, "p95": 0.0, "count": 0}


def test_deleting_user_cascades(store: MemoryStore):
    """cascade 설정이 실제로 걸려 있는지 확인한다."""
    user_id = store.ensure_user()
    store.set_user_allergens(user_id, ["땅콩"])
    store.start_session(user_id)

    with store.session() as db:
        db.delete(db.get(User, user_id))

    with store.session() as db:
        assert db.query(UserAllergen).count() == 0
        assert db.query(ChatSession).count() == 0


def test_observation_keeps_embedding(store: MemoryStore):
    """④ 기능의 테이블. 임베딩은 같은 행의 VECTOR 열에 둔다."""
    user_id = store.ensure_user()
    session_id = store.start_session(user_id)
    utterance, response = _sample_response()
    turn_id = store.log_turn(session_id, utterance, response)

    with store.session() as db:
        db.add(
            Observation(
                user_id=user_id,
                turn_id=turn_id,
                description="책상 위 검은 백팩",
                embedding=[0.2] * 1536,
            )
        )

    with store.session() as db:
        obs = db.query(Observation).one()
        assert len(obs.embedding) == 1536

    # 사용자를 지우면 관찰 기록도 같이 지워진다
    with store.session() as db:
        db.delete(db.get(User, user_id))
    with store.session() as db:
        assert db.query(Observation).count() == 0


def test_turn_with_filler_is_marked(store: MemoryStore):
    """먼저 "잠깐만요"를 읽은 턴은 필러로 남는다."""
    session_id = store.start_session(store.ensure_user())
    utterance, response = _sample_response()

    late = store.log_turn(session_id, utterance, response, filler_sent=True)
    quick = store.log_turn(session_id, utterance, response)

    with store.session() as db:
        assert db.get(TurnVoice, late).is_filler_sent is True
        assert db.get(TurnVoice, quick).is_filler_sent is False
