"""DB 스키마가 실제로 만들어지고 돌아가는지 확인한다.

Postgres 를 띄우지 않고 SQLite 인메모리로 돌린다.
테이블·관계·외래키가 맞는지 보는 데는 그걸로 충분하고,
CI 에서 컨테이너를 띄우지 않아도 되기 때문이다.

Postgres 에서만 터지는 문제(타입 불일치 등)는 여기서 안 잡힌다.
그건 실제로 붙여보고 확인해야 한다.

실행: pytest tests/test_db_schema.py
"""

import pytest

from jarviseo.memory.models import (
    Belonging,
    BelongingImage,
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


@pytest.fixture
def store() -> MemoryStore:
    """매 테스트마다 빈 인메모리 DB 를 준다."""
    s = MemoryStore(database_url="sqlite://")
    s.init_schema()
    return s


def test_테이블이_전부_만들어진다(store: MemoryStore):
    from sqlalchemy import inspect

    tables = set(inspect(store.engine).get_table_names())
    # 팀 ERD(ERDCloud JARVISEO)의 17개 테이블
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
        "belonging",
        "belonging_image",
        "observation",
    }
    assert tables == expected, f"빠진 것: {expected - tables} / ERD 에 없는 것: {tables - expected}"


def test_사용자와_알레르기(store: MemoryStore):
    user_id = store.ensure_user()

    # 같은 이메일로 다시 불러도 새로 만들지 않는다.
    assert store.ensure_user() == user_id

    store.set_user_allergens(user_id, ["땅콩", "우유"])
    assert set(store.get_user_allergens(user_id)) == {"땅콩", "우유"}

    # 통째로 교체된다. 빠진 것은 조회에서 사라진다.
    store.set_user_allergens(user_id, ["땅콩"])
    assert store.get_user_allergens(user_id) == ["땅콩"]


def test_알레르기는_지우지_않고_표시만_한다(store: MemoryStore):
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
        latency_ms={"stt": 120.0, "detect": 45.0, "vlm": 830.0, "tts": 210.0},
    )
    return utterance, response


def test_대화_한_건을_기록한다(store: MemoryStore):
    user_id = store.ensure_user()
    session_id = store.start_session(user_id, title="데모")
    utterance, response = _sample_response()

    turn_id = store.log_turn(
        session_id, utterance, response, image_path="data/x.jpg", pointing_variant="v2"
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
        assert voice.stt_raw_text == "저거 뭐야?"
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


def test_후보와_단서별_점수가_남는다(store: MemoryStore):
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


def test_되물은_턴은_정답_라벨로_남지_않는다(store: MemoryStore):
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


def test_지연_백분위(store: MemoryStore):
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


def test_사용자를_지우면_딸린_기록도_지워진다(store: MemoryStore):
    """cascade 설정이 실제로 걸려 있는지 확인한다."""
    user_id = store.ensure_user()
    store.set_user_allergens(user_id, ["땅콩"])
    store.start_session(user_id)

    with store.session() as db:
        db.delete(db.get(User, user_id))

    with store.session() as db:
        assert db.query(UserAllergen).count() == 0
        assert db.query(ChatSession).count() == 0


def test_소지품과_관찰_기록에_임베딩이_남는다(store: MemoryStore):
    """③④ 기능의 테이블. 임베딩은 같은 행의 VECTOR 열에 둔다."""
    user_id = store.ensure_user()
    session_id = store.start_session(user_id)
    utterance, response = _sample_response()
    turn_id = store.log_turn(session_id, utterance, response)

    with store.session() as db:
        bag = Belonging(user_id=user_id, name="검은 백팩")
        db.add(bag)
        db.flush()
        db.add(BelongingImage(belonging_id=bag.belonging_id, embedding=[0.1] * 512))
        db.add(
            Observation(
                user_id=user_id,
                turn_id=turn_id,
                belonging_id=bag.belonging_id,
                description="책상 위 검은 백팩",
                embedding=[0.2] * 1536,
            )
        )

    with store.session() as db:
        image = db.query(BelongingImage).one()
        assert len(image.embedding) == 512
        obs = db.query(Observation).one()
        assert len(obs.embedding) == 1536
        assert obs.belonging.name == "검은 백팩"

    # 사용자를 지우면 소지품·관찰 기록도 같이 지워진다
    with store.session() as db:
        db.delete(db.get(User, user_id))
    with store.session() as db:
        assert db.query(Belonging).count() == 0
        assert db.query(Observation).count() == 0
