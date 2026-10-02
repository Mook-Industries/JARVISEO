"""데이터베이스 스키마 — ERD 를 코드로 옮긴 것.

**원본은 팀 ERDCloud 다이어그램 `JARVISEO-v5` 이고, 사람이 읽는 사본이
``docs/ERD.md`` 에 있다. 셋이 어긋나면 ERD 가 맞다.**
(예전 주석에는 "코드가 맞다"고 되어 있었으나, 세 사람이 같은 그림을 보고
작업해야 해서 다이어그램을 기준으로 삼기로 바꿨다.)

설계 원칙
---------
- Enum 은 네이티브 DB 타입 대신 문자열로 저장한다.
  Postgres 의 ENUM 은 값을 추가할 때마다 마이그레이션이 필요한데,
  ``trigger_type`` 이나 ``verdict`` 는 프로젝트 도중에 늘어날 것이 확실하다.
- 길이가 정해지지 않은 목록(bbox, cue_scores, metrics 등)은 JSON 으로 둔다.
  낱개로 조회할 일이 없어서 테이블을 나눌 이유가 없다.
- ``JSON`` 은 Postgres 와 SQLite 양쪽에서 동작한다. Postgres 전용 JSONB 를
  쓰면 테스트용 SQLite 로 되돌릴 수 없다. ERD 에는 JSONB 로 적혀 있지만
  운영 DB 가 Postgres 이므로 실제 컬럼은 JSONB 로 올라간다.
- ``BIGSERIAL`` 은 SQLite 에서 자동증가가 안 되므로 ``BigInteger`` 에
  sqlite 변형을 붙여 ``INTEGER`` 로 떨어지게 한다.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# BIGSERIAL 대응. SQLite 는 INTEGER PRIMARY KEY 만 자동증가한다.
BigPK = BigInteger().with_variant(Integer, "sqlite")
SmallPK = SmallInteger().with_variant(Integer, "sqlite")

# 확신도·점수류. ERD 의 NUMERIC(4,3) 과 같다.
Score = Numeric(4, 3)


class Base(DeclarativeBase):
    """모든 테이블의 부모. ``Base.metadata`` 로 스키마를 만든다."""


# ==========================================================================
# 사용자
# ==========================================================================


class User(Base):
    """사용자. 로그인 계정이자 알레르기·설정의 주인."""

    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(BigPK, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(String(60), default="")  # bcrypt. 평문 금지
    nickname: Mapped[str | None] = mapped_column(String(50))
    # USER / ADMIN. 성능 리포트 접근 제어에 쓴다.
    role: Mapped[str] = mapped_column(String(20), default="USER")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    setting: Mapped[UserSetting | None] = relationship(
        back_populates="user", cascade="all, delete-orphan", uselist=False
    )
    allergens: Mapped[list[UserAllergen]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    sessions: Mapped[list[ChatSession]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class UserSetting(Base):
    """사용자당 1행. 음성 취향과 세션 정책.

    PK 를 FK 로 그대로 쓴다(식별 관계). 사용자 1명에 설정은 1개뿐이라
    별도 id 를 두면 "설정이 두 개인 사용자"가 만들어질 수 있다.
    """

    __tablename__ = "user_setting"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True
    )
    tts_volume: Mapped[int] = mapped_column(SmallInteger, default=70)
    tts_instructions: Mapped[str] = mapped_column(Text, default="")  # 말투·속도 지시문
    tts_voice: Mapped[str] = mapped_column(String(50), default="ko-KR-SunHiNeural")
    # 무응답이 이 분수만큼 이어지면 세션을 닫는다.
    session_timeout_min: Mapped[int] = mapped_column(SmallInteger, default=10)
    hud_animation: Mapped[bool] = mapped_column(Boolean, default=True)
    show_detection_box: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped[User] = relationship(back_populates="setting")


class Allergen(Base):
    """알레르기 마스터. "우유", "땅콩" 같은 표준 이름."""

    __tablename__ = "allergen"

    allergen_id: Mapped[int] = mapped_column(SmallPK, primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    category: Mapped[str] = mapped_column(String(20), default="")

    synonyms: Mapped[list[IngredientSynonym]] = relationship(
        back_populates="allergen", cascade="all, delete-orphan"
    )


class IngredientSynonym(Base):
    """성분표 표기 → 알레르겐 매핑. "탈지분유" → "우유".

    **성분 판정 recall 이 이 표의 품질에 달려 있다.** 성분표에 적힌 말과
    알레르겐 이름이 다르기 때문에, 이 사전이 비면 아무것도 못 걸러낸다.
    """

    __tablename__ = "ingredient_synonym"

    synonym_id: Mapped[int] = mapped_column(BigPK, primary_key=True)
    allergen_id: Mapped[int] = mapped_column(
        SmallInteger, ForeignKey("allergen.allergen_id", ondelete="CASCADE"), index=True
    )
    alias: Mapped[str] = mapped_column(String(100), unique=True)

    allergen: Mapped[Allergen] = relationship(back_populates="synonyms")


class UserAllergen(Base):
    """사용자가 등록한 알레르기 하나.

    지우지 않고 ``deleted_at`` 으로 표시만 한다. 실수로 지운 뒤 다시 켰을 때
    "언제부터 언제까지 등록돼 있었나"가 남아 있어야 판정 기록을 설명할 수 있다.
    살아 있는 행에만 유니크를 걸어서, 같은 알레르겐을 두 번 등록하는 것은 막되
    지웠다가 다시 등록하는 것은 되게 한다.
    """

    __tablename__ = "user_allergen"
    __table_args__ = (
        Index(
            "uq_user_allergen_alive",
            "user_id",
            "allergen_id",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
    )

    user_allergen_id: Mapped[int] = mapped_column(BigPK, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    allergen_id: Mapped[int] = mapped_column(
        SmallInteger, ForeignKey("allergen.allergen_id"), index=True
    )
    # true 면 미량·교차오염 의심까지 경고한다. 같은 제조 시설 사용 문구 등.
    is_strict: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    # 소프트 삭제 시각. NULL 이면 사용 중.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="allergens")
    allergen: Mapped[Allergen] = relationship()


# ==========================================================================
# 대화
# ==========================================================================


class ChatSession(Base):
    """앱을 켜고 끌 때까지의 한 구간. 턴을 묶는 단위다.

    벤치마크를 "어느 실행에서 잰 숫자인가"로 묶을 때도 이 단위를 쓴다.
    """

    __tablename__ = "chat_session"

    session_id: Mapped[int] = mapped_column(BigPK, primary_key=True)
    # 비로그인 사용도 가능하므로 NULL 을 허용한다.
    user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("users.user_id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str | None] = mapped_column(String(200))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # TIMEOUT(무응답) / USER(사용자 종료) / APP_CLOSE(앱 종료)
    end_reason: Mapped[str | None] = mapped_column(String(20))

    user: Mapped[User | None] = relationship(back_populates="sessions")
    turns: Mapped[list[SessionTurn]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class SessionTurn(Base):
    """질문 한 번. 이 프로젝트에서 가장 중요한 테이블이다.

    "자비서, 저거 뭐야?" 부터 음성 답변까지가 한 행이다.
    대시보드의 채팅 이력, 지연 그래프, 정확도 측정이 전부 여기서 갈라져 나온다.

    이미지는 **질문 시점 1장만** 경로로 남긴다. 영상이나 연속 사진을 저장하면
    개인정보 문제가 생기고, 1080p 한 장이 수백 KB 라 DB 가 금방 무거워진다.
    """

    __tablename__ = "session_turn"

    turn_id: Mapped[int] = mapped_column(BigPK, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("chat_session.session_id", ondelete="CASCADE"), index=True
    )
    turn_no: Mapped[int] = mapped_column(Integer, default=1)
    asked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    # WAKEWORD(호출어) / BARGE_IN(재생 중 끼어들기) / CLARIFY_REPLY(되묻기에 대한 대답)
    trigger_type: Mapped[str] = mapped_column(String(20), default="WAKEWORD")

    question_text: Mapped[str] = mapped_column(Text, default="")
    answer_text: Mapped[str | None] = mapped_column(Text)

    image_path: Mapped[str | None] = mapped_column(String(500))
    ocr_text: Mapped[str | None] = mapped_column(Text)

    speech_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    speech_ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stt_raw_text: Mapped[str | None] = mapped_column(Text)
    stt_confidence: Mapped[Decimal | None] = mapped_column(Score)

    is_interrupted: Mapped[bool | None] = mapped_column(Boolean, default=False)
    is_filler_sent: Mapped[bool | None] = mapped_column(Boolean, default=False)
    # 개발·평가 모드에서만 저장한다(STT 평가셋용). 운영에서는 NULL.
    audio_path: Mapped[str | None] = mapped_column(String(500))

    session: Mapped[ChatSession] = relationship(back_populates="turns")
    inference: Mapped[TurnInference | None] = relationship(
        back_populates="turn", cascade="all, delete-orphan", uselist=False
    )
    candidates: Mapped[list[TurnCandidate]] = relationship(
        back_populates="turn", cascade="all, delete-orphan"
    )
    ingredient: Mapped[TurnIngredient | None] = relationship(
        back_populates="turn", cascade="all, delete-orphan", uselist=False
    )


# ==========================================================================
# ① 지시 대상 특정  (담당: 최홍묵)
# ==========================================================================


class TurnInference(Base):
    """한 턴의 추론 결과와 단계별 지연.  턴과 1:1.

    PK 를 FK 로 그대로 쓴다. 턴 하나에 추론 결과는 하나뿐이다.
    지연을 열로 두는 이유는 단계가 6개로 고정돼 있고, 한 턴의 전체 흐름을
    한 행으로 읽는 편이 대시보드 질의가 단순하기 때문이다.
    """

    __tablename__ = "turn_inference"

    turn_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("session_turn.turn_id", ondelete="CASCADE"), primary_key=True
    )

    target_label: Mapped[str | None] = mapped_column(String(50))
    confidence: Mapped[Decimal | None] = mapped_column(Score)
    # 1위 - 2위 점수차. 임계값 미만이면 답하지 않고 되묻는다.
    margin: Mapped[Decimal | None] = mapped_column(Score)
    # 가리킴 로직 버전(v1~v5). 운영 데이터로 버전을 비교하려면 필수.
    pointing_variant: Mapped[str | None] = mapped_column(String(20), index=True)
    # 손끝 좌표(0~1 정규화). 벡터 재계산·디버깅용.
    fingertip_xy: Mapped[dict | None] = mapped_column(JSON)
    # Laplacian variance. 재촬영 임계값을 사후에 조정할 근거.
    blur_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    # USER 면 되묻기로 사용자가 직접 고른 것 = 정답 라벨로 쓸 수 있다.
    resolved_by: Mapped[str | None] = mapped_column(String(10))
    is_reask: Mapped[bool] = mapped_column(Boolean, default=False)
    is_retake: Mapped[bool] = mapped_column(Boolean, default=False)
    allergy_hit: Mapped[bool | None] = mapped_column(Boolean)

    # turn_candidate.candidate_id 참조. 되묻기 전 1위와 최종 확정을 구분해
    # "되묻기가 정확도를 몇 %p 올렸나"를 계산한다.
    chosen_candidate_id: Mapped[int | None] = mapped_column(BigInteger)
    # 발화 시작 시각과 실제로 선택된 프레임의 캡처 시각 차이.
    # 프레임 선택 로직이 의도대로 도는지 검증하는 유일한 근거다.
    frame_offset_ms: Mapped[int | None] = mapped_column(Integer)
    # 후보가 1개뿐이면 margin 이 무의미하다. 난이도 구분에 쓴다.
    detected_count: Mapped[int | None] = mapped_column(SmallInteger)
    # YOLO 가중치 버전. 재학습하면 같은 pointing_variant 라도 결과가 달라진다.
    detector_version: Mapped[str | None] = mapped_column(String(30))

    stt_ms: Mapped[int | None] = mapped_column(Integer)
    route_ms: Mapped[int | None] = mapped_column(Integer)
    vision_ms: Mapped[int | None] = mapped_column(Integer)
    llm_ms: Mapped[int | None] = mapped_column(Integer)
    # 응답 텍스트 확정 → 첫 오디오 청크 수신까지(TTFB). 전체 합성 시간이 아니다.
    tts_ms: Mapped[int | None] = mapped_column(Integer)
    total_ms: Mapped[int | None] = mapped_column(Integer)

    turn: Mapped[SessionTurn] = relationship(back_populates="inference")


class TurnCandidate(Base):
    """한 턴의 가리킴 후보 하나. 상위 N개를 남긴다.

    ``cue_scores`` 를 따로 남기는 이유
    ---------------------------------
    합산 ``score`` 만 있으면 ablation 결과가 "정확도 숫자"에서 끝난다.
    어느 단서가 이 후보를 밀어올렸는지 알아야 실패 케이스를 되짚을 수 있다.
    """

    __tablename__ = "turn_candidate"
    __table_args__ = (
        # 한 턴에서 최종 채택은 1개뿐이다.
        Index(
            "uq_turn_candidate_chosen",
            "turn_id",
            unique=True,
            postgresql_where=text("is_chosen"),
            sqlite_where=text("is_chosen"),
        ),
    )

    candidate_id: Mapped[int] = mapped_column(BigPK, primary_key=True)
    turn_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("session_turn.turn_id", ondelete="CASCADE"), index=True
    )
    rank: Mapped[int] = mapped_column(SmallInteger, default=1)
    label: Mapped[str] = mapped_column(String(50), index=True)
    score: Mapped[Decimal] = mapped_column(Score, default=0)
    # 0~1 정규화 좌표. 픽셀로 넣으면 해상도가 바뀔 때 못 쓴다.
    bbox: Mapped[dict] = mapped_column(JSON, default=dict)
    # {"center":0.30,"point":0.82,"gaze":0.15,"lang":0.55,"ctx":0.00}
    cue_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    is_chosen: Mapped[bool] = mapped_column(Boolean, default=False)

    turn: Mapped[SessionTurn] = relationship(back_populates="candidates")


# ==========================================================================
# ② 식품 성분 판정  (담당: 권용현)
# ==========================================================================


class Product(Base):
    """바코드로 조회한 제품 정보 캐시.

    HACCP 공공데이터 응답과 OCR 결과를 같은 모양으로 담는다.
    캐시가 있는 이유는 같은 과자를 여러 번 물어볼 것이기 때문이다.
    외부 API 를 매번 때리면 느리고 할당량도 금방 닳는다.
    """

    __tablename__ = "product"

    barcode: Mapped[str] = mapped_column(String(20), primary_key=True)
    product_name: Mapped[str] = mapped_column(String(200), default="")
    report_no: Mapped[str] = mapped_column(String(30), default="")  # 품목보고번호
    raw_ingredients: Mapped[str] = mapped_column(Text, default="")  # 원재료 원문
    ingredients: Mapped[list] = mapped_column(JSON, default=list)  # 파싱한 성분 목록
    allergen_notice: Mapped[str | None] = mapped_column(String(500))  # 알레르기 표시 문구
    cross_contamination: Mapped[str | None] = mapped_column(String(500))  # 교차오염 문구
    source: Mapped[str | None] = mapped_column(String(10))  # API / OCR
    ocr_conf: Mapped[Decimal | None] = mapped_column(Score)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    judgements: Mapped[list[TurnIngredient]] = relationship(back_populates="product")


class TurnIngredient(Base):
    """한 턴의 성분 판정 결과.  턴과 1:1.

    원문(``response_text``)과 걸린 성분(``matched``)을 둘 다 남기는 이유
    -----------------------------------------------------------------
    성분표에 적힌 말("탈지분유")과 그것이 해당하는 알레르겐("우유")은 다르다.
    둘 다 있어야 "탈지분유가 들어 있어 우유 알레르기에 해당합니다"라고
    설명할 수 있고, 동의어 사전이 틀렸을 때 어디가 틀렸는지 찾을 수 있다.
    """

    __tablename__ = "turn_ingredient"

    turn_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("session_turn.turn_id", ondelete="CASCADE"), primary_key=True
    )
    barcode: Mapped[str | None] = mapped_column(
        String(20), ForeignKey("product.barcode"), index=True
    )
    source: Mapped[str | None] = mapped_column(String(10))  # CACHE / API / OCR
    verdict: Mapped[str | None] = mapped_column(String(10), index=True)  # 위험 / 주의 / 안전
    matched: Mapped[list] = mapped_column(JSON, default=list)  # 걸린 성분 표기
    normalized: Mapped[list] = mapped_column(JSON, default=list)  # 정규화된 알레르겐
    ocr_conf: Mapped[Decimal | None] = mapped_column(Score)
    retry_count: Mapped[int] = mapped_column(SmallInteger, default=0)
    response_text: Mapped[str | None] = mapped_column(Text)
    ocr_ms: Mapped[int | None] = mapped_column(Integer)
    match_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    turn: Mapped[SessionTurn] = relationship(back_populates="ingredient")
    product: Mapped[Product | None] = relationship(back_populates="judgements")


# ==========================================================================
# 평가  (담당: 최홍묵)
# ==========================================================================


class EvalRun(Base):
    """실험 한 번. ablation 표의 한 줄이 된다.

    지표를 열로 늘리지 않고 ``metrics`` JSON 하나로 두는 이유는,
    측정할 지표가 프로젝트 도중에 계속 늘어나기 때문이다.
    """

    __tablename__ = "eval_run"

    run_id: Mapped[int] = mapped_column(BigPK, primary_key=True)
    variant: Mapped[str] = mapped_column(String(20), index=True)  # v1 ~ v5
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    # {"accuracy":0.872,"reask_rate":0.14,"allergy_recall":0.96,
    #  "yolo_map50":0.731,"p50_ms":1840,"p95_ms":3120}
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    note: Mapped[str | None] = mapped_column(Text)

    samples: Mapped[list[EvalSample]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvalSample(Base):
    """평가셋 한 건의 채점 결과.

    ``turn_id`` 를 남기면 실사용 턴을 평가셋으로 재사용할 수 있고,
    틀린 케이스에서 그 턴의 이미지와 후보 목록까지 그대로 따라갈 수 있다.
    """

    __tablename__ = "eval_sample"

    sample_id: Mapped[int] = mapped_column(BigPK, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("eval_run.run_id", ondelete="CASCADE"), index=True
    )
    turn_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("session_turn.turn_id", ondelete="SET NULL"), index=True
    )
    gt_label: Mapped[str] = mapped_column(String(50))
    pred_label: Mapped[str | None] = mapped_column(String(50))
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    run: Mapped[EvalRun] = relationship(back_populates="samples")


__all__ = [
    "Base",
    "User",
    "UserSetting",
    "Allergen",
    "IngredientSynonym",
    "UserAllergen",
    "ChatSession",
    "SessionTurn",
    "TurnInference",
    "TurnCandidate",
    "Product",
    "TurnIngredient",
    "EvalRun",
    "EvalSample",
]
