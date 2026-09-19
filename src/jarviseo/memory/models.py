"""데이터베이스 스키마 — ERD 를 코드로 옮긴 것.  담당: C

이 파일이 Notion 의 DB 설계 문서와 같은 것을 말해야 한다.
**둘이 어긋나면 이 파일이 아니라 Notion 을 고친다.** 코드가 실제로 도는 쪽이다.

설계 원칙
---------
- 시각은 두 종류를 구분해서 쓴다.
  ``created_at`` 은 벽시계(언제 일어난 일인가),
  ``*_monotonic`` 은 ``time.monotonic()`` 값(프레임과 발화를 맞출 때 쓴다).
  monotonic 값은 프로그램을 껐다 켜면 기준이 바뀌므로 세션 밖에서는 의미가 없다.
- Enum 은 네이티브 DB 타입 대신 문자열로 저장한다.
  Postgres 의 ENUM 타입은 값을 추가할 때마다 마이그레이션이 필요한데,
  ``Intent`` 나 ``CueKind`` 는 프로젝트 도중에 늘어날 것이 확실하다.
- 길이가 정해지지 않은 목록(matched_terms 등)은 JSON 으로 둔다.
  낱개로 조회할 일이 없어서 테이블을 나눌 이유가 없다.
- ``JSON`` 은 Postgres 와 SQLite 양쪽에서 동작한다. Postgres 전용 JSONB 를
  쓰면 SQLite 로 되돌릴 수 없다.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """모든 테이블의 부모. ``Base.metadata`` 로 스키마를 만든다."""


# --------------------------------------------------------------------------
# 사용자 — "만능 비서"가 기억하는 개인 정보
# --------------------------------------------------------------------------


class User(Base):
    """주인님. 지금은 1명이지만 테이블로 둔다.

    데모에서 "다른 사람으로 바꿔보세요"를 보여줄 수 있고,
    알레르기 판정이 누구 기준인지가 기록에 남는다.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    note: Mapped[str] = mapped_column(Text, default="")  # 자유 메모
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    allergens: Mapped[list[UserAllergen]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    sessions: Mapped[list[Session]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    belongings: Mapped[list[Belonging]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class UserAllergen(Base):
    """사용자가 등록한 알레르기 하나.

    "땅콩 알러지가 있는 주인님께는 추천드리지 않습니다"의 근거가 여기서 나온다.
    severity 는 표시 문구를 고르는 데 쓴다 — 'avoid' 면 강하게 경고한다.
    """

    __tablename__ = "user_allergens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    allergen: Mapped[str] = mapped_column(String(50))  # "땅콩", "우유" 등
    severity: Mapped[str] = mapped_column(String(20), default="avoid")  # avoid | caution

    user: Mapped[User] = relationship(back_populates="allergens")


# --------------------------------------------------------------------------
# 세션과 대화
# --------------------------------------------------------------------------


class Session(Base):
    """앱을 켜고 끌 때까지의 한 구간.

    monotonic 시각의 기준점이다. 세션이 다르면 monotonic 값을 서로 비교할 수 없다.
    벤치마크를 "어느 실행에서 잰 숫자인가"로 묶을 때도 이 단위를 쓴다.
    """

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 어떤 환경에서 돌린 실행인지. 벤치마크 표를 만들 때 이 값으로 묶는다.
    device: Mapped[str] = mapped_column(String(50), default="")  # "macbook-m3" 등
    frame_source: Mapped[str] = mapped_column(String(30), default="")  # usbcam | folder ...

    user: Mapped[User] = relationship(back_populates="sessions")
    turns: Mapped[list[Turn]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Frame(Base):
    """질문 시점에 고른 프레임 한 장.

    이미지 자체는 DB 에 넣지 않고 파일로 두고 경로만 저장한다.
    1080p 한 장이 수백 KB 라 DB 에 쌓으면 금방 무거워진다.

    sharpness 를 남기는 이유는, 나중에 "틀린 답이 흐린 프레임 때문이었나"를
    되짚어볼 수 있어야 하기 때문이다.
    """

    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_path: Mapped[str] = mapped_column(String(500))
    source_id: Mapped[str] = mapped_column(String(100))  # "usbcam:0", "folder/img_003.jpg"
    captured_at_monotonic: Mapped[float] = mapped_column(Float)
    sharpness: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Turn(Base):
    """대화 한 번. 이 프로젝트에서 가장 중요한 테이블이다.

    "자비서, 저거 뭐야?" 부터 음성 답변까지가 한 행이다.
    대시보드의 채팅 이력, 지연 그래프, 정확도 측정이 전부 여기서 나온다.

    recapture_count 와 clarify_count 를 남기는 이유
    ------------------------------------------------
    되묻기는 이 프로젝트의 차별점이라 "얼마나 자주 되물었나"가 지표가 된다.
    너무 잦으면 임계값이 높은 것이고, 0 이면 임계값이 무의미했다는 뜻이다.
    """

    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    frame_id: Mapped[int | None] = mapped_column(ForeignKey("frames.id"))

    # 사용자 발화
    utterance: Mapped[str] = mapped_column(Text)
    utterance_started_monotonic: Mapped[float] = mapped_column(Float)
    stt_confidence: Mapped[float | None] = mapped_column(Float)

    # 라우팅 결과. types.Intent 의 값을 그대로 넣는다.
    intent: Mapped[str] = mapped_column(String(30), index=True)

    # 최종 응답
    response: Mapped[str] = mapped_column(Text, default="")
    needs_clarify: Mapped[bool] = mapped_column(Boolean, default=False)

    recapture_count: Mapped[int] = mapped_column(Integer, default=0)
    clarify_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    session: Mapped[Session] = relationship(back_populates="turns")
    frame: Mapped[Frame | None] = relationship()
    latencies: Mapped[list[TurnLatency]] = relationship(
        back_populates="turn", cascade="all, delete-orphan"
    )
    detections: Mapped[list[Detection]] = relationship(
        back_populates="turn", cascade="all, delete-orphan"
    )
    target: Mapped[TargetResolution | None] = relationship(
        back_populates="turn", cascade="all, delete-orphan", uselist=False
    )
    allergen: Mapped[AllergenJudgement | None] = relationship(
        back_populates="turn", cascade="all, delete-orphan", uselist=False
    )


class TurnLatency(Base):
    """한 턴의 단계별 소요 시간.

    턴마다 열을 늘리는 대신 행으로 쌓는 이유는, 단계가 계속 늘어날 것이기
    때문이다. 단서를 추가하면 측정할 구간도 같이 늘어난다.

    팀장 책임 지표(응답 지연 p50/p95)의 원천 데이터가 이 테이블이다.
    """

    __tablename__ = "turn_latencies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    turn_id: Mapped[int] = mapped_column(ForeignKey("turns.id"), index=True)
    stage: Mapped[str] = mapped_column(String(30), index=True)  # stt | detect | vlm | tts ...
    elapsed_ms: Mapped[float] = mapped_column(Float)

    turn: Mapped[Turn] = relationship(back_populates="latencies")


# --------------------------------------------------------------------------
# ① 지시 대상 특정  (담당: A · 팀장)
# --------------------------------------------------------------------------


class Detection(Base):
    """한 턴에서 검출한 물체 하나. 손끝도 여기 들어간다(label="fingertip")."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    turn_id: Mapped[int] = mapped_column(ForeignKey("turns.id"), index=True)
    label: Mapped[str] = mapped_column(String(50), index=True)
    confidence: Mapped[float] = mapped_column(Float)

    # BBox 를 따로 테이블로 빼지 않는다. 항상 네 값이 같이 다니고 낱개로 조회할 일이 없다.
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)

    # 단서별 점수. {"hand": 0.8, "gaze": 0.3, ...}
    # ablation 을 돌린 뒤 "어느 단서가 이 물체를 밀어올렸나"를 되짚을 때 쓴다.
    cue_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    total_score: Mapped[float] = mapped_column(Float, default=0.0)

    turn: Mapped[Turn] = relationship(back_populates="detections")


class TargetResolution(Base):
    """'저거'가 무엇으로 결정됐는지.

    is_correct 는 사람이 나중에 채점해서 채우는 칸이다.
    **대상 선택 정확도가 이 열에서 나온다.** 평가셋을 돌린 뒤
    대시보드에서 하나씩 맞다/틀리다를 눌러 채우는 것을 전제로 둔다.
    """

    __tablename__ = "target_resolutions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    turn_id: Mapped[int] = mapped_column(ForeignKey("turns.id"), unique=True, index=True)
    chosen_detection_id: Mapped[int | None] = mapped_column(ForeignKey("detections.id"))

    margin: Mapped[float] = mapped_column(Float)  # 1위 - 2위 점수차
    needs_clarify: Mapped[bool] = mapped_column(Boolean, default=False)
    clarify_question: Mapped[str | None] = mapped_column(Text)

    # 이 판정에서 켜져 있던 단서들. ["hand", "salience"]
    # ablation 표의 어느 줄에 해당하는 기록인지가 여기서 정해진다.
    enabled_cues: Mapped[list] = mapped_column(JSON, default=list)

    # 사람이 채우는 정답 라벨. None 이면 아직 채점 전.
    is_correct: Mapped[bool | None] = mapped_column(Boolean)

    turn: Mapped[Turn] = relationship(back_populates="target")


# --------------------------------------------------------------------------
# ② 식품 성분 판정  (담당: B)
# --------------------------------------------------------------------------


class AllergenJudgement(Base):
    """성분표를 읽고 내린 판정.

    matched_terms 와 matched_allergens 를 따로 두는 이유
    ---------------------------------------------------
    성분표에 적힌 말("탈지분유")과 그것이 해당하는 알레르겐("우유")은 다르다.
    둘 다 남겨야 "탈지분유가 들어 있어 우유 알레르기에 해당합니다"라고
    설명할 수 있고, 동의어 사전이 틀렸을 때 어디가 틀렸는지 찾을 수 있다.
    """

    __tablename__ = "allergen_judgements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    turn_id: Mapped[int] = mapped_column(ForeignKey("turns.id"), unique=True, index=True)

    verdict: Mapped[str] = mapped_column(String(20), index=True)  # safe | warn | uncertain
    matched_terms: Mapped[list] = mapped_column(JSON, default=list)  # ["탈지분유"]
    matched_allergens: Mapped[list] = mapped_column(JSON, default=list)  # ["우유"]
    reason: Mapped[str] = mapped_column(Text, default="")

    # OCR 원문과 파싱 결과. 인식률을 되짚어볼 때 원문이 있어야 한다.
    ocr_raw_text: Mapped[str] = mapped_column(Text, default="")
    ingredients: Mapped[list] = mapped_column(JSON, default=list)
    ocr_confidence: Mapped[float | None] = mapped_column(Float)

    # 사람이 채우는 정답 라벨. recall 계산의 근거.
    is_correct: Mapped[bool | None] = mapped_column(Boolean)

    turn: Mapped[Turn] = relationship(back_populates="allergen")


# --------------------------------------------------------------------------
# ③④ 소지품 재인식 · 개인 기억  (담당: C)
# --------------------------------------------------------------------------


class Belonging(Base):
    """등록한 개인 물품.

    임베딩 자체는 Chroma 에 있고 여기에는 ``chroma_id`` 만 둔다.
    벡터를 두 군데 두면 둘이 어긋났을 때 어느 쪽이 맞는지 알 수 없다.

    같은 물건을 여러 각도에서 여러 번 등록하므로 이름이 겹칠 수 있다.
    그래서 name 에 unique 를 걸지 않는다.
    """

    __tablename__ = "belongings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    image_path: Mapped[str] = mapped_column(String(500))
    chroma_id: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(back_populates="belongings")


class Observation(Base):
    """ "아까 본 그거"를 위해 남겨두는 과거 장면 기록.

    벡터는 Chroma 에, 사람이 읽을 설명과 경로는 여기에 둔다.
    """

    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    turn_id: Mapped[int | None] = mapped_column(ForeignKey("turns.id"))
    text: Mapped[str] = mapped_column(Text)
    image_path: Mapped[str | None] = mapped_column(String(500))
    chroma_id: Mapped[str] = mapped_column(String(100), unique=True)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
