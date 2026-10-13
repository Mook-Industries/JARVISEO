"""관계형 저장소 접근.

Postgres 에 붙어서 대화 기록·지연시간·판정 결과를 남기고 조회한다.
스키마는 ``models.py`` 에 있고, 그 원본은 팀 ERD(``docs/ERD.md``)다.

연결 대상은 ``config.DATABASE_URL`` 하나로 정해진다. Postgres 든 SQLite 든
이 파일의 코드는 똑같다. 발표 데모에서 컨테이너를 못 띄우는 상황이 오면
환경변수만 바꿔서 SQLite 로 돌릴 수 있다.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from jarviseo import config
from jarviseo.memory.models import (
    Allergen,
    Base,
    ChatSession,
    Product,
    SessionTurn,
    TurnCandidate,
    TurnInference,
    TurnIngredient,
    TurnVoice,
    User,
    UserAllergen,
)
from jarviseo.types import AssistantResponse, IngredientSource, ProductInfo, Utterance

if TYPE_CHECKING:
    # 타입 표기에만 쓴다. 실제로 불러오면 DB 모듈이 OpenCV·OCR 까지 끌고 온다.
    from jarviseo.nutrition.service import IngredientCheck

__all__ = ["MemoryStore", "STAGE_COLUMNS"]


# 파이프라인이 넘겨주는 단계 이름 → 그 지연을 담는 열.
# 두 이름을 모두 받는 이유는, 그래프 노드가 쓰는 말("detect", "vlm")과
# ERD 의 열 이름("vision_ms", "llm_ms")이 다르기 때문이다.
# ERD 에서 지연 열이 세 테이블에 나뉘어 있다. 음성은 turn_voice,
# 추론은 turn_inference, 전체는 session_turn 이다.
STAGE_COLUMNS = {
    "stt": TurnVoice.stt_ms,
    "route": TurnInference.route_ms,
    "router": TurnInference.route_ms,
    "detect": TurnInference.vision_ms,
    "vision": TurnInference.vision_ms,
    "vlm": TurnInference.llm_ms,
    "llm": TurnInference.llm_ms,
    "tts": TurnVoice.tts_ms,
    "total": SessionTurn.total_ms,
}


class MemoryStore:
    """DB 연결과 질의를 감싼다.

    앱 전체에서 하나만 만들어 돌려 쓴다. 연결 풀이 안에 있어서,
    요청마다 새로 만들면 연결이 쌓인다.
    """

    def __init__(self, database_url: str | None = None, echo: bool = False) -> None:
        """
        Args:
            database_url: 없으면 ``config.DATABASE_URL`` 을 쓴다.
            echo: True 면 실행되는 SQL 을 전부 찍는다. 디버깅할 때만 켠다.
        """
        self.database_url = database_url or config.DATABASE_URL
        # pool_pre_ping: 컨테이너가 재시작되면 죽은 연결이 풀에 남는다.
        # 쓰기 전에 살아 있는지 확인해서 "server closed the connection" 을 막는다.
        self.engine = create_engine(self.database_url, echo=echo, pool_pre_ping=True)
        self._make_session = sessionmaker(bind=self.engine, expire_on_commit=False)

    # -- 수명주기 ----------------------------------------------------------

    def init_schema(self) -> None:
        """테이블이 없으면 만든다. 앱 시작할 때 한 번 부른다.

        스키마를 **바꿀 때는 이것만으로 부족하다.** create_all 은 없는 테이블만
        만들고 기존 테이블의 열은 건드리지 않는다. 열을 추가·변경하려면
        DB 를 지우고 다시 만들거나(개발 중에는 이게 빠르다) Alembic 을 붙여야 한다.

        Postgres 면 pgvector 확장부터 켠다. 이게 없으면 ``VECTOR`` 열이 있는
        테이블을 만들 때 ``type "vector" does not exist`` 로 실패한다.
        """
        if self.engine.dialect.name == "postgresql":
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        """모든 테이블을 지운다. 개발 중 스키마를 갈아엎을 때만 쓴다."""
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[OrmSession]:
        """세션을 열고, 끝나면 커밋하고 닫는다. 예외가 나면 롤백한다.

        with store.session() as db:
            db.add(...)
        """
        db = self._make_session()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # -- 사용자 ------------------------------------------------------------

    def ensure_user(self, email: str = "owner@jarviseo.local", nickname: str = "owner") -> int:
        """사용자가 없으면 만들고 user_id 를 반환한다.

        email 이 식별자다. 데모에서는 기본값 하나만 쓰면 된다.
        """
        with self.session() as db:
            user = db.scalar(select(User).where(User.email == email))
            if user is None:
                user = User(email=email, nickname=nickname)
                db.add(user)
                db.flush()
            return user.user_id

    def ensure_allergen(self, name: str, category: str = "") -> int:
        """알레르기 마스터에 이름을 등록하고 allergen_id 를 반환한다."""
        with self.session() as db:
            row = db.scalar(select(Allergen).where(Allergen.name == name))
            if row is None:
                row = Allergen(name=name, category=category)
                db.add(row)
                db.flush()
            return row.allergen_id

    def get_user_allergens(self, user_id: int) -> list[str]:
        """살아 있는 알레르기 목록. ``allergen.judge_allergens`` 가 이걸 받아 쓴다."""
        with self.session() as db:
            stmt = (
                select(Allergen.name)
                .join(UserAllergen, UserAllergen.allergen_id == Allergen.allergen_id)
                .where(UserAllergen.user_id == user_id)
                .where(UserAllergen.deleted_at.is_(None))
                .order_by(Allergen.name)
            )
            return list(db.scalars(stmt))

    def set_user_allergens(self, user_id: int, allergens: list[str]) -> None:
        """알레르기 목록을 통째로 바꾼다.

        지우지 않고 ``deleted_at`` 을 채운다. 과거 판정 기록이 "그때는 등록돼
        있었다"를 설명할 수 있어야 하기 때문이다. 다시 등록하면 되살린다.
        """
        wanted = {a for a in allergens if a}
        now = datetime.now(UTC)

        for name in wanted:
            self.ensure_allergen(name)

        with self.session() as db:
            id_by_name = (
                {
                    name: aid
                    for name, aid in db.execute(
                        select(Allergen.name, Allergen.allergen_id).where(Allergen.name.in_(wanted))
                    ).all()
                }
                if wanted
                else {}
            )

            rows = list(db.scalars(select(UserAllergen).where(UserAllergen.user_id == user_id)))
            existing = {r.allergen_id: r for r in rows}

            for row in rows:
                alive = row.allergen_id in id_by_name.values()
                if not alive and row.deleted_at is None:
                    row.deleted_at = now

            for name in wanted:
                aid = id_by_name[name]
                row = existing.get(aid)
                if row is None:
                    db.add(UserAllergen(user_id=user_id, allergen_id=aid))
                elif row.deleted_at is not None:
                    row.deleted_at = None

    # -- 제품 캐시 (② 식품 성분) --------------------------------------------

    def get_product(self, barcode: str) -> ProductInfo | None:
        """캐시된 제품 정보. 없거나 ``invalid`` 로 표시된 행이면 None.

        반환값의 source 는 CACHE 다. 원래 API 에서 왔든 OCR 에서 왔든,
        이번 턴에서는 캐시에서 꺼낸 것이기 때문이다.
        """
        with self.session() as db:
            row = db.get(Product, barcode)
            if row is None or row.status == "invalid":
                return None
            return ProductInfo(
                barcode=row.barcode,
                product_name=row.product_name or "",
                report_no=row.report_no or "",
                raw_ingredients=row.raw_ingredients or "",
                allergen_notice=row.allergen_notice or "",
                cross_contamination=row.cross_contamination or "",
                source=IngredientSource.CACHE,
            )

    def save_product(
        self, info: ProductInfo, status: str = "verified", ingredients: list[str] | None = None
    ) -> None:
        """제품 정보를 캐시에 넣거나 덮어쓴다.

        status: verified(API 에서 옴 / OCR 결과가 다시 일치) · pending(OCR 첫 저장)
        바코드가 없는 OCR 결과는 저장하지 않는다. 다음에 찾을 열쇠가 없다.
        """
        if not info.barcode:
            return
        with self.session() as db:
            row = db.get(Product, info.barcode)
            if row is None:
                row = Product(barcode=info.barcode)
                db.add(row)
            row.product_name = info.product_name
            row.report_no = info.report_no
            row.raw_ingredients = info.raw_ingredients
            row.allergen_notice = info.allergen_notice or None
            row.cross_contamination = info.cross_contamination or None
            if ingredients is not None:
                row.ingredients = ingredients
            row.status = status

    def log_ingredient(self, turn_id: int, check: IngredientCheck) -> None:
        """성분 판정 결과를 턴에 붙여 남긴다 (``turn_ingredient``, ``session_turn.ocr_text``).

        ``log_turn`` 으로 턴을 만든 뒤 부른다. 재촬영하면 같은 턴의 기록을 새 결과로 덮어쓴다.

        barcode 는 product 테이블에 있을 때만 적는다. 외래키라서, 바코드는 읽었는데
        제품을 못 찾고 캐시도 안 한 경우에 적으면 Postgres 에서 저장이 실패한다.
        retry_count 는 재촬영 횟수다(같은 사진 재처리는 세지 않는다).
        """
        j = check.judgement
        read = check.barcode.barcode if check.barcode else None
        with self.session() as db:
            barcode = read.code if read and db.get(Product, read.code) is not None else None
            row = db.get(TurnIngredient, turn_id) or TurnIngredient(turn_id=turn_id)
            row.barcode = barcode
            row.source = j.source.value if j.source else None
            row.verdict = j.verdict.value
            row.matched = list(j.matched_terms)
            row.may_contain = list(j.may_contain_terms)
            row.normalized = list(check.ingredients)
            row.ocr_conf = round(check.ocr_conf, 3) if check.ocr_conf is not None else None
            row.retry_count = check.retake_count
            row.response_text = check.message
            row.ocr_ms = check.timings_ms.get("ocr")
            row.match_ms = check.timings_ms.get("match")
            db.add(row)

            turn = db.get(SessionTurn, turn_id)
            if turn is not None and check.ocr_text:
                turn.ocr_text = check.ocr_text

    # -- 세션 --------------------------------------------------------------

    def start_session(self, user_id: int | None = None, title: str = "") -> int:
        """대화 세션을 시작하고 session_id 를 반환한다.

        비로그인 사용도 가능하므로 user_id 는 없어도 된다.
        실행 장비·프레임 입력원 같은 벤치마크 조건은 ERD 에 열이 없으므로
        ``eval_run.metrics`` 에 함께 적어 둔다.
        """
        with self.session() as db:
            row = ChatSession(user_id=user_id, title=title or None)
            db.add(row)
            db.flush()
            return row.session_id

    def end_session(self, session_id: int, reason: str = "USER") -> None:
        """세션을 닫는다. reason: TIMEOUT / USER / APP_CLOSE."""
        with self.session() as db:
            row = db.get(ChatSession, session_id)
            if row is not None:
                row.ended_at = datetime.now(UTC)
                row.end_reason = reason

    # -- 대화 기록 ---------------------------------------------------------

    def log_turn(
        self,
        session_id: int,
        utterance: Utterance,
        response: AssistantResponse,
        image_path: str | None = None,
        trigger_type: str = "WAKEWORD",
        pointing_variant: str | None = None,
        detector_version: str | None = None,
        stt_raw_text: str | None = None,
    ) -> int:
        """대화 한 번을 기록하고 turn_id 를 반환한다.

        한 턴이 네 테이블에 나뉘어 들어간다.
        ``session_turn`` (질문·응답·전체 지연) / ``turn_voice`` (STT·TTS) /
        ``turn_inference`` (판정·지연) / ``turn_candidate`` (후보 목록).
        한 트랜잭션 안에서 같이 써야 "턴은 있는데 추론 결과가 없는"
        반쪽 기록이 남지 않는다.

        stt_raw_text 는 호출어·필러를 빼기 전 받아쓴 원문(``SpeechToText.last_raw_text``)이다.
        안 넘기면 utterance.text 를 그대로 적는다.
        """
        with self.session() as db:
            turn = SessionTurn(
                session_id=session_id,
                trigger_type=trigger_type,
                question_text=utterance.text,
                answer_text=response.text,
                image_path=image_path,
            )
            db.add(turn)
            db.flush()

            voice = TurnVoice(turn_id=turn.turn_id, stt_raw_text=stt_raw_text or utterance.text)
            inference = TurnInference(
                turn_id=turn.turn_id,
                is_reask=response.needs_clarify,
                pointing_variant=pointing_variant,
                detector_version=detector_version,
            )

            frame = response.used_frame
            if frame is not None:
                inference.blur_score = frame.sharpness
                # 발화 시작과 실제로 고른 프레임이 얼마나 벌어졌는지.
                # 둘 다 time.monotonic() 기준이라 그대로 빼면 된다.
                # 잘라내지 않고 반올림한다. 0.1초가 99ms 로 기록되면 측정값이 아니라 버그다.
                inference.frame_offset_ms = round((frame.timestamp - utterance.started_at) * 1000)

            # 열이 속한 테이블의 행에 지연을 적는다.
            row_by_table = {"session_turn": turn, "turn_voice": voice, "turn_inference": inference}
            for stage, ms in (response.latency_ms or {}).items():
                column = STAGE_COLUMNS.get(stage)
                if column is not None:
                    setattr(row_by_table[column.class_.__tablename__], column.key, round(ms))
            if turn.total_ms is None and response.latency_ms:
                turn.total_ms = round(sum(response.latency_ms.values()))

            target = response.target
            if target is not None:
                inference.margin = target.margin
                inference.detected_count = len(target.candidates)
                if target.chosen is not None:
                    inference.target_label = target.chosen.detection.label
                    inference.confidence = target.chosen.detection.confidence
                # 되물은 턴에서는 아직 아무도 고르지 않았다. 사용자가 실제로 고르는 것은
                # 다음 턴(CLARIFY_REPLY)이다. 여기서 USER 를 적으면 USER 행을 정답 라벨로
                # 쓸 때 "모델이 헷갈려 한 후보"가 정답으로 섞여 들어간다.
                if response.needs_clarify:
                    inference.resolved_by = None
                elif trigger_type == "CLARIFY_REPLY":
                    inference.resolved_by = "USER"
                else:
                    inference.resolved_by = "MODEL"

                for rank, cand in enumerate(target.candidates, start=1):
                    box = cand.detection.bbox
                    db.add(
                        TurnCandidate(
                            turn_id=turn.turn_id,
                            rank=rank,
                            label=cand.detection.label,
                            score=cand.total_score,
                            bbox={
                                "x1": box.x1,
                                "y1": box.y1,
                                "x2": box.x2,
                                "y2": box.y2,
                            },
                            cue_scores={c.kind.value: c.score for c in cand.cue_scores},
                            is_chosen=cand is target.chosen,
                        )
                    )

            db.add(voice)
            db.add(inference)
            return turn.turn_id

    # -- 대시보드 ----------------------------------------------------------

    def latency_percentiles(self, stage: str, session_id: int | None = None) -> dict[str, float]:
        """특정 단계의 p50/p95 지연을 계산한다.

        Postgres 의 percentile_cont 를 쓰지 않고 파이썬에서 계산한다.
        SQLite 에는 그 함수가 없어서, 쓰면 발표 데모용 SQLite 폴백이 깨진다.
        행 수가 수천 개 수준이라 속도 차이는 문제가 되지 않는다.

        Args:
            stage: "stt" / "detect" / "vlm" / "tts" / "total" 등.
                ``STAGE_COLUMNS`` 에 없는 이름이면 전부 0 을 돌려준다.

        Returns:
            {"p50": ..., "p95": ..., "count": ...}. 기록이 없으면 전부 0.
        """
        empty = {"p50": 0.0, "p95": 0.0, "count": 0}
        column = STAGE_COLUMNS.get(stage)
        if column is None:
            return empty

        model = column.class_
        with self.session() as db:
            stmt = select(column).where(column.is_not(None))
            if session_id is not None:
                if model is not SessionTurn:
                    stmt = stmt.join(SessionTurn, SessionTurn.turn_id == model.turn_id)
                stmt = stmt.where(SessionTurn.session_id == session_id)
            values = sorted(float(v) for v in db.scalars(stmt))

        if not values:
            return empty

        def pick(q: float) -> float:
            # 가장 가까운 순위값을 고른다(nearest-rank).
            # 보간하지 않는 이유는, 실제로 관측된 값만 보고하는 편이
            # "p95 가 320ms" 를 설명하기 쉽기 때문이다.
            index = min(len(values) - 1, int(q * len(values)))
            return values[index]

        return {"p50": pick(0.50), "p95": pick(0.95), "count": len(values)}

    def recent_turns(self, session_id: int | None = None, limit: int = 50) -> list[SessionTurn]:
        """최근 대화 기록. 대시보드의 채팅 이력에 쓴다."""
        with self.session() as db:
            stmt = select(SessionTurn).order_by(SessionTurn.asked_at.desc()).limit(limit)
            if session_id is not None:
                stmt = stmt.where(SessionTurn.session_id == session_id)
            return list(db.scalars(stmt))
