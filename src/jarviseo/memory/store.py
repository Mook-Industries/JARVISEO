"""관계형 저장소 접근.  담당: C

Postgres 에 붙어서 대화 기록·지연시간·판정 결과를 남기고 조회한다.
스키마는 ``models.py`` 에 있다.

연결 대상은 ``config.DATABASE_URL`` 하나로 정해진다. Postgres 든 SQLite 든
이 파일의 코드는 똑같다. 발표 데모에서 컨테이너를 못 띄우는 상황이 오면
환경변수만 바꿔서 SQLite 로 돌릴 수 있다.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from jarviseo import config
from jarviseo.memory.models import (
    Base,
    Frame,
    Session,
    Turn,
    TurnLatency,
    User,
    UserAllergen,
)
from jarviseo.types import AssistantResponse, Utterance

__all__ = ["MemoryStore"]


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
        """
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

    def ensure_user(self, name: str = "owner") -> int:
        """사용자가 없으면 만들고 id 를 반환한다."""
        with self.session() as db:
            user = db.scalar(select(User).where(User.name == name))
            if user is None:
                user = User(name=name)
                db.add(user)
                db.flush()
            return user.id

    def get_user_allergens(self, user_id: int) -> list[str]:
        """등록된 알레르기 목록. ``allergen.judge_allergens`` 가 이걸 받아 쓴다."""
        with self.session() as db:
            rows = db.scalars(select(UserAllergen.allergen).where(UserAllergen.user_id == user_id))
            return list(rows)

    def set_user_allergens(self, user_id: int, allergens: list[str]) -> None:
        """알레르기 목록을 통째로 바꾼다."""
        with self.session() as db:
            db.query(UserAllergen).filter(UserAllergen.user_id == user_id).delete()
            db.add_all(UserAllergen(user_id=user_id, allergen=a) for a in allergens)

    # -- 세션 --------------------------------------------------------------

    def start_session(self, user_id: int, device: str = "", frame_source: str = "") -> int:
        """실행 세션을 시작하고 id 를 반환한다.

        device 와 frame_source 를 남기는 이유는, 나중에 벤치마크 표를 만들 때
        "맥북에서 잰 것"과 "Colab 에서 잰 것"을 섞으면 안 되기 때문이다.
        """
        with self.session() as db:
            row = Session(user_id=user_id, device=device, frame_source=frame_source)
            db.add(row)
            db.flush()
            return row.id

    # -- 대화 기록 ---------------------------------------------------------

    def log_turn(
        self,
        session_id: int,
        utterance: Utterance,
        response: AssistantResponse,
        frame_path: str | None = None,
    ) -> int:
        """대화 한 번을 기록하고 turn id 를 반환한다.

        ``response.latency_ms`` 를 단계별로 풀어서 별도 행으로 저장한다.
        나중에 p50/p95 를 내려면 단계별 원본이 남아 있어야 한다.
        """
        with self.session() as db:
            frame_id = None
            if response.used_frame is not None and frame_path is not None:
                f = response.used_frame
                width, height = f.size
                frame = Frame(
                    image_path=frame_path,
                    source_id=f.source_id,
                    captured_at_monotonic=f.timestamp,
                    sharpness=f.sharpness,
                    width=width,
                    height=height,
                )
                db.add(frame)
                db.flush()
                frame_id = frame.id

            turn = Turn(
                session_id=session_id,
                frame_id=frame_id,
                utterance=utterance.text,
                utterance_started_monotonic=utterance.started_at,
                stt_confidence=utterance.confidence,
                intent=str(response.intent),
                response=response.text,
                needs_clarify=response.needs_clarify,
            )
            db.add(turn)
            db.flush()

            db.add_all(
                TurnLatency(turn_id=turn.id, stage=stage, elapsed_ms=ms)
                for stage, ms in response.latency_ms.items()
            )
            return turn.id

    # -- 대시보드 ----------------------------------------------------------

    def latency_percentiles(self, stage: str, session_id: int | None = None) -> dict[str, float]:
        """특정 단계의 p50/p95 지연을 계산한다.

        Postgres 의 percentile_cont 를 쓰지 않고 파이썬에서 계산한다.
        SQLite 에는 그 함수가 없어서, 쓰면 발표 데모용 SQLite 폴백이 깨진다.
        행 수가 수천 개 수준이라 속도 차이는 문제가 되지 않는다.

        Returns:
            {"p50": ..., "p95": ..., "count": ...}. 기록이 없으면 전부 0.
        """
        with self.session() as db:
            stmt = select(TurnLatency.elapsed_ms).where(TurnLatency.stage == stage)
            if session_id is not None:
                stmt = stmt.join(Turn).where(Turn.session_id == session_id)
            values = sorted(db.scalars(stmt))

        if not values:
            return {"p50": 0.0, "p95": 0.0, "count": 0}

        def pick(q: float) -> float:
            # 가장 가까운 순위값을 고른다(nearest-rank).
            # 보간하지 않는 이유는, 실제로 관측된 값만 보고하는 편이
            # "p95 가 320ms" 를 설명하기 쉽기 때문이다.
            index = min(len(values) - 1, int(q * len(values)))
            return values[index]

        return {"p50": pick(0.50), "p95": pick(0.95), "count": len(values)}

    def recent_turns(self, session_id: int | None = None, limit: int = 50) -> list[Turn]:
        """최근 대화 기록. 대시보드의 채팅 이력에 쓴다."""
        with self.session() as db:
            stmt = select(Turn).order_by(Turn.created_at.desc()).limit(limit)
            if session_id is not None:
                stmt = stmt.where(Turn.session_id == session_id)
            return list(db.scalars(stmt))
