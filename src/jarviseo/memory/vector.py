"""pgvector 벡터 검색.  담당: 문태현

책임 지표: RAG 응답 근거율

벡터 DB 를 따로 두지 않고 Postgres 의 pgvector 확장을 쓴다.
임베딩을 기록과 같은 행에 두면, 벡터와 기록이 어긋날 일이 없고 백업도 하나로 끝난다.

ERD 의 observation 테이블(지난 질문·대답 기록의 임베딩, memory/dialog.py 가 넣는다)을
검색한다. "아까 본 그거"

    memory = VectorMemory(store)
    memory.remember_observation(user_id, turn_id, "질문: 이거 뭐야?\n대답: 검은 백팩이에요.")
    hits = memory.search(user_id, "아까 본 가방 어디 있었지?")

'근거율'이 무엇인가
-------------------
LLM 이 답을 지어내지 않고 검색해온 기록을 근거로 답한 비율이다.
답변에 쓰인 문장이 실제로 검색 결과에 있던 내용인지를 세면 된다.
이 숫자가 낮으면 RAG 가 붙어 있어도 의미가 없다.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from openai import OpenAIError
from sqlalchemy import BindParameter, bindparam, select
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import defer
from sqlalchemy.types import UserDefinedType

from jarviseo import config
from jarviseo.memory.embed import FakeEmbedder, TextEmbedder
from jarviseo.memory.models import ChatSession, Observation, SessionTurn
from jarviseo.memory.store import MemoryStore
from jarviseo.memory.when import recent_dialog_range, time_range
from jarviseo.types import MemoryHit

__all__ = ["RECENT_TURNS", "VectorMemory"]

log = logging.getLogger(__name__)

# 지난 대화 자체를 물을 때("방금 물어봤던 거 뭐지?") 덧붙이는 최근 턴 수
RECENT_TURNS = 3


class VectorMemory:
    """observation 에 기록 글을 임베딩과 함께 넣고, 질문과 뜻이 가까운 것을 찾는다."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        embedder: TextEmbedder | FakeEmbedder | None = None,
        min_score: float = config.MEMORY_MIN_SCORE,
    ) -> None:
        """store 는 앱 전체에서 하나 쓰는 MemoryStore 를 넘긴다. 없으면 config 의 DB 로 만든다.
        embedder 는 테스트·그래프 개발에서 FakeEmbedder 를 끼울 때만 넘긴다.
        """
        self.store = store or MemoryStore()
        self.embedder = embedder or TextEmbedder()
        self.min_score = min_score
        # 마지막 search 에서 질문 임베딩 + DB 검색에 걸린 시간(ms). 실패했으면 None.
        # 그래프가 latency_ms["recall"] 로 넘긴다.
        self.last_latency_ms: float | None = None

    def remember_observation(
        self, user_id: int, turn_id: int, text: str, place: str | None = None
    ) -> int:
        """기록 글(지난 질문·대답)을 넣고 observation_id 를 돌려준다.

        임베딩이 실패해도 설명은 남긴다(embedding 이 NULL). 그 행은 검색에 안 나온다.
        """
        embedding = self._embed(text)
        with self.store.session() as db:
            observation = Observation(
                user_id=user_id,
                turn_id=turn_id,
                description=text,
                place=place,
                embedding=embedding,
                embed_model=self.embedder.model if embedding is not None else None,
            )
            db.add(observation)
            db.flush()
            return observation.observation_id

    def search(
        self, user_id: int, query: str, top_k: int = 5, now: datetime | None = None
    ) -> list[MemoryHit]:
        """user_id 의 기록 중 query 와 가까운 순으로 최대 top_k 개를 찾는다.

        점수(코사인 유사도)가 min_score 보다 낮은 결과는 버린다. 억지로 top_k 개를 채우면
        LLM 이 엉뚱한 기록을 근거로 삼는다. 임베딩이 실패하면 빈 목록을 준다.
        기억 검색이 안 돼도 답은 나가야 하기 때문이다.

        query 에 때 낱말("아까", "어제")이 있으면 그때 기록만 찾는다(memory/when.py).
        지난 대화 자체를 물으면("방금 물어봤던 거 뭐지?") 그때의 최근 턴 RECENT_TURNS 개를
        덧붙이고 전부 시간순(오래된 것 먼저)으로 준다. 물건 이름이 없어 뜻으로는 못 찾기 때문이다.
        최근 턴은 임베딩 없이 읽으므로 임베딩이 실패해도 나온다.
        now 는 테스트에서 지금 시각을 정할 때만 넘긴다.
        """
        self.last_latency_ms = None
        started = time.perf_counter()
        hits = self._similar(user_id, query, top_k, now)
        dialog_window = recent_dialog_range(query, now)
        if dialog_window is not None:
            hits = _merge(hits, self.recent_turns(user_id, *dialog_window))
            self.last_latency_ms = (time.perf_counter() - started) * 1000
        return hits

    def recent_turns(
        self, user_id: int, start: datetime, end: datetime | None = None, limit: int = RECENT_TURNS
    ) -> list[MemoryHit]:
        """user_id 가 start 이후(end 전)에 한 질문 중 마지막 limit 개를 시간순으로 준다.

        observation 이 아니라 session_turn 을 읽는다. 모른다는 대답 · 미리 정한 안내 · 기억 질문도
        기억에는 안 넣지만(dialog.py), "방금 뭐 물어봤지?"의 답은 그 질문이라 빼면 안 된다.
        지금 질문은 대답한 뒤에 기록하므로 여기 안 나온다.
        """
        found = (
            select(SessionTurn)
            .join(ChatSession, SessionTurn.session_id == ChatSession.session_id)
            .where(ChatSession.user_id == user_id, SessionTurn.question_text != "")
            .where(SessionTurn.asked_at >= start)
        )
        if end is not None:
            found = found.where(SessionTurn.asked_at < end)
        found = found.order_by(SessionTurn.asked_at.desc(), SessionTurn.turn_id.desc()).limit(limit)
        with self.store.session() as db:
            return [_turn_hit(turn) for turn in reversed(list(db.scalars(found)))]

    def _similar(
        self, user_id: int, query: str, top_k: int, now: datetime | None
    ) -> list[MemoryHit]:
        """query 와 뜻이 가까운 기록. 임베딩이 실패하면 빈 목록이다."""
        started = time.perf_counter()  # monotonic 은 윈도우에서 15.6ms 단위라 못 쓴다
        vector = self._embed(query)
        if vector is None:
            return []

        found = (
            select(Observation, SessionTurn.image_path)
            .join(SessionTurn, Observation.turn_id == SessionTurn.turn_id)
            .where(Observation.user_id == user_id, Observation.embedding.is_not(None))
        )
        window = time_range(query, now)
        if window is not None:
            start, end = window
            found = found.where(Observation.observed_at >= start)
            if end is not None:
                found = found.where(Observation.observed_at < end)
        with self.store.session() as db:
            if self.store.engine.dialect.name == "postgresql":
                distance = Observation.embedding.cosine_distance(_binary_vector(db, vector))
                found = found.options(defer(Observation.embedding))  # 결과에는 벡터가 필요 없다
                rows = db.execute(found.add_columns(distance).order_by(distance).limit(top_k))
                scored = [(obs, image, 1.0 - dist) for obs, image, dist in rows]
            else:
                # SQLite 에는 pgvector 연산자가 없다. 테스트·데모용이라 행을 다 읽어 잰다.
                rows = [(o, img, _cosine(vector, o.embedding)) for o, img in db.execute(found)]
                scored = sorted(rows, key=lambda row: row[2], reverse=True)[:top_k]
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        return [_hit(obs, image, score) for obs, image, score in scored if score >= self.min_score]

    def _embed(self, text: str) -> list[float] | None:
        if not text.strip():
            return None
        try:
            return self.embedder.embed(text)
        except OpenAIError as e:
            log.warning("임베딩 실패: %s", e)
            return None


def _merge(similar: list[MemoryHit], recent: list[MemoryHit]) -> list[MemoryHit]:
    """뜻으로 찾은 기록과 최근 턴을 합쳐 시간순으로 둔다. 같은 턴이면 최근 턴 쪽만 남긴다."""
    recent_ids = {hit.metadata["turn_id"] for hit in recent}
    merged = [hit for hit in similar if hit.metadata.get("turn_id") not in recent_ids] + recent
    return sorted(merged, key=lambda hit: hit.observed_at or 0.0)


def _turn_hit(turn: SessionTurn) -> MemoryHit:
    """최근 턴 하나를 기억 검색 결과 모양으로. 글은 dialog.dialog_text 와 같은 모양이다."""
    asked_at = turn.asked_at
    if asked_at is not None and asked_at.tzinfo is None:
        asked_at = asked_at.replace(tzinfo=UTC)  # SQLite 는 시간대를 빼고 돌려준다
    return MemoryHit(
        memory_id=f"turn-{turn.turn_id}",
        kind="turn",
        text=f"질문: {turn.question_text}\n대답: {turn.answer_text or '(대답 없음)'}",
        score=0.0,  # 뜻으로 찾은 것이 아니다
        observed_at=asked_at.timestamp() if asked_at is not None else None,
        image_path=Path(turn.image_path) if turn.image_path else None,
        metadata={"turn_id": turn.turn_id},
    )


def _binary_vector(db: OrmSession, vector: list[float]) -> BindParameter:
    """질문 벡터를 이진(float32 1536개, 약 6KB)으로 보내는 바인드 값.

    VECTOR 열 타입은 값을 '[0.0123…, …]' 글자(약 30KB)로 바꿔 보낸다. 윈도우 Docker Desktop
    포트로 8KB 넘게 보내면 왕복마다 약 44ms 가 붙어서(1,000행 검색 60ms 중 대부분),
    psycopg 에 pgvector 어댑터를 붙이고 타입 변환 없이 numpy 배열을 그대로 넘긴다.
    """
    # psycopg 는 Postgres 에 붙을 때만 있으면 된다.
    # SQLite 로만 쓰는 환경에서도 이 파일이 import 되게 여기서 불러온다.
    from pgvector.psycopg import register_vector

    pooled = db.connection().connection  # 풀의 DBAPI 연결. info 는 연결이 살아 있는 동안 남는다
    if not pooled.info.get("pgvector"):
        register_vector(pooled.dbapi_connection)
        pooled.info["pgvector"] = True
    return bindparam("query_vector", np.asarray(vector, dtype=np.float32), type_=_AsIs())


class _AsIs(UserDefinedType):
    """값을 바꾸지 않고 드라이버에 넘기는 타입.

    NullType 으로 두면 SQLAlchemy 가 비교 상대(VECTOR 열)의 타입을 덮어써서 다시 글자가 된다.
    """

    cache_ok = True

    def get_col_spec(self, **kw: object) -> str:
        return "VECTOR"


def _cosine(a: list[float], b: np.ndarray) -> float:
    a = np.asarray(a)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _hit(obs: Observation, image_path: str | None, score: float) -> MemoryHit:
    observed_at = obs.observed_at
    if observed_at is not None and observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)  # SQLite 는 시간대를 빼고 돌려준다
    return MemoryHit(
        memory_id=str(obs.observation_id),
        kind="observation",
        text=obs.description or "",
        score=round(float(score), 4),
        observed_at=observed_at.timestamp() if observed_at is not None else None,
        image_path=Path(image_path) if image_path else None,
        metadata={"turn_id": obs.turn_id, "place": obs.place},
    )
