"""pgvector 벡터 검색.  담당: 문태현

책임 지표: RAG 응답 근거율

벡터 DB 를 따로 두지 않고 Postgres 의 pgvector 확장을 쓴다.
임베딩을 기록과 같은 행에 두면, 벡터와 기록이 어긋날 일이 없고 백업도 하나로 끝난다.

ERD 의 observation 테이블(과거에 본 장면의 설명 임베딩)을 검색한다. "아까 본 그거"

    memory = VectorMemory(store)
    memory.remember_observation(user_id, turn_id, "책상 위에 검은 백팩이 있다")
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
from datetime import UTC
from pathlib import Path

import numpy as np
from openai import OpenAIError
from sqlalchemy import select

from jarviseo import config
from jarviseo.memory.embed import FakeEmbedder, TextEmbedder
from jarviseo.memory.models import Observation, SessionTurn
from jarviseo.memory.store import MemoryStore
from jarviseo.types import MemoryHit

__all__ = ["VectorMemory"]

log = logging.getLogger(__name__)


class VectorMemory:
    """observation 에 장면 설명을 임베딩과 함께 넣고, 질문과 뜻이 가까운 것을 찾는다."""

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
        """지금 본 장면 설명을 기록하고 observation_id 를 돌려준다.

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

    def search(self, user_id: int, query: str, top_k: int = 5) -> list[MemoryHit]:
        """user_id 의 기록 중 query 와 가까운 순으로 최대 top_k 개를 찾는다.

        점수(코사인 유사도)가 min_score 보다 낮은 결과는 버린다. 억지로 top_k 개를 채우면
        LLM 이 엉뚱한 기록을 근거로 삼는다. 임베딩이 실패하면 빈 목록을 준다.
        기억 검색이 안 돼도 답은 나가야 하기 때문이다.
        """
        self.last_latency_ms = None
        started = time.monotonic()
        vector = self._embed(query)
        if vector is None:
            return []

        found = (
            select(Observation, SessionTurn.image_path)
            .join(SessionTurn, Observation.turn_id == SessionTurn.turn_id)
            .where(Observation.user_id == user_id, Observation.embedding.is_not(None))
        )
        with self.store.session() as db:
            if self.store.engine.dialect.name == "postgresql":
                distance = Observation.embedding.cosine_distance(vector)
                rows = db.execute(found.add_columns(distance).order_by(distance).limit(top_k))
                scored = [(obs, image, 1.0 - dist) for obs, image, dist in rows]
            else:
                # SQLite 에는 pgvector 연산자가 없다. 테스트·데모용이라 행을 다 읽어 잰다.
                rows = [(o, img, _cosine(vector, o.embedding)) for o, img in db.execute(found)]
                scored = sorted(rows, key=lambda row: row[2], reverse=True)[:top_k]
        self.last_latency_ms = (time.monotonic() - started) * 1000
        return [_hit(obs, image, score) for obs, image, score in scored if score >= self.min_score]

    def _embed(self, text: str) -> list[float] | None:
        if not text.strip():
            return None
        try:
            return self.embedder.embed(text)
        except OpenAIError as e:
            log.warning("임베딩 실패: %s", e)
            return None


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
