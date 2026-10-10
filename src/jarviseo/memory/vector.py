"""pgvector 벡터 검색.  담당: 문태현

책임 지표: RAG 응답 근거율

벡터 DB 를 따로 두지 않고 Postgres 의 pgvector 확장을 쓴다.
임베딩을 기록과 같은 행에 두면, 벡터와 기록이 어긋날 일이 없고 백업도 하나로 끝난다.

ERD 의 observation 테이블(과거에 본 장면의 설명 임베딩)을 검색한다. "아까 본 그거"

'근거율'이 무엇인가
-------------------
LLM 이 답을 지어내지 않고 검색해온 기록을 근거로 답한 비율이다.
답변에 쓰인 문장이 실제로 검색 결과에 있던 내용인지를 세면 된다.
이 숫자가 낮으면 RAG 가 붙어 있어도 의미가 없다.
"""

from __future__ import annotations

from pathlib import Path

from jarviseo.types import MemoryHit

__all__ = ["VectorMemory"]


class VectorMemory:
    """pgvector 유사도 검색을 감싼다."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url
        raise NotImplementedError

    def remember_observation(self, text: str, image_path: Path | None = None) -> str:
        """지금 본 장면을 기록해둔다. 나중에 "아까 본 그거"로 찾을 수 있게."""
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[MemoryHit]:
        """질의와 비슷한 기록을 찾는다.

        점수가 낮은 결과는 걸러서 반환한다. 억지로 top_k 개를 채우면
        LLM 이 엉뚱한 기록을 근거로 삼는다.
        """
        raise NotImplementedError
