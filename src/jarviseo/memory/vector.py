"""Chroma 벡터 검색.  담당: C

책임 지표: RAG 응답 근거율

컬렉션 2개를 쓴다.
- belongings   : 등록한 개인 물품의 이미지 임베딩. "내 가방 어디 있어?"
- observations : 과거에 본 장면의 설명 임베딩. "아까 본 그거"

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
    """Chroma 컬렉션을 감싼다."""

    def __init__(self, persist_path: Path) -> None:
        self.persist_path = persist_path
        raise NotImplementedError

    def register_belonging(self, name: str, image_path: Path, note: str = "") -> str:
        """개인 물품을 등록한다. 반환값은 memory_id.

        같은 물건을 여러 각도에서 여러 장 등록해야 재인식이 된다.
        한 장만 등록하면 조명이나 각도가 바뀌는 순간 못 찾는다.
        """
        raise NotImplementedError

    def remember_observation(self, text: str, image_path: Path | None = None) -> str:
        """지금 본 장면을 기록해둔다. 나중에 "아까 본 그거"로 찾을 수 있게."""
        raise NotImplementedError

    def search(self, query: str, kind: str | None = None, top_k: int = 5) -> list[MemoryHit]:
        """질의와 비슷한 기록을 찾는다.

        kind 로 컬렉션을 좁힐 수 있다("belonging" / "observation").
        점수가 낮은 결과는 걸러서 반환한다. 억지로 top_k 개를 채우면
        LLM 이 엉뚱한 기록을 근거로 삼는다.
        """
        raise NotImplementedError
