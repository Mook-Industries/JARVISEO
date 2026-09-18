"""SQLite 정형 저장소.  담당: C

대화 기록, 단계별 지연시간, 판정 결과를 남긴다.
대시보드의 "단계별 지연 그래프"와 팀장 책임 지표(응답 지연 p50/p95)가
전부 이 테이블에서 나온다.

전체 스키마(10개 테이블)는 팀 Notion 의 DB 설계 문서가 원본이다.
여기서는 코드가 실제로 쓰는 것만 정의한다.
"""

from __future__ import annotations

from pathlib import Path

from jarviseo.types import AssistantResponse, Utterance

__all__ = ["MemoryStore"]


class MemoryStore:
    """SQLite 연결과 질의를 감싼다."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        raise NotImplementedError

    def init_schema(self) -> None:
        """테이블이 없으면 만든다. 앱 시작할 때 한 번 부른다."""
        raise NotImplementedError

    def log_turn(self, utterance: Utterance, response: AssistantResponse) -> int:
        """대화 한 번을 기록하고 id 를 반환한다.

        response.latency_ms 를 단계별로 풀어서 저장한다.
        나중에 p50/p95 를 내려면 단계별 원본이 남아 있어야 한다.
        """
        raise NotImplementedError

    def get_user_allergens(self, user_id: str = "owner") -> list[str]:
        """사용자가 등록한 알레르기 목록. allergen.py 가 이걸 받아 쓴다."""
        raise NotImplementedError

    def latency_percentiles(self, stage: str) -> dict[str, float]:
        """특정 단계의 p50/p95 지연을 계산한다. 대시보드용."""
        raise NotImplementedError
