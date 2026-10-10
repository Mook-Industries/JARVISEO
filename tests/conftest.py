"""DB 를 쓰는 테스트가 함께 쓰는 fixture."""

from __future__ import annotations

import pytest

from jarviseo.memory.store import MemoryStore


@pytest.fixture
def store() -> MemoryStore:
    """매 테스트마다 빈 인메모리 DB 를 준다."""
    s = MemoryStore(database_url="sqlite://")
    s.init_schema()
    return s
