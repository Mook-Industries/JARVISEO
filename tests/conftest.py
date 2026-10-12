"""DB 를 쓰는 테스트가 함께 쓰는 fixture.

``store`` 를 받는 테스트는 SQLite 인메모리와 실제 Postgres 에서 한 번씩 돈다.
Postgres 쪽은 ``JARVISEO_TEST_DATABASE_URL`` 이 있고 거기 붙을 수 있을 때만 돌고,
아니면 건너뛴다. CI 의 db job 이 pgvector 컨테이너를 띄우고 이 값을 넣어 준다.

    JARVISEO_TEST_DATABASE_URL=postgresql+psycopg://jarviseo:jarviseo@localhost:5432/jarviseo_test

테스트마다 테이블을 전부 지우고 다시 만든다. 개발 DB(``JARVISEO_DATABASE_URL``)를
가리키면 데이터가 날아가므로 꼭 테스트용 DB 를 따로 쓴다.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from jarviseo.memory.store import MemoryStore

POSTGRES_URL = os.getenv("JARVISEO_TEST_DATABASE_URL")


@pytest.fixture(scope="session")
def postgres_url() -> str:
    """테스트 Postgres 에 붙을 수 있는지 한 번만 보고 주소를 준다.

    윈도우에서는 컨테이너가 꺼져 있으면 연결이 몇 분씩 매달려서 3초에 끊는다.
    """
    if not POSTGRES_URL:
        pytest.skip("JARVISEO_TEST_DATABASE_URL 이 없어 Postgres 테스트를 건너뜀")
    engine = create_engine(POSTGRES_URL, connect_args={"connect_timeout": 3})
    try:
        with engine.connect():
            pass
    except OperationalError as e:
        # CI 에서는 실패로 둔다. 전부 건너뛰고 초록불이 뜨면 Postgres 를 안 본 것을 모른다.
        if os.getenv("CI"):
            raise
        pytest.skip(f"테스트 Postgres 에 못 붙음: {e.orig}")
    finally:
        engine.dispose()
    return POSTGRES_URL


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def store(request: pytest.FixtureRequest) -> Iterator[MemoryStore]:
    """매 테스트마다 빈 DB 를 준다."""
    if request.param == "sqlite":
        s = MemoryStore(database_url="sqlite://")
        s.init_schema()
        yield s
        return

    s = MemoryStore(database_url=request.getfixturevalue("postgres_url"))
    s.drop_all()
    s.init_schema()
    yield s
    s.engine.dispose()
