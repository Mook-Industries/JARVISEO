"""Postgres 가 꺼져 있으면 기다리지 않고 제한 시간 안에 예외를 낸다(#131).

실제 Postgres 는 필요 없다. 아무도 듣지 않는 포트로 연결한다.
리눅스·맥은 바로 거절되고, 윈도우는 제한 시간이 지나야 끊긴다.
"""

import socket
import time

import pytest
from sqlalchemy.exc import OperationalError

from jarviseo import config
from jarviseo.memory import MemoryStore


def _closed_port() -> int:
    """잠깐 열었다 닫은 포트. 아무도 듣지 않는다."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_closed_postgres_port_fails_within_timeout(monkeypatch):
    """닫힌 포트로 연결하면 끝없이 기다리지 않고 제한 시간 안에 OperationalError 를 낸다."""
    monkeypatch.setattr(config, "DB_CONNECT_TIMEOUT_SEC", 1)
    url = f"postgresql+psycopg://jarviseo:jarviseo@127.0.0.1:{_closed_port()}/jarviseo"
    store = MemoryStore(database_url=url)

    started = time.perf_counter()
    with pytest.raises(OperationalError):
        store.init_schema()
    assert time.perf_counter() - started < 10
    store.engine.dispose()


def test_sqlite_gets_no_connect_timeout():
    """SQLite 드라이버는 connect_timeout 을 모른다. 넘기지 않아야 열린다."""
    store = MemoryStore(database_url="sqlite://")
    store.init_schema()
    assert store.ensure_user() > 0
