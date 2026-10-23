"""deploy/schema.sql — 모델과 같은지, 실제 Postgres 에 두 번 넣어도 되는지.

클라우드 서버는 이 파일로 스키마를 만든다(deploy/update.sh). 모델만 고치고 다시 안 뽑으면
서버 DB 가 모델과 어긋나므로 여기서 막는다.
"""

import importlib.util
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from jarviseo.memory import MemoryStore
from jarviseo.memory.models import Base

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "schema_sql.py"
_spec = importlib.util.spec_from_file_location("schema_sql", _SCRIPT)
schema_sql = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(schema_sql)


def test_schema_sql_matches_models():
    """deploy/schema.sql 이 지금 모델에서 뽑은 것과 같다(다르면 scripts/schema_sql.py 를 다시)."""
    assert schema_sql.OUT.read_text(encoding="utf-8") == schema_sql.render()


@pytest.mark.postgres
def test_schema_sql_creates_all_tables_twice(postgres_url):
    """빈 Postgres 에 두 번 넣어도 오류 없이 모델의 테이블 전부와 pgvector 확장이 생긴다."""
    MemoryStore(database_url=postgres_url).drop_all()
    engine = create_engine(postgres_url)
    sql = schema_sql.OUT.read_text(encoding="utf-8")
    try:
        for _ in range(2):
            with engine.begin() as conn:
                conn.exec_driver_sql(sql)
        with engine.connect() as conn:
            tables = set(inspect(conn).get_table_names())
            vector = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
            has_vector = vector.scalar() == 1
    finally:
        MemoryStore(database_url=postgres_url).drop_all()
        engine.dispose()

    assert tables >= set(Base.metadata.tables)
    assert has_vector
