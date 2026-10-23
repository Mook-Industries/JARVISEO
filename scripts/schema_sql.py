"""DB 스키마(memory/models.py)를 Postgres SQL 로 뽑아 deploy/schema.sql 에 쓴다.  담당: 문태현

클라우드 서버의 대시보드 이미지에는 jarviseo.memory 를 돌릴 패키지(pgvector 등)가 없어서
``MemoryStore().init_schema()`` 를 거기서 부를 수 없다. 대신 같은 스키마를 SQL 로 만들어 두고,
서버가 배포할 때마다 psql 로 흘려 넣는다(deploy/update.sh). 없는 테이블 · 인덱스만 만들고
있는 것은 건드리지 않으므로 몇 번을 넣어도 된다. init_schema 와 같은 성질이다.

모델을 바꿨으면 다시 뽑아 같이 커밋한다. 안 바꾸면 tests/scripts/test_schema_sql.py 가 알려 준다.

    python scripts/schema_sql.py              # deploy/schema.sql 을 새로 쓴다
    python scripts/schema_sql.py --check      # 파일이 모델과 같은지만 본다(다르면 끝 코드 1)
"""

import argparse
import sys
from pathlib import Path

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from jarviseo.memory.models import Base

OUT = Path(__file__).resolve().parents[1] / "deploy" / "schema.sql"
HEADER = """\
-- scripts/schema_sql.py 가 memory/models.py 에서 뽑은 것. 손으로 고치지 않는다.
-- 없는 테이블 · 인덱스만 만든다. 열을 바꾼 것은 반영되지 않는다(init_schema 와 같다).
CREATE EXTENSION IF NOT EXISTS vector;
"""


def render() -> str:
    dialect = postgresql.dialect()
    parts = [HEADER]
    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table, if_not_exists=True).compile(dialect=dialect)).strip()
        parts.append(ddl + ";\n")
        for index in sorted(table.indexes, key=lambda i: i.name or ""):
            parts.append(
                str(CreateIndex(index, if_not_exists=True).compile(dialect=dialect)) + ";\n"
            )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="모델에서 Postgres 스키마 SQL 을 뽑는다")
    parser.add_argument("--check", action="store_true", help="파일이 모델과 같은지만 본다")
    args = parser.parse_args()
    sql = render()
    if args.check:
        same = OUT.is_file() and OUT.read_text(encoding="utf-8") == sql
        print(
            "같음"
            if same
            else f"다름 — python scripts/schema_sql.py 로 {OUT.name} 를 다시 뽑으세요"
        )
        sys.exit(0 if same else 1)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(sql.encode("utf-8"))
    print(f"{OUT} ({len(Base.metadata.sorted_tables)}개 테이블)")


if __name__ == "__main__":
    main()
