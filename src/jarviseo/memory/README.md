# memory — ④ 개인 기억 + DB 저장소 · 담당 문태현

관계형 기록과 벡터 검색을 Postgres 하나(pgvector 확장)로 다룬다.
**모든 팀원이 결과를 적재하는 곳**이라 스키마 변경은 팀 합의 후에 한다.

| 파일 | 하는 일 |
|---|---|
| `models.py` | 테이블 정의 — ERD 를 코드로 옮긴 것 |
| `store.py` | Postgres 접근 (대화 기록 · 지연시간 · 판정 결과) |
| `vector.py` | pgvector 검색 — `observation` 테이블 |
| `embed.py` | 문장 임베딩 (OpenAI `text-embedding-3-small`) + 키 없이 쓰는 `FakeEmbedder` |

## 알아둘 것

- **원본은 팀 ERD(`docs/ERD.md`, ERDCloud `JARVISEO-v9`)다.** 어긋나면 ERD 가 맞다.
  세 사람이 같은 그림을 보고 작업해야 하므로 다이어그램을 기준으로 삼는다.
- 이미지 파일은 DB 에 넣지 않고 경로(`image_path`)만 둔다. 임베딩은 같은 행의 `embedding` 열(pgvector)에 둔다.
- Enum 은 문자열로 저장한다. Postgres ENUM 은 값 추가마다 마이그레이션이 필요하다.
- 테이블 생성: `python -c "from jarviseo.memory import MemoryStore; MemoryStore().init_schema()"`
