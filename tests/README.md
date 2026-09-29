# tests — 자동 테스트

CI(`.github/workflows/ci.yml`)에서 ubuntu · windows · macOS 세 OS 로 돌아간다.

| 파일 | 무엇을 지키나 |
|---|---|
| `test_types_contract.py` | `types.py` 의 모듈 간 계약이 깨지지 않았는지 |
| `test_db_schema.py` | 테이블·관계·외래키가 실제로 만들어지는지 |

## 로컬에서 돌리는 법

```bash
ruff check .
ruff format --check .
pytest -q
```

**이 세 줄을 통과해야 PR 이 머지된다.** 푸시 전에 꼭 돌린다.

## 알아둘 것

- DB 테스트는 Postgres 를 띄우지 않고 SQLite 인메모리로 돈다.
  Postgres 에서만 터지는 문제(타입 불일치 등)는 여기서 안 잡힌다.
- 테스트 이름은 한글로 쓴다. 실패했을 때 무엇이 깨졌는지 바로 읽히게.
