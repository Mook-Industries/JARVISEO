# tests — 자동 테스트

CI(`.github/workflows/ci.yml`)에서 ubuntu · windows · macOS 세 OS 로 돌아간다.
DB 테스트는 여기에 더해 db job 이 pgvector 컨테이너를 띄워 실제 Postgres 에서 한 번 더 돌린다.

| 파일 | 무엇을 지키나 |
|---|---|
| `test_types_contract.py` | `types.py` 의 모듈 간 계약이 깨지지 않았는지 |
| `test_db_schema.py` | 테이블·관계·외래키가 실제로 만들어지는지 |
| `test_memory_embed.py` | 임베딩 요청 모양과 Fake 임베딩이 실제 클래스와 같은 모양인지 (API 는 가짜로) |
| `test_memory_vector.py` | 기억 검색: 저장·가까운 순·점수 문턱·사용자 구분·임베딩 실패 |
| `test_voice_fake.py` | Fake STT·TTS 가 실제 클래스와 같은 모양인지 |
| `test_voice_vad.py` | VAD 가 발화를 맞게 잘라 내는지 (마이크 없이) |
| `test_voice_stt.py` | STT 요청 모양과 받아쓴 글자 정리 (API 는 가짜로) |
| `test_voice_tts.py` | TTS 요청 모양·속도 변환·읽을 글자 다듬기와 재생, 스트림 끊김 (API·스피커는 가짜로) |
| `test_voice_gate.py` | 재생 중 STT 입력 차단·0.3초 유예와 호출어 훅 (마이크·스피커 없이) |

## 로컬에서 돌리는 법

```bash
ruff check .
ruff format --check .
pytest -q
```

**이 세 줄을 통과해야 PR 이 머지된다.** 푸시 전에 꼭 돌린다.

## 알아둘 것

- DB 테스트(`store` fixture, `conftest.py`)는 SQLite 인메모리와 실제 Postgres 에서 한 번씩 돈다.
  Postgres 쪽은 `JARVISEO_TEST_DATABASE_URL` 이 없거나 붙을 수 없으면 건너뛴다(CI 에서는 실패).
  테스트마다 테이블을 지우고 다시 만드니 **개발 DB 말고 테스트용 DB 를 따로** 가리킨다.

  ```bash
  docker run -d --rm --name jarviseo-test-pg -p 5433:5432     -e POSTGRES_USER=jarviseo -e POSTGRES_PASSWORD=jarviseo -e POSTGRES_DB=jarviseo_test     pgvector/pgvector:pg16
  JARVISEO_TEST_DATABASE_URL=postgresql+psycopg://jarviseo:jarviseo@localhost:5433/jarviseo_test     pytest -q -m postgres
  docker stop jarviseo-test-pg
  ```
- 테스트 이름은 한글로 쓴다. 실패했을 때 무엇이 깨졌는지 바로 읽히게.
