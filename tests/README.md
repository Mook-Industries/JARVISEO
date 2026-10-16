# tests — 자동 테스트

CI(`.github/workflows/ci.yml`)에서 ubuntu · windows · macOS 세 OS 로 돌아간다.
DB 테스트는 여기에 더해 db job 이 pgvector 컨테이너를 띄워 실제 Postgres 에서 한 번 더 돌린다.

테스트 위치는 **시험하는 코드가 있는 폴더**를 따른다. `src/jarviseo/pointing/resolver.py` 의 테스트는 `tests/pointing/test_resolver.py` 다.
`tests/` 바로 아래에는 `src/jarviseo/` 바로 아래 모듈(`types.py` · `config.py`)의 테스트, 공통 fixture(`conftest.py`), 공통 데이터(`data/`)만 둔다.
여러 모듈을 가로질러 주인이 없는 흐름 테스트가 생기면 `tests/integration/` 에 둔다. 모듈 하나만 돌릴 때는 `pytest tests/voice` 처럼 폴더를 준다.

| 파일 | 무엇을 지키나 |
|---|---|
| `test_types_contract.py` | `types.py` 의 모듈 간 계약이 깨지지 않았는지 |
| `memory/test_db_schema.py` | 테이블·관계·외래키가 실제로 만들어지는지 |
| `memory/test_embed.py` | 임베딩 요청 모양과 Fake 임베딩이 실제 클래스와 같은 모양인지 (API 는 가짜로) |
| `memory/test_vector.py` | 기억 검색: 저장·가까운 순·점수 문턱·사용자 구분·임베딩 실패 |
| `voice/test_fake.py` | Fake STT·TTS 가 실제 클래스와 같은 모양인지 |
| `voice/test_vad.py` | VAD 가 발화를 맞게 잘라 내는지 (마이크 없이) |
| `voice/test_stt.py` | STT 요청 모양과 받아쓴 글자 정리 (API 는 가짜로) |
| `voice/test_tts.py` | TTS 요청 모양·속도 변환·읽을 글자 다듬기와 재생, 스트림 끊김 (API·스피커는 가짜로) |
| `voice/test_gate.py` | 재생 중 STT 입력 차단·0.3초 유예와 호출어 훅 (마이크·스피커 없이) |
| `voice/test_mic.py` | `MicStream` 이 마이크 하나를 여럿이 같은 순서로 읽게 하는지, 지나간 블록부터 읽기 |
| `voice/test_wakeword.py` | 호출어 감지·2초 중복 무시·재생 중 임계값 0.8, 감지 시각부터 `listen()` (가짜 점수로) |
| `voice/test_collector.py` | 답이 늦으면 "잠깐만요"를 먼저 읽고, 기다리는 동안 다시 부르면 늦은 답을 버리는지 |
| `memory/test_dialog.py` | 질문·대답이 뒤에서 observation 에 들어가고 기억 검색으로 찾히는지, 빈 대답은 안 넣고 실패해도 안 멈추는지 |
| `vlm/test_client.py` | VLM 요청 모양(사진·질문·기억)·실패하면 None·사진 줄이기, Fake VLM 이 실제와 같은 모양인지 (API 는 가짜로) |
| `workflow/test_state.py` | 단계별 지연·되묻기 대답이 덮이지 않고 합쳐지는지, 같은 단계는 시간을 더하는지 |
| `workflow/test_router.py` | 성분 → 기억 → 가리킴 → 나머지 순으로 의도를 정하는지, 띄어쓰기를 붙여 써도 같은지 |
| `workflow/test_graph.py` | 질문 한 번이 그래프를 끝까지 도는지 — 사진·기억·호출어만 부르기·사과 문장·"잠깐만요"·답 버리기·대화 기억과 DB 기록 (Fake 로) |
| `workflow/nodes/test_pointing.py` | 가리킴 길(검출 → 대상 결정 ⇄ 되묻기 → 크롭)과 재촬영 — 크롭이 VLM 에 가는지, 되묻기 상한·한 글자 대답, 가리킴 기록 |
| `workflow/nodes/test_ingredient.py` | 성분 질문이 VLM 없이 성분 판정 문장으로 끝나는지 — 크롭·알레르기 전달, 부품·사진 없음, 재촬영 안내 |
| `pointing/test_crop.py` | 대상 크롭이 사진 경계를 넘거나 비어도 보낼 사진이 나오는지 |
| `pointing/test_detector.py` | 후보 검출이 사람·작은 박스를 빼고, YOLO 결과를 Detection 으로 바꾸는지 (모델은 가짜로) |
| `pointing/test_cues.py` | 가운데 단서가 가운데·크게 찍힌 후보에 높은 점수(0~1)를 주는지 |
| `pointing/test_resolver.py` | 대상 결정: 순위·margin·되묻기 문장, 되묻기 대답으로 후보 좁히기, 단서 스위치 |

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
- 테스트 함수 이름은 영어 snake_case `test_<대상>_<상황이나 기대>` 로 쓰고, 무엇을 확인하는지는 바로 아래 한글 독스트링에 쓴다.
  예: `test_filter_drops_person` + `"""사용자 손·팔이 person 으로 잡혀도 후보에 들어가지 않는다."""`.
  실패 화면에 함수 소스가 독스트링째 나와서 무엇이 깨졌는지 바로 읽힌다. 한 파일 안에 같은 이름이 두 번 나오면 앞의 테스트가 조용히 사라지니 ruff `F811` 을 지킨다.
- 테스트 데이터(`tests/data/`)는 하위 폴더에서 `Path(__file__).parents[1] / "data"` 로 찾는다. 폴더가 한 단계 깊어서 `.parent` 로는 못 찾는다.
