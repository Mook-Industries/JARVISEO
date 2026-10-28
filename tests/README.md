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
| `voice/test_mic.py` | `MicStream` 이 마이크 하나를 여럿이 같은 순서로 읽게 하는지, 지나간 블록부터 읽기, 블록 시각이 느리거나 빠른 장치·늦은 읽기·몰아 주는 드라이버에서도 실제 시각을 따라가는지(#69) |
| `voice/test_wakeword.py` | 호출어 감지·2초 중복 무시·재생 중 임계값 0.8, 감지 시각부터 `listen()` (가짜 점수로) |
| `voice/test_collector.py` | 답이 늦으면 "잠깐만요"를 먼저 읽고, 기다리는 동안 다시 부르면 늦은 답을 버리는지 |
| `memory/test_dialog.py` | 질문·대답이 뒤에서 observation 에 들어가고 기억 검색으로 찾히는지, 빈 대답은 안 넣고 실패해도 안 멈추는지 |
| `vlm/test_client.py` | VLM 요청 모양(사진·질문·기억)·실패하면 None·사진 줄이기, Fake VLM 이 실제와 같은 모양인지 (API 는 가짜로) |
| `workflow/test_state.py` | 단계별 지연·되묻기 대답이 덮이지 않고 합쳐지는지, 같은 단계는 시간을 더하는지 |
| `workflow/test_router.py` | 성분 → 기억 → 가리킴 → 나머지 순으로 의도를 정하는지, 띄어쓰기를 붙여 써도 같은지. 촬영 발화에서 놓쳤던 말(이건 · 쟤 · 저기 · 왼쪽꺼 · 단독 "뭐야?" · 알러지 · 들어 있어)과, 비슷해 보여도 GENERAL 이어야 하는 말(저기요 · 그건 그렇고 · 얘기 · 이 근처 · 앞으로 할 거 · 찾기 · 영역 · 장면 질문) (#147). 고르는 말(이 병 · 왼쪽 컵 · 작은 거 · 보라색 뭐야?)은 가리킴, 물건을 집지 않으면(책 추천 · 이 병원 · 작은 방) GENERAL, 붙여 쓴 말은 못 읽음 (#166) |
| `workflow/test_graph.py` | 질문 한 번이 그래프를 끝까지 도는지 — 사진·기억·호출어만 부르기·사과 문장·"잠깐만요"·답 버리기·대화 기억과 DB 기록 (Fake 로) |
| `workflow/nodes/test_pointing.py` | 가리킴 길(검출 → 대상 결정 ⇄ 되묻기 → 크롭)과 재촬영 — 크롭이 VLM 에 가는지, 되묻기 상한·한 글자 대답, 가리킴 기록, 손 단서(detect 가 가리키는 손만 상태에 두는지 · 실제 `resolve_target` 이 손 쪽을 되묻지 않고 고르는지 · 되묻기 중에도 손은 한 번만 찾는지 · 손 검출기가 없을 때) |
| `workflow/nodes/test_ingredient.py` | 성분 질문이 VLM 없이 성분 판정 문장으로 끝나는지 — 크롭·알레르기 전달, 부품·사진 없음, 재촬영 안내 |
| `tools/test_label_targets.py` | 가리킴 정답 표시 도구: `labels.csv` 읽고 쓰기(헤더 한 번 · 한글 · `0`/`1`), 이미 표시한 사진 건너뛰기, 후보를 왼쪽부터 번호, 미리보기 좌표를 원본으로 되돌리기, 한글 경로 사진 읽기 (YOLO · 화면 없이) |
| `tools/test_extract_frames.py` | 영상 → 학습 사진: 영상 번호 이름 규칙 · 번호 겹침, 시각으로 간격 맞추기(들쭉날쭉 · 끊김), 기본 3초 간격, dHash 중복 거르기(마지막으로 남긴 장과 비교), `frames.csv` 헤더 한 번, 합성 mp4 로 한 바퀴(못 쓰는 OS 는 건너뜀) |
| `tools/test_auto_label.py` | 자동 라벨 · 학습 묶음(ultralytics 없이, 가짜 선생): 종류 번호 순서 고정, 글귀 표, YOLO 좌표 · 겹친 박스 합치기, Label Studio 작업 · 설정(밝기 · 대비 조절), 선생 성적(박스 · 종류 · product), `label` 덮어쓰기 거부, `build` 영상 단위 분할 · Label Studio 번호를 이름으로 다시 맞추기 · `data.yaml` 에 `path` 없음 · 적은 종류 경고 · 회차 폴더 덮어쓰기 거부 |
| `scripts/test_run_ablation.py` | 가리킴 ablation: split · 세션(기본 1 · 2 라 나중에 붙은 세션이 안 섞임) · 제외 거르기, 미구현 버전 건너뛰기, 7분류 채점 순서, v1 · v2 실제 `resolve_target` 채점(가운데 · 좌우 대칭 되묻기 · 손이 가리킨 쪽 · 손 단서를 쓴 사진만 '방향 오판'), 지표(손 없는 사진이 없으면 None), `eval_run` · `eval_sample` 기록, 실행 조건(커밋 +dirty · git 없음 · 가중치 이름과 내용 지문 · 없는 가중치 경로), 같은 조건(가중치 지문 · 세션)의 이전 실행과만 비교(다른 가중치 · 다른 세션 · 조건 없는 옛 실행은 건너뜀, 커밋은 안 봄), 기준 가중치 대비 새로 잡힌 / 새로 놓친 사진, 사진별 CSV, 임시 버전 `v2+lang`(raw 사진의 색으로 고름, 사진이 없으면 색 기권) · `v2+lang+ctx`(정답표 `prev_label` 로 "그거"를 읽음) · `v2+lang+C`(실제 비서 설정 — 색 읽기와 손 우선 크롭이 같이 켜짐) |
| `pointing/test_crop.py` | 대상 크롭이 사진 경계를 넘거나 비어도 보낼 사진이 나오는지 |
| `pointing/test_detector.py` | 후보 검출이 사람·작은 박스를 빼고, YOLO 결과를 Detection 으로 바꾸는지, 이름이 달라도 겹친 박스를 하나로 줄이는 NMS 옵션(`agnostic_nms`)을 넘기는지, 가중치를 안 주면 설정(`JARVISEO_DETECTOR_WEIGHTS`)을 따르고 실제로 읽은 파일 위치를 남기는지, 한글 이름(`product` = 식품) (모델은 가짜로) |
| `pointing/test_hands.py` | 손 관절: 3D 굽힘 각도, 가리키는 손 판정(검지만 편 손 · 다 편 손 · 주먹 · 기준값), 검지 끝에서 뻗는 방향 · 각도, 여러 손 중 가리키는 손 고르기(없으면 기권), MediaPipe 결과를 픽셀 좌표로 바꾸기, 모델 파일 없음 안내. 모델 파일 · 평가셋 사진이 있으면 실제 사진 한 장으로 검지 끝 위치 확인(CI 는 건너뜀) |
| `pointing/test_cues.py` | 가운데 단서가 가운데·크게 찍힌 후보에, 손 방향 단서가 검지 선 위 후보에 높은 점수(0~1)를 주는지(σ 만큼 벗어나면 e^-0.5, 손 뒤쪽은 0 쪽). 라우터가 쓰는 낱말 역할 `lang_role`(종류 · 색 · 관계 · 위치 · 크기, 조사 떼기) |
| `pointing/test_resolver.py` | 대상 결정: 순위·margin·되묻기 문장(한글 이름이 없으면 영어, `product` 는 "식품"), 되묻기 대답으로 후보 좁히기, 단서 스위치, 손 단서(대칭 두 컵을 손으로 가름 · 손 없음/방향 없음이면 기권해 v1 과 같음 · 손 단서만 켰는데 손이 없으면 다시 가리켜 달라고 함) |
| `scripts/test_measure_latency.py` | 지연 리포트: 턴이 있는 최근 세션 고르기, 길(가리킴·성분·나머지) 나누기, 끊긴 턴 빼기, 30턴 미만 경고 |
| `scripts/test_check_hands.py` | 손 관절 측정: 사진별 가리키는 손 · 정답 박스 중심까지 각도, 손 있음/없음 나눠 세기 · 30° 경계, 가리키는 손이 없을 때, 표 한 줄, CSV · 한글 경로 그림 저장 (MediaPipe 없이) |

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
