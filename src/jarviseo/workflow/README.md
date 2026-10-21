# workflow — LangGraph 오케스트레이션 · 담당 최홍묵

질문 한 번이 `graph.invoke()` 한 번이다. 노드와 분기 그림은 `graph.py` 머리말에 있다.

```
listen(STT + 말을 시작한 순간의 사진) ─▶ (recapture ↺) ─▶ route(키워드 규칙)
  ├ RECALL      : recall ─▶ generate
  ├ POINTING    : detect ─▶ resolve ⇄ clarify ─▶ crop_target ─▶ generate
  ├ INGREDIENT  : detect ─▶ resolve ⇄ clarify ─▶ crop_target ─▶ check_ingredient   (VLM 안 부름)
  └ GENERAL     : generate (전체 사진)
─▶ speak ─▶ log
```

| 파일 | 하는 일 |
|---|---|
| `graph.py` | 조립만 한다. 노드를 잇고, 분기의 결과 이름 → 다음 노드 매핑을 정하고, 부품을 `partial` 로 묶는다 |
| `nodes/` | 기능별 노드 파일. 노드 뒤 분기 함수도 그 파일에 있다 (아래 표) |
| `deps.py` | 부품 묶음 `GraphDeps` 와 "가리킬 수 있나" 판단 `can_point` |
| `messages.py` | 미리 정한 안내 문장과 그 답을 만드는 `canned_response` |
| `state.py` | 그래프를 흐르는 상태 = **노드 사이의 계약**. 리듀서 `merge_latency` |
| `router.py` | 질문 글자로 의도 정하기 `classify_intent()` (성분 → 기억 → 가리킴 → 나머지) |

| `nodes/` 파일 | 노드 | 분기가 돌려주는 결과 |
|---|---|---|
| `voice.py` | `listen` · `speak` | `no_question` / `bad_frame` / `heard` |
| `capture.py` | `recapture` | `gave_up` / `still_bad` / `good` |
| `intent.py` | `route` | 의도 뒤 `recall` / `needs_target` / `ingredient` / `answer`, 크롭 뒤 `ingredient` / `answer` |
| `pointing.py` | `detect` · `resolve` · `clarify` · `crop_target` | `ambiguous` / `chosen`, `called_again` / `gave_up` / `replied` |
| `ingredient.py` | `check_ingredient` | — |
| `answer.py` | `generate` | `dropped` / `answered` |
| `memory.py` | `recall` · `log` | — |

## 부품 (`GraphDeps`)

실행 스크립트(`scripts/assistant.py`)는 실제 것을, 테스트는 Fake 를 넣는다. 없어도 되는 칸은 None 이다.

| 칸 | 없으면 | 실제 | 테스트 |
|---|---|---|---|
| `frames` | 사진 없이 묻는다. 가리킴·재촬영도 안 한다 | `FrameBuffer` | `FrameBuffer` 에 직접 넣는다 |
| `detector` · `resolve` | 지시어 질문은 전체 사진으로 VLM 이, 성분 질문은 사진 전체로 판정한다 | `ObjectDetector` · `resolve_target` | 테스트 파일 안의 작은 가짜 |
| `hands` | 손 단서가 기권한다(가운데 단서만, v1 처럼). `assistant.py` 는 모델 파일 · mediapipe 가 없으면 이 칸을 비우고 "손 단서 꺼짐"을 찍는다 | `HandDetector` — `detect` 노드가 YOLO 와 같이 한 번 찾고 가리키는 손만 상태 `hand` 에 둔다. 되묻기로 `resolve` 가 다시 돌아도 다시 안 찾는다 | 테스트 파일 안의 작은 가짜(부른 횟수를 셈) |
| `checker` | 성분 질문에 "확인할 수 없어요"로 답한다. **VLM 으로 넘기지 않는다** | `IngredientChecker` | 테스트 파일 안의 작은 가짜 |
| `memory` | 기억 검색 결과가 빈 목록이다 | `VectorMemory` | `FakeEmbedder` 를 쓴 `VectorMemory` |
| `store` · `dialog` | 턴·성분 기록과 대화 기억 저장을 안 한다. 알레르기 목록도 비어 있다 | `MemoryStore` · `DialogMemory` | SQLite `MemoryStore` |
| `collector` | "잠깐만요" 없이 답을 기다린다 | `ResponseCollector` | `FakeTextToSpeech` 를 쓴 `ResponseCollector` |
| `called_since` | 되묻기 대답을 기다리는 동안 다시 불러도 그 말을 대답으로 받는다 | `BargeIn.called_since` | `lambda t: …` |
| `sharpness_threshold` | 흐림은 보지 않는다(사진이 없을 때만 다시 고른다) | 안경 카메라로 잰 값 | `100.0` |

## 알아둘 것

- **노드는 각 모듈의 함수를 부르기만 한다.** 로직을 여기 넣지 않는다.
  그래야 각자 자기 모듈만 고쳐도 되고 이 폴더에서 충돌이 안 난다.
- 노드는 `(state, deps)` 를 받는 보통 함수다. `build_graph()` 가 `functools.partial` 로 `deps` 를 묶어 넣는다.
  상태에는 이번 질문에서 바뀌는 값만 둔다.
- **분기 함수는 다음 노드 이름이 아니라 결과 이름을 돌려준다**(예: `ambiguous`). 어느 노드로 갈지는 `graph.py`
  매핑이 정한다. 노드 파일이 다른 기능의 노드 이름을 몰라도 되고, 흐름을 바꿀 때 `graph.py` 한 곳만 고친다.
- **의도 분류는 키워드 규칙이다. LLM 을 부르지 않는다.** 말 끝에서 대답 첫 소리까지 3초 목표 안에
  API 왕복을 하나 더 넣을 자리가 없다.
- 되돌아가는 분기 두 개: 사진이 없거나 흐리면 `recapture`(최대 2번), 대상이 모호하면 `clarify`(최대 2번).
  상한을 넘으면 안내를 읽고 끝낸다. 되묻기는 대답까지 그래프 안에서 듣고, 사진은 다시 고르지 않는다.
  되묻는 중이나 대답을 기다리는 중에 "자비서"라고 다시 부르면(`called_again`) 대답으로 받지 않고 빈 대답으로
  기록만 한다. `interrupted` 라서 그 호출은 다음 턴(BARGE_IN)이 새 질문으로 받는다. 대답을 기다리는 중이면
  5초를 다 채우지 않고 그 자리에서 듣기를 멈춘다(`listen(stop=...)`).
  되묻기 대답은 "컵"처럼 한 글자여도 받는다(`listen(min_letters=1)`).
- 턴이 끝나면 질문·대답을 대화 기억(`DialogMemory`)에 맡긴다. 빈 대답, 미리 정한 안내 문장(`CANNED`),
  기억을 물은 턴은 넣지 않는다. 새 내용이 없고, 넣으면 다음 기억 검색에서 원래 기록보다 먼저 찾힌다.
- VLM 과 성분 판정에는 가리킨 대상의 **크롭**이 간다. 기록(`used_frame`)에는 원본 장면이 남는다.
- **성분 질문은 VLM 을 부르지 않는다.** 판정은 규칙(`nutrition/`)이 내리고 그 문장을 그대로 읽는다.
  재촬영 안내가 와도 그 문장을 읽고 턴을 끝낸다.
- 지연은 단계 이름(`route` `detect` `resolve` `crop` `ingredient` `stt` `recall` `vlm` `tts` `total`)으로 모은다.
  `memory/store.py` 의 `STAGE_COLUMNS` 에 있는 이름은 `log_turn` 이 그대로 DB 열에 적는다.
  말하거나 듣는 시간이 안 섞이는 노드는 `_timed` 가 재고, 듣기·기억 검색·답 생성·읽기는 부품이 잰 값(`last_latency_ms`)을 쓴다.
- 노드 이름은 상태 키와 다르게 짓는다. langgraph 1.2 는 같아도 받아 주지만 그림·로그에서 노드인지 값인지 헷갈린다.
  그래서 상태 키 `crop` · `ingredient` 를 만드는 노드는 `crop_target` · `check_ingredient` 다.
- `build_graph()` 가 재귀 한도 30 을 걸어서 준다. langgraph 1.2 의 기본값(10007)으로는 루프 버그를 못 잡는다.
- `state.py` 필드 추가는 `types.py` 와 같은 규칙을 따른다.
