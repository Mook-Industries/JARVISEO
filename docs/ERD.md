# 데이터베이스 설계 (ERD)

DBMS는 **PostgreSQL**. 원본은 팀 ERDCloud 다이어그램 **JARVISEO-v6** 이고,
이 문서는 그것을 그대로 옮긴 것이다. 다이어그램이 바뀌면 이 문서도 같이 고친다.

> ⚠️ **현재 `src/jarviseo/memory/models.py` 는 이 ERD 와 다른 스키마를 쓰고 있다.**
> 문서 맨 아래 "코드와의 불일치" 참고. 통합 전에 한쪽으로 맞춰야 한다.

테이블 13개, 네 덩어리다.

| 덩어리 | 테이블 | 담당 |
|---|---|---|
| 사용자 | `users` `user_setting` `user_allergen` | 공통 |
| 대화 | `chat_session` `session_turn` | 공통 |
| ① 가리킴 | `turn_inference` `turn_candidate` | 최홍묵 |
| ② 성분 | `turn_ingredient` `product` `allergen` `ingredient_synonym` | 권용현 |
| 평가 | `eval_run` `eval_sample` | 최홍묵 |

---

## 관계도

```mermaid
erDiagram
    users ||--o| user_setting : "설정을 가진다"
    users ||--o{ user_allergen : "알레르기를 등록한다"
    allergen ||--o{ user_allergen : "참조된다"
    allergen ||--o{ ingredient_synonym : "동의어를 가진다"
    users ||--o{ chat_session : "대화한다"
    chat_session ||--o{ session_turn : "턴을 가진다"
    session_turn ||--|| turn_inference : "추론 결과"
    session_turn ||--o{ turn_candidate : "후보 목록"
    session_turn ||--o| turn_ingredient : "성분 판정"
    product ||--o{ turn_ingredient : "조회된다"
    eval_run ||--o{ eval_sample : "샘플을 가진다"
    session_turn ||--o{ eval_sample : "실패 케이스로 참조"
```

`session_turn` 이 중심이다. 질문 한 번이 한 행이고 나머지가 달라붙는다.

---

## 사용자

### 사용자 · `users`

| 논리명 | 물리명 | 타입 | 설명 |
|---|---|---|---|
| 사용자 ID | `user_id` | BIGSERIAL **PK** | |
| 이메일 | `email` | VARCHAR(255) | |
| 비밀번호 해시 | `password_hash` | VARCHAR(60) | bcrypt. 평문 저장 금지 |
| 닉네임 | `nickname` | VARCHAR(50) NULL | |
| 권한 | `role` | VARCHAR(20) | `USER` / `ADMIN` — 성능 리포트 접근 제어 |
| 가입일시 | `created_at` | TIMESTAMPTZ | |
| 수정일시 | `updated_at` | TIMESTAMPTZ | |

### 사용자 설정 · `user_setting`

사용자당 1행. 음성·HUD 취향과 세션 정책.

| 논리명 | 물리명 | 타입 | 설명 |
|---|---|---|---|
| 사용자 ID | `user_id` | BIGINT **FK → users** | |
| 음성 크기 | `tts_volume` | SMALLINT | 기본 70 |
| 음성 지시문 | `tts_instructions` | TEXT | 말투·속도 |
| 보이스 | `tts_voice` | VARCHAR(50) | 기본 `ko-KR-SunHiNeural` |
| 세션 초기화 시간(분) | `session_timeout_min` | SMALLINT | 무응답 N분 경과 시 종료 (기본 10) |
| HUD 애니메이션 | `hud_animation` | BOOLEAN | |
| 대상 검출 표시 | `show_detection_box` | BOOLEAN | |
| 수정일시 | `updated_at` | TIMESTAMPTZ | |

### 사용자 알레르기 · `user_allergen`

| 논리명 | 물리명 | 타입 | 설명 |
|---|---|---|---|
| 사용자 알레르기 ID | `user_allergen_id` | BIGSERIAL **PK** | |
| 사용자 ID | `user_id` | BIGINT | |
| 알레르기 ID | `allergen_id2` | SMALLSERIAL NULL | → `allergen` |
| 엄격 모드 | `is_strict` | BOOLEAN | true면 미량·교차오염 의심도 경고 |
| 등록일시 | `created_at` | TIMESTAMPTZ | |
| 수정일시 | `updated_at` | TIMESTAMPTZ | |
| 삭제일시 | `deleted_at` | TIMESTAMPTZ NULL | 소프트 삭제. NULL이면 사용 중 |

### 알레르기 마스터 · `allergen`

| 논리명 | 물리명 | 타입 | 설명 |
|---|---|---|---|
| 알레르기 ID | `allergen_id` | SMALLSERIAL **PK** | |
| 알레르기명 | `name` | VARCHAR(50) **UK** | |
| 분류 | `category` | VARCHAR(20) | |

### 성분 동의어 사전 · `ingredient_synonym`

"탈지분유 → 우유" 같은 매핑. 성분 판정 recall 이 여기 품질에 달려 있다.

| 논리명 | 물리명 | 타입 | 설명 |
|---|---|---|---|
| 동의어 ID | `synonym_id` | BIGSERIAL **PK** | |
| 알레르기 ID | `allergen_id` | SMALLSERIAL | → `allergen` |
| 동의어 | `alias` | VARCHAR(100) **UQ** | |

---

## 대화

### 대화 세션 · `chat_session`

| 논리명 | 물리명 | 타입 | 설명 |
|---|---|---|---|
| 세션 ID | `session_id` | BIGSERIAL **PK** | |
| 사용자 ID | `user_id` | BIGINT NULL | 비로그인 사용도 가능해 NULL 허용 |
| 세션 제목 | `title` | VARCHAR(200) NULL | |
| 시작일시 | `started_at` | TIMESTAMPTZ | |
| 수정일시 | `updated_at` | TIMESTAMPTZ | |
| 종료일시 | `ended_at` | TIMESTAMPTZ NULL | |
| 종료 사유 | `end_reason` | VARCHAR(20) NULL | `TIMEOUT` / `USER` / `APP_CLOSE` |

### 질문·응답 턴 · `session_turn`

**가장 중요한 테이블.** 질문 한 번이 한 행이다.

| 논리명 | 물리명 | 타입 | 설명 |
|---|---|---|---|
| 턴 ID | `turn_id` | BIGSERIAL **PK** | |
| 세션 ID | `session_id` | BIGINT | → `chat_session` |
| 턴 번호 | `turn_no` | INTEGER | |
| 질문일시 | `asked_at` | TIMESTAMPTZ | |
| 트리거 방식 | `trigger_type` | VARCHAR(20) | `WAKEWORD` / `BARGE_IN` / `CLARIFY_REPLY` |
| 질문 내용 | `question_text` | TEXT | |
| 응답 내용 | `answer_text` | TEXT NULL | |
| 이미지 경로 | `image_path` | VARCHAR(500) NULL | 질문 시점 1장. 개인정보 이슈로 영상·연속 사진은 저장 안 함 |
| OCR 원문 | `ocr_text` | TEXT NULL | |
| 발화 시작 시각 | `speech_started_at` | TIMESTAMPTZ NULL | 프레임 선택 기준 |
| 발화 종료 시각 | `speech_ended_at` | TIMESTAMPTZ NULL | |
| STT 원문 | `stt_raw_text` | TEXT NULL | |
| STT 신뢰도 | `stt_confidence` | NUMERIC(4,3) NULL | |
| 재생 중단 여부 | `is_interrupted` | BOOLEAN NULL | |
| 선응답 여부 | `is_filler_sent` | BOOLEAN NULL | |
| 음성 파일 경로 | `audio_path` | VARCHAR(500) NULL | 개발·평가 모드에서만 저장. 운영 시 NULL |

---

## ① 가리킴 (담당 최홍묵)

### 턴 추론 결과 · `turn_inference`

`session_turn` 과 1:1 (PK를 FK로 공유하는 식별 관계).

| 논리명 | 물리명 | 타입 | 설명 |
|---|---|---|---|
| 턴 ID | `turn_id` | BIGINT **PK/FK → session_turn** | |
| 대상 라벨 | `target_label` | VARCHAR(50) NULL | |
| 확신도 | `confidence` | NUMERIC(4,3) NULL | |
| 마진 | `margin` | NUMERIC(4,3) NULL | 1위−2위. 임계값 미만이면 되묻기 |
| 가리킴 로직 버전 | `pointing_variant` | VARCHAR(20) NULL | v1~v5. 운영 데이터로 버전 비교 |
| 손끝 좌표 | `fingertip_xy` | JSONB NULL | 0~1 정규화. 벡터 재계산·디버깅 |
| 흐림 점수 | `blur_score` | NUMERIC(8,2) NULL | Laplacian variance |
| 최종 확정 주체 | `resolved_by` | VARCHAR(10) NULL | `USER` 면 되묻기로 사용자가 고른 것 = 정답 라벨로 활용 |
| 되묻기 여부 | `is_reask` | BOOLEAN | |
| 재촬영 여부 | `is_retake` | BOOLEAN | |
| 알레르기 경고 | `allergy_hit` | BOOLEAN NULL | |
| 음성 인식 지연 | `stt_ms` | INTEGER NULL | |
| 라우팅 지연 | `route_ms` | INTEGER NULL | |
| 비전 처리 지연 | `vision_ms` | INTEGER NULL | |
| LLM 지연 | `llm_ms` | INTEGER NULL | |
| TTS 첫 청크 지연 | `tts_ms` | INTEGER NULL | 텍스트 확정 → 첫 오디오 청크(TTFB). 전체 합성 시간 아님 |
| 전체 지연 | `total_ms` | INTEGER NULL | |
| 선택 후보 ID | `chosen_candidate_id` | BIGINT NULL | `turn_candidate.candidate_id` 참조. 되묻기 전 1위와 최종 확정을 구분해 **되묻기의 정확도 기여**를 계산 |
| 프레임 시차 | `frame_offset_ms` | INTEGER NULL | 발화 시작 시각 ↔ 선택된 프레임 캡처 시각 차이. 프레임 선택 로직 검증 근거 |
| 검출 개수 | `detected_count` | SMALLINT NULL | 후보가 1개뿐이면 `margin` 이 무의미하다. 난이도 구분용 |
| 검출 모델 버전 | `detector_version` | VARCHAR(30) NULL | YOLO 가중치 버전. 재학습하면 같은 `pointing_variant` 라도 결과가 달라진다 |

### 가리킴 후보 · `turn_candidate`

한 턴의 후보 물체들. 상위 N개를 남긴다.

| 논리명 | 물리명 | 타입 | 설명 |
|---|---|---|---|
| 후보 ID | `candidate_id` | BIGSERIAL **PK** | |
| 턴 ID | `turn_id` | BIGINT **FK → session_turn** | |
| 순위 | `rank` | SMALLINT | |
| 라벨 | `label` | VARCHAR(50) | |
| 점수 | `score` | NUMERIC(4,3) | |
| 박스 좌표 | `bbox` | JSONB | 0~1 정규화. 픽셀로 넣으면 해상도 바뀔 때 못 씀 |
| 선택 여부 | `is_chosen` | BOOLEAN | 최종 채택 1개만 true |
| 단서별 점수 | `cue_scores` | JSONB | `{"center":0.30,"point":0.82,"gaze":0.15,"lang":0.55,"ctx":0.00}` — 합산 `score` 만으로는 어느 단서가 이 후보를 밀어올렸는지 알 수 없다. ablation 해석과 실패 케이스 추적의 근거 |

---

## ② 성분 (담당 권용현)

### 턴 성분분석 결과 · `turn_ingredient`

| 논리명 | 물리명 | 타입 | 설명 |
|---|---|---|---|
| 턴 ID | `turn_id` | BIGINT **FK → session_turn** | |
| 바코드 | `barcode` | VARCHAR(20) | → `product` |
| 조회 경로 | `source` | VARCHAR(10) NULL | 캐시 / API / OCR |
| 판정 | `verdict` | VARCHAR(10) NULL | 위험 / 주의 / 안전 |
| 걸린 성분 | `matched` | JSONB NULL | |
| 정규화 성분 | `normalized` | JSONB NULL | |
| OCR 신뢰도 | `ocr_conf` | NUMERIC(4,3) NULL | |
| 재시도 횟수 | `retry_count` | SMALLINT NULL | |
| 응답 문장 | `response_text` | TEXT NULL | |
| OCR 지연 | `ocr_ms` | INTEGER NULL | |
| 판정 지연 | `match_ms` | INTEGER NULL | |
| 생성일시 | `created_at` | TIMESTAMPTZ NULL | |

### 바코드 제품 저장 · `product`

HACCP 공공데이터 / OCR 결과 캐시.

| 논리명 | 물리명 | 타입 | 설명 |
|---|---|---|---|
| 바코드 | `barcode` | VARCHAR(20) **PK** | |
| 제품명 | `product_name` | VARCHAR(200) | |
| 품목보고번호 | `report_no` | VARCHAR(30) | |
| 원재료 원문 | `raw_ingredients` | TEXT | |
| 원재료 목록 | `ingredients` | JSONB | |
| 알레르기 표시 문구 | `allergen_notice` | VARCHAR(500) NULL | |
| 교차오염 문구 | `cross_contamination` | VARCHAR(500) NULL | |
| 출처 | `source` | VARCHAR(10) NULL | |
| OCR 신뢰도 | `ocr_conf` | NUMERIC(4,3) NULL | |
| 생성·수정일시 | `created_at` `updated_at` | TIMESTAMPTZ NULL | |

---

## 평가 (담당 최홍묵)

### 실험 실행 · `eval_run`

| 논리명 | 물리명 | 타입 | 설명 |
|---|---|---|---|
| 실험 ID | `run_id` | BIGSERIAL **PK** | |
| 버전 | `variant` | VARCHAR(20) | v1~v5 |
| 실행일시 | `run_at` | TIMESTAMPTZ | |
| 샘플 수 | `sample_count` | INTEGER | |
| 지표 묶음 | `metrics` | JSONB | `{"accuracy":0.872,"reask_rate":0.14,"allergy_recall":0.96,"yolo_map50":0.731,"p50_ms":1840,"p95_ms":3120}` |
| 비고 | `note` | TEXT NULL | |

### 평가 샘플 · `eval_sample`

| 논리명 | 물리명 | 타입 | 설명 |
|---|---|---|---|
| 샘플 ID | `sample_id` | BIGSERIAL **PK** | |
| 실험 ID | `run_id` | BIGINT **FK → eval_run** | |
| 턴 ID | `turn_id` | BIGINT NULL **FK → session_turn** | 실사용 턴 재사용 시 연결. 실패 케이스에서 이미지·후보까지 추적 |
| 정답 라벨 | `gt_label` | VARCHAR(50) | |
| 예측 라벨 | `pred_label` | VARCHAR(50) NULL | |
| 정답 여부 | `is_correct` | BOOLEAN | |

---

## 코드 반영 상태

`src/jarviseo/memory/models.py` 는 이 ERD 의 13개 테이블을 그대로 옮긴 상태다.
`pytest tests/test_db_schema.py` 가 테이블 목록까지 확인한다.

**ERD 에 없어서 코드에서도 빠진 것**

- `belongings` · `observations` — ③④ 소지품 재인식 / 개인 기억의 관계형 앵커.
  지금은 Chroma(`memory/vector.py`)에만 있고 Postgres 쪽 대응 테이블이 없다.
  두 기능을 Should 로 남겨둘지, 테이블을 추가할지 팀에서 정해야 한다.
- 실행 장비 · 프레임 입력원 — 예전 스키마의 `sessions.device` / `frame_source`.
  벤치마크를 "맥북에서 잰 것"과 "Colab 에서 잰 것"으로 나눌 근거였다.
  당분간 `eval_run.metrics` 에 같이 적어 둔다.

**ERD 쪽에서 고쳐야 할 것** (2026-09-29 v6 기준)

기본값·코멘트가 다른 컬럼에서 복사돼 온 자리다. 그대로 DDL 로 뽑으면 터진다.

| 테이블 | 컬럼 | 문제 |
| --- | --- | --- |
| `user_setting` | `tts_instructions` | TEXT 인데 기본값이 `1.0` — 말투 지시문에 숫자 |
| `product` | `report_no` | 기본값 `false`, 코멘트가 `is_strict` 것 |
| `product` | `raw_ingredients` · `ingredients` | 기본값이 `now()` — 텍스트·JSONB 에 타임스탬프 |
| `product` | `allergen_notice` | 코멘트가 `deleted_at` 것("소프트 삭제 시각") |
| `user_allergen` | `allergen_id2` | 이름 끝의 `2`. 타입도 `SMALLSERIAL`(자동증가)인데 실제로는 참조값이라 `SMALLINT` 여야 함 |
| `ingredient_synonym` | `allergen_id` | 같은 문제. `SMALLSERIAL` → `SMALLINT` |
| `allergen` · `ingredient_synonym` | PK | PK 가 NULL 허용으로 표시돼 있음 |

**관계선이 빠진 곳** — 컬럼은 있는데 선이 안 그어져 있다.

- `chat_session.user_id` → `users`
- `session_turn.session_id` → `chat_session`
- `user_allergen.user_id` → `users`, `user_allergen.allergen_id2` → `allergen`
- `ingredient_synonym.allergen_id` → `allergen`
- `turn_ingredient.barcode` → `product`
- `user_setting.user_id` 는 선은 있으나 카디널리티가 1:N (`ZERO_OR_ONE_OR_MANY`). 1:1 이어야 한다

코드(`models.py`)에는 이 관계들이 전부 FK 로 걸려 있다. 그림만 맞추면 된다.
