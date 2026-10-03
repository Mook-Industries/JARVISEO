# 자비서 (JARVISEO)

**팀 무크 인더스트리** · KDT2604 최종 프로젝트

일반 안경테에 초소형 USB 카메라를 붙인 착용형 AI 비서.
"자비서, 저거 뭐야?" 하고 부르면, 카메라가 본 장면을 판단해 음성으로 답한다.

> 🚀 **처음 참여하는 팀원은 [docs/START_HERE.md](docs/START_HERE.md) 부터 보세요.**
> 설치 · 매일 쓰는 git 명령 · 내가 건드릴 폴더가 한 장에 정리돼 있습니다.
> 이 README 의 나머지는 나중에 필요할 때 찾아보면 됩니다.

---

## 구조

질문 주제에는 제한이 없다. 범용 멀티모달 모델이 기본 응답을 맡고,
그 위에 범용 모델이 하지 못하는 네 가지를 특화 레이어로 얹는 2계층이다.

```
사용자 질문 (주제 무제한)
        │
        ▼
[기본 계층 · 항상 동작]  VLM 호출 → 무엇이든 응답
        │  + 해당되는 특화 모듈이 있으면 맥락을 덧붙임
        │  (해당 없으면 기본 응답이 그대로 나감)
        ▼
 ① 지시 대상 특정   손·시선·언어 단서로 '저거'를 결정      [Must]
 ② 식품 성분 판정   성분표 검출 + OCR + 개인 알레르기      [Must]
 ③ 소지품 재인식    등록한 개인 물품 임베딩 매칭           [Should]
 ④ 개인 기억 검색   '아까 본 그거' 과거 관찰 기록 조회      [Should]
```

파이프라인:

```
호출어 감지 → 발화 시작 시점 프레임 캡처 → STT → 의도 라우팅
  → (병렬) 객체 검출 · OCR · 기억 검색 → 지시 대상 특정
  → VLM 응답 생성 → TTS
```

LangGraph 가 전체를 묶는다. 프레임이 흐리면 `recapture` 로 되돌아가고,
지시 대상이 모호하면 `clarify` 로 빠져 사용자에게 되묻는다.

---

## 폴더 구조

```
src/jarviseo/
├── types.py          ★ 모듈 간 인터페이스 계약 — 가장 먼저 읽을 것
├── config.py           전역 설정 (.env 로드, 경로)
├── capture/            프레임 입력 추상화 + 선명도 판단
├── pointing/         ① 지시 대상 특정        (담당 최홍묵)
├── nutrition/        ② 식품 성분 판정        (담당 권용현)
├── memory/           ③④ 소지품 · 개인 기억   (담당 문태현)
├── voice/              웨이크워드 · STT · TTS (담당 문태현)
├── vlm/                클라우드 VLM 호출
├── graph/              LangGraph 오케스트레이션 (담당 최홍묵)
└── api/                FastAPI · 대시보드      (담당 권용현)

data/        학습 데이터 · 가중치 · 영상 (공용 Drive 에서 받음, Git 제외)
notebooks/   Colab 학습 노트북
scripts/spike/  기술 검증 5종
docs/        회의록 · 실험 기록
tests/
```

---

## 시작하기

파이썬은 **3.12** 로 통일한다. 강의에서 쓴 버전(3.12.5)이라 수업 코드가
그대로 돌아간다. 3.13 은 쓰지 않는다 — ultralytics·easyocr 가 새 파이썬을
늦게 따라와서 설치가 막히는 일이 있다.

```bash
git clone <레포 주소>
cd JARVISEO
```

세 단계다. 파이썬은 각자 깔고, **도커는 데이터베이스 하나만 띄운다.**

### 1. 파이썬 환경

**팀 기본은 venv 다.** 파이썬에 기본으로 들어 있어서 따로 설치할 게 없다.

```bash
python -m venv .venv
source .venv/bin/activate        # 윈도우: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

conda 를 쓰는 사람은 앞의 두 줄만 이걸로 바꾼다. 나머지는 같다.

```bash
conda create -n JARVISEO python=3.12 -y && conda activate JARVISEO
```

### 2. 데이터베이스 (도커)

Postgres 만 컨테이너로 띄운다. 셋이 각자 설치하고 버전 맞추는 것보다 싸다.

처음이면 [처음 DB 띄우기](#처음-db-띄우기--erd-대로-테이블-만들기)를 0단계부터 따라 한다. 해 본 적 있으면 이 두 줄이다.

```bash
docker compose up -d db
python -c "from jarviseo.memory import MemoryStore; MemoryStore().init_schema()"
```

> **도커가 하는 일은 이게 전부다.** 파이썬은 위에서 만든 환경에서 그대로 돈다.
> 카메라·마이크도 네이티브로 도니까 아무 제약이 없다.

### 3. 환경변수

```bash
cp .env.example .env
```

`.env` 를 열어 `OPENAI_API_KEY` 를 채운다. 키는 팀 채널에서 공유받는다.
**`.env` 는 절대 커밋하지 않는다.**

카메라 없이 개발하려면 입력원을 바꾼다.

```
JARVISEO_FRAME_SOURCE=folder
JARVISEO_FRAME_PATH=data/datasets/sample
```

---

### (비상구) 설치가 도저히 안 될 때

윈도우에서 easyocr·opencv 설치가 며칠째 막히는 팀원이 있으면, 파이썬 환경까지
통째로 컨테이너로 옮길 수 있다. **위 1번 대신** 쓰는 방법이고,
잘 깔린 사람은 볼 필요 없다.

```bash
docker compose build          # 처음 한 번, 5~10분
docker compose run --rm dev   # 컨테이너 셸
```

```bash
docker compose run --rm dev pytest
docker compose up api          # 대시보드 → http://localhost:8000
```

**이 컨테이너 안에서는 카메라·마이크가 안 된다.** 맥과 윈도우의 Docker 는
리눅스 VM 안에서 돌아서 USB 장치를 넘길 수 없다. 설정 문제가 아니라 구조라
우회할 수 없다. 그래서 이걸 쓰는 사람은 파일 입력원(`folder` / `video`)으로만
개발한다. 어차피 카메라는 한 대뿐이라 대부분의 작업이 파일 입력으로 돌아간다.

---

## 데이터베이스

**Postgres 를 docker compose 로 띄운다.** 네이티브로 파이썬을 돌리더라도
DB 만큼은 컨테이너를 쓴다. 셋이 각자 Postgres 를 설치하고 버전을 맞추는
것보다 이쪽이 훨씬 싸다.

### 처음 DB 띄우기 — ERD 대로 테이블 만들기

처음 해 보는 사람 기준으로 적었다. 명령은 전부 **레포 루트 폴더**(이 README 가 있는 곳)에서 친다.
맥은 터미널, 윈도우는 PowerShell 이면 된다.

용어 두 개만 먼저.
- **이미지**: DB 프로그램이 통째로 들어 있는 설치 파일 같은 것. 여기서는 `pgvector/pgvector:pg16`(Postgres 16 + 벡터 검색 확장).
- **컨테이너**: 그 이미지를 실제로 실행한 것. 내 컴퓨터에 Postgres 를 직접 설치하지 않아도 DB 가 하나 떠 있게 된다.

**0단계 — Docker Desktop 켜기**

Docker Desktop 을 설치하고 **실행해 둔다**(맥은 메뉴 막대, 윈도우는 작업 표시줄에 고래 아이콘).
꺼져 있으면 아래 명령이 전부 `Cannot connect to the Docker daemon` 으로 실패한다.

```bash
docker --version
```

버전이 나오면 준비 끝.

**1단계 — DB 컨테이너 띄우기**

```bash
docker compose up -d db
```

`docker-compose.yml` 의 `db` 서비스만 띄운다. `-d` 는 "뒤에서 돌게 두고 터미널은 돌려줘"라는 뜻이다.
처음에는 이미지를 내려받느라 1~2분 걸린다.

```bash
docker compose ps
```

`STATUS` 에 `healthy` 가 보이면 DB 가 준비된 것이다. `starting` 이면 몇 초 뒤에 다시 본다.

> ⚠️ **예전에 `postgres:16-alpine` 이미지로 띄운 적이 있으면** 먼저 한 번 비운다.
> 이미지가 pgvector 판으로 바뀌어서, 옛 데이터 폴더를 그대로 쓰면 꼬일 수 있다.
> ```bash
> docker compose down -v
> ```
> `-v` 는 **DB 안의 데이터까지 지운다.** 개발용 데이터라 지워도 되지만, 남겨야 할 게 있으면 먼저 말한다.

**2단계 — 테이블 만들기**

먼저 [시작하기 1번](#1-파이썬-환경)에서 만든 파이썬 환경을 켠다
(venv 는 `source .venv/bin/activate` · 윈도우 `.venv\Scripts\activate`, conda 는 `conda activate JARVISEO`).

```bash
python -c "from jarviseo.memory import MemoryStore; MemoryStore().init_schema()"
```

`models.py`(ERD 를 코드로 옮긴 것)를 읽어서 **없는 테이블만** 만든다. 아무것도 출력되지 않으면 성공이다.
임베딩 열(`VECTOR`)에 필요한 pgvector 확장도 이 명령이 먼저 켜 준다.

**3단계 — 확인하기**

```bash
docker compose exec db psql -U jarviseo -d jarviseo -c "\dt"
```

`docker compose exec db` 는 "떠 있는 db 컨테이너 안에서 명령을 실행해줘"이고, `psql` 은 Postgres 에 SQL 을 보내는 프로그램이다.
**ERD 와 같은 테이블 17개**가 나오면 성공이다. 그림으로 보고 싶으면 DBeaver 같은 DB 도구로 접속한다.

| 항목 | 값 |
|---|---|
| Host | `localhost` |
| Port | `5432` |
| Database · User · Password | 전부 `jarviseo` |

> **ERD 가 바뀌어 `models.py` 가 따라 바뀌면** `docker compose down -v` 로 비우고 1~2단계를 다시 한다.
> 2단계는 없는 테이블만 만들어서, 이미 있는 테이블의 열은 바꾸지 않기 때문이다.

**자주 막히는 곳**

| 증상 | 원인 · 해결 |
|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop 이 꺼져 있다. 켜고 고래 아이콘이 멈출 때까지 기다린다 |
| `port is already allocated` / `5432` 사용 중 | 내 컴퓨터에 Postgres 가 따로 설치돼 돌고 있다. 그걸 끄거나, 팀에 말해 포트를 바꾼다 |
| 2단계에서 `connection refused` | DB 가 아직 안 떴다. `docker compose ps` 로 `healthy` 를 확인하고 다시 한다 |
| 2단계에서 `No module named jarviseo` | 파이썬 환경이 안 켜졌거나 `pip install -e .` 를 안 했다 |
| `type "vector" does not exist` | 옛 `postgres:16-alpine` 컨테이너가 떠 있다. 1단계의 ⚠️ 대로 `down -v` 후 다시 띄운다 |

### 끄고 비우기

```bash
docker compose stop        # 잠깐 끄기. 데이터 남음
docker compose down        # 컨테이너 지우기. 데이터 남음
docker compose down -v     # 데이터까지 전부 지우기 — 스키마를 갈아엎었을 때
```

### 두 저장소를 나눠 쓴다

| | 무엇을 | 왜 |
|---|---|---|
| **Postgres** | 대화 로그, 단계별 지연, 판정 결과 | "몇 번째 턴의 무엇"이 분명한 정형 기록 |
| **pgvector** | 소지품·관찰 기록 임베딩 | "아까 본 그거"처럼 이름을 모르는 질의용 |

pgvector 는 Postgres 의 확장이라 DB 를 하나 더 띄우지 않는다.
임베딩은 그 기록과 같은 행에 둔다 —
벡터 DB 를 따로 두면 두 곳이 어긋났을 때 어느 쪽이 맞는지 알 수 없다.

### 스키마

**원본은 ERDCloud `JARVISEO` 다이어그램이고, `src/jarviseo/memory/models.py` 가 그것을 따른다.**
사람이 읽는 사본은 `docs/ERD.md`, 그림은 `docs/erd-diagram/JARVISEO-ERD.png` 다.
셋이 어긋나면 ERD 가 맞다 — 세 사람이 같은 그림을 보고 작업해야 하기 때문이다.

열을 추가·변경할 때 주의할 것이 있다. `init_schema()` 는 **없는 테이블만
만들고 기존 테이블은 건드리지 않는다.** 개발 중에는 `docker compose down -v`
로 지우고 다시 만드는 게 빠르다.

### 컨테이너 없이 돌려야 할 때

발표 데모에서 맥북 메모리가 빠듯하면 `.env` 에서 이 한 줄만 켜면 된다.

```
JARVISEO_DATABASE_URL=sqlite:///data/jarviseo.db
```

SQLAlchemy 를 끼고 쓰기 때문에 코드는 한 줄도 안 바꿔도 된다.

---

## 역할과 책임 지표

각자 자신의 숫자를 갖는다. 결과보고서와 이력서가 이 표에서 그대로 나온다.

| 구분 | 소유 범위 | 책임 지표 |
|---|---|---|
| **최홍묵**<br>지시 대상 특정 | 데이터 설계·라벨 기준, YOLO 학습, 다중 단서 융합, ablation, LangGraph, 전체 통합, 지연 최적화 | 대상 선택 정확도, fingertip mAP@0.5, 응답 지연 p50/p95 |
| **권용현**<br>성분표 인식 · 대시보드 | 성분표 데이터 설계·라벨 기준, YOLO 학습, OCR 파이프라인, 인식률 개선, FastAPI·대시보드 | 성분표 검출 mAP, OCR 인식률, 알레르기 판정 recall |
| **문태현**<br>음성 · 검색 · 배포 | 웨이크워드 커스텀 학습, STT·TTS, 벡터DB·RAG, 도커·CI·배포 | 웨이크워드 FAR/FRR, RAG 응답 근거율 |

---

## 어디가 딥러닝이고 어디가 아닌가

발표에서 자주 꼬이는 지점이라 미리 정리해둔다.

| 부분 | 무엇을 쓰나 | 학습하나 |
|---|---|---|
| 손끝 검출 | YOLO 파인튜닝 | **예** |
| 성분표 영역 검출 | YOLO 파인튜닝 | **예** |
| 웨이크워드 "자비서" | 커스텀 오디오 모델 | **예** |
| 후보 물체 검출 | 사전학습 YOLO 그대로 | 아니요 |
| 성분표 글자 읽기 | 기성 OCR 엔진 | 아니요 |
| 알레르기 판정 | 동의어 사전 + 규칙 | 아니요 |
| 다중 단서 융합 | 기하 계산 + 가중치 공식 | 아니요 (※) |
| 최종 답변 생성 | 클라우드 VLM | 아니요 |

※ 융합 가중치를 로지스틱 회귀로 학습시키면 정형 데이터 기반 머신러닝이
여기 들어온다. W4 체크포인트에서 여유가 있으면 검토한다.

---

## 팀 규칙

### 크로스 플랫폼 (맥 1대 · 윈도우 2대)

- 파일 입출력에 **`encoding="utf-8"` 을 반드시 명시한다.** 윈도우 기본값은
  cp949 라서, 생략하면 맥에서만 되고 팀원 노트북에서 한글이 깨진다.
- 경로는 **`pathlib.Path`** 를 쓴다. 문자열 결합 금지.
- 카메라 열기의 OS 분기는 `capture/frame_source.py` 안에서만 한다.
- 모델 배포 포맷은 **ONNX** 로 통일한다. CoreML 은 맥 전용이라 팀원이 못 쓴다.
- 오디오는 **sounddevice** 로 통일한다. PyAudio 는 윈도우 설치가 자주 실패한다.

### 계약 파일을 바꿀 때

`types.py` 와 `graph/state.py` 는 남의 코드가 의존하는 파일이다.
필드를 추가·삭제·개명할 때는 팀 채널에 공지하고, 커밋 타입 뒤에 `!` 를
붙인다 (`feat(types)!: Frame 에 sharpness 필드 추가`).

전체 규칙은 [작업 컨벤션](.github/CONTRIBUTING.md)을 본다.

### 평가

- **baseline 은 학습 전에 잰다.** 학습을 시작한 뒤에는 사전학습 상태의
  성능을 되돌려 잴 수 없다.
- **test 셋은 최종 측정 전까지 열지 않는다.** 보면서 튜닝하면 나온 숫자가
  실제 성능이 아니게 된다.
- 모든 실험은 `docs/experiments.md` 에 기록한다.

---

## 알려진 함정

| 함정 | 대응 |
|---|---|
| TTS 되먹임 루프 | 재생 중 마이크 게이트, 종료 후 0.3초 여유 |
| 모션 블러 | 버퍼에서 Laplacian variance 가장 높은 프레임 선택 |
| 질문 시점 프레임 오차 | VAD 가 발화 **시작** 시각을 반환하게 하고 그 기준으로 선택 |
| 알레르겐 표기 불일치 | '우유' ↔ '탈지분유'·'카제인'·'유청'. **동의어 사전이 정확도를 좌우한다** |
| 맥 Docker + USB 카메라 | 캡처는 네이티브, 처리 서버만 컨테이너 |
| 허브 미지참 | 맥북에 USB-A 포트가 없다. 발표 당일 준비물 1순위 |

전체 목록과 설계 결정 근거는 인수인계서를 본다.

---

## 일정

| 주차 | 기간 | 목표 |
|---|---|---|
| W1 | 9/21~9/27 | 기술 검증 5종, 데이터 촬영 착수, 모듈 골격 |
| W2 | 9/28~10/4 | 10/1 설계 제출물, 개별 모델 1차 학습, 파이프라인 1차 연결 |
| W3 | 10/5~10/11 | 카메라 실물 통합, 모듈 통합, 지시 대상 특정 v1~v2 |
| W4 | 10/12~10/18 | v3~v5 단서 추가, 통합 테스트, 대시보드 |
| W5 | 10/19~10/25 | **ablation study 및 성능 측정** |
| W6 | 10/26~11/3 | 코드 프리즈(10/26), 제출물, 리허설, 발표(11/3) |
