# 시작하기 — 팀원용

레포에 파일이 많지만 **첫 주에 볼 건 이 문서 하나입니다.**
나머지는 필요해질 때 그때 보면 됩니다.

---

## 1. 설치 (한 번만, 10분)

```bash
git clone https://github.com/Mook-Industries/JARVISEO.git
cd JARVISEO
python -m venv .venv
```

가상환경을 켭니다. **터미널을 새로 열 때마다 이 줄을 다시 쳐야 합니다.**

```bash
.venv\Scripts\activate          # 윈도우
```

```bash
source .venv/bin/activate       # 맥
```

켜지면 프롬프트 앞에 `(.venv)` 가 붙습니다. 그 상태에서 설치합니다.

```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

잘 됐는지 확인:

```bash
python -c "import jarviseo, cv2; print('OK')"
```

> `pip install -e .` 는 **우리 코드를 파이썬이 찾을 수 있게 등록**하는 겁니다.
> 이게 없으면 `import jarviseo` 가 안 됩니다. 한 번만 하면 됩니다.

`.env` 를 열어 `JARVISEO_LLM_API_KEY` 를 채웁니다. 키는 팀 채널에서 공유받습니다.
**`.env` 는 절대 커밋하지 않습니다.**

> **conda 를 쓰는 사람은** 위의 venv 두 줄 대신 이걸 쓰세요. 나머지는 같습니다.
> ```bash
> conda create -n JARVISEO python=3.12 -y
> conda activate JARVISEO
> ```
> 파이썬 버전만 3.12 면 되고, venv 든 conda 든 상관없습니다.

---

## 2. 매일 하는 흐름

이 다섯 줄이 전부입니다. 외우지 말고 필요할 때 복사하세요.

```bash
# ① 최신 코드 받기 (작업 시작 전 항상)
git switch main && git pull

# ② 내 브랜치 만들기
git switch -c feat/12-wakeword

# ③ 작업하고 커밋
git add .
git commit -m "feat(voice): 웨이크워드 감지 기능 추가"

# ④ 올리기
git push -u origin feat/12-wakeword
```

그 다음 GitHub 에서 **PR 열기 → 리뷰 1명 승인 → Squash and merge → 브랜치 삭제.**

### 브랜치 이름

```
<타입>/<이슈번호>-<영어-소문자>
feat/12-wakeword
fix/15-windows-path
```

한글과 공백은 넣지 마세요. 맥과 윈도우에서 깨집니다.

### 커밋 메시지

```
<타입>(<폴더이름>): 한글로 무엇을 했는지
```

타입은 이 다섯 개면 충분합니다.

| 타입 | 언제 |
|---|---|
| `feat` | 새로 만들었을 때 |
| `fix` | 안 되던 걸 고쳤을 때 |
| `docs` | 문서만 고쳤을 때 |
| `refactor` | 동작은 그대로, 코드만 정리 |
| `chore` | 설정 파일 등 나머지 |

```
❌  수정
❌  fix: 버그 수정
✅  fix(capture): 윈도우에서 한글 파일명 이미지가 안 열리던 문제 수정
```

### 지켜야 할 것 세 가지

1. **`main` 에 직접 push 하지 않는다.** 항상 브랜치 → PR.
2. **브랜치는 1~3일 안에 끝낸다.** 오래 들고 있을수록 충돌이 커집니다.
3. **PR 은 작게.** 300줄 넘으면 리뷰어가 제대로 못 봅니다.

---

## 3. 내가 건드릴 폴더

```
src/jarviseo/
├── pointing/     지시 대상 특정        → 최홍묵
├── capture/      프레임 입력           → 최홍묵
├── graph/        파이프라인 연결        → 최홍묵
│
├── nutrition/    성분표 · OCR · 알레르기 → 권용현
│
├── memory/       DB · 벡터 검색        → 문태현
├── voice/        웨이크워드 · STT · TTS → 문태현
└── api/          서버 · 대시보드        → 권용현
```

**자기 폴더만 건드리면 충돌이 안 납니다.** 폴더를 나눈 이유가 이겁니다.

딱 하나 예외가 `src/jarviseo/types.py` 입니다. 셋이 주고받는 데이터 모양이
적혀 있어서, **여기를 고치면 남의 코드가 조용히 깨집니다.**
고쳐야 할 일이 생기면 먼저 팀 채널에 말하세요.

---

## 4. 지금은 안 봐도 되는 것

레포에 있지만 **첫 주에는 열 필요 없습니다.**

| 파일 | 언제 보나 |
|---|---|
| `Dockerfile`, `docker-compose.yml` 의 `dev`/`api` | 설치가 도저히 안 될 때만 |
| `.github/CONTRIBUTING.md` | 규칙이 헷갈릴 때 찾아보는 용도 |
| `src/jarviseo/memory/models.py` | C 만 |
| `docs/experiments.md` | 모델 학습 시작할 때 (W2) |
| 내 폴더가 아닌 모든 `.py` | 안 봐도 됩니다 |

지금 대부분의 `.py` 파일은 **함수 이름과 설명만 있고 속은 비어 있습니다.**
`NotImplementedError` 가 보이면 "아직 아무도 안 만든 자리"라는 뜻입니다.
그 자리를 채우는 게 앞으로 할 일입니다.

---

## 5. 데이터베이스 (필요해지면)

파이프라인을 실제로 돌릴 때만 필요합니다. 코드만 짜는 동안은 없어도 됩니다.

```bash
docker compose up -d db
python -c "from jarviseo.memory import MemoryStore; MemoryStore().init_schema()"
```

Docker Desktop 이 켜져 있어야 합니다. **설치할 건 Docker Desktop 하나뿐이고,
Postgres 는 따로 설치하지 않습니다.**

---

## 6. 카메라가 없어도 개발됩니다

안경 카메라는 팀에 한 대뿐이라, 카메라가 없어도 개발이 되게 만들어 뒀습니다.
**녹화된 영상이나 이미지 폴더**를 카메라 대신 쓰면 됩니다.
`.env` 에서 두 줄만 바꾸면 됩니다.

```
JARVISEO_FRAME_SOURCE=folder
JARVISEO_FRAME_PATH=data/datasets/sample
```

코드는 어느 쪽이든 똑같이 동작합니다. 영상·이미지는 공용 Drive 에서 받으세요.

---

## 7. 막혔을 때

**2시간 룰** — 혼자 2시간 넘게 막히면 팀 채널에 올립니다.
못해서가 아니라, 혼자 하루 태우는 게 팀 전체 손해라서 그렇습니다.

올릴 때 이 세 가지를 같이 적으면 훨씬 빨리 풀립니다.

```
1. 무엇을 하려고 했는지
2. 어떤 명령을 쳤는지
3. 에러 메시지 전문 (마지막 줄만 말고 전부)
```

---

## 첫 주에 할 일

`scripts/spike/` 폴더를 보세요. **"되는지 확인만 하는" 짧은 코드 5개**가
있습니다. 잘 만들 필요 없고 30줄 내외로 돌아가기만 하면 됩니다.

| 담당 | 파일 | 확인할 것 |
|---|---|---|
| 전원 | `spike_01_capture.py` | 프레임이 들어오는가 (**지금 바로 실행 가능**) |
| 문태현 | `spike_02_stt.py` | 마이크에서 한국어가 글자로 나오는가 |
| 최홍묵 | `spike_03_vlm.py` | 이미지 한 장 보내면 답이 오는가 |
| 최홍묵 | `spike_04_graph.py` | LangGraph 노드 2개가 도는가 |
| 최홍묵, 권용현 | `spike_05_yolo.py` | 사전학습 YOLO 가 물체를 잡는가 |

1번은 지금 바로 됩니다. 노트북 웹캠으로 돌아갑니다.

```bash
python scripts/spike/spike_01_capture.py
```

숫자가 주르륵 나오면 성공입니다. 고개를 흔들면서 다시 돌려보세요.
선명도 숫자가 떨어지는 게 보일 겁니다.

---

더 자세한 규칙은 [`.github/CONTRIBUTING.md`](../.github/CONTRIBUTING.md) 에
있습니다. **지금 읽을 필요는 없고**, PR 을 올리다 막히면 그때 찾아보세요.
