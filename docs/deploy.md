# 노트북 설치·실행

자비서 본체(`scripts/assistant.py`: 호출어 → 받아쓰기 → VLM → 음성 대답)는 **노트북에서 직접** 돈다.
카메라·마이크·스피커가 노트북에 꽂혀 있고, 맥·윈도우의 Docker 는 이 장치를 컨테이너로 넘기지 못하기 때문이다.
대화 기록은 노트북의 SQLite(`data/assistant.db`)에 남는다. 클라우드 대시보드로 보내는 일은 #143·#144 에서 한다.

이 문서는 두 부분이다.

- **처음 한 번 — 설치**: 새 노트북에서 30분쯤. 인터넷이 필요하다.
- **매번 — 켜기**: 바탕화면 아이콘 더블클릭. 이 노트북에서 30초쯤 뒤 "자비서"를 기다린다.

---

## 처음 한 번 — 설치

### 1. Git 과 Miniconda 설치

| | 윈도우 | 맥 |
|---|---|---|
| Git | <https://git-scm.com/download/win> 기본값으로 설치 | 터미널에서 `git --version` (없으면 설치 창이 뜬다) |
| Miniconda | <https://docs.anaconda.com/miniconda/> 의 Windows 설치 파일 | 같은 곳의 macOS 설치 파일 (애플 칩이면 arm64) |

윈도우는 설치 뒤 시작 메뉴의 **Anaconda Prompt** 를 연다. 아래 명령은 전부 거기서 친다. 맥은 터미널.

### 2. 레포 받기와 파이썬 환경

```bash
git clone https://github.com/Mook-Industries/JARVISEO.git
cd JARVISEO
conda create -n JARVISEO python=3.12 -y
conda activate JARVISEO
pip install -r requirements.txt
pip install -e .
```

`pip install` 은 torch·EasyOCR·YOLO 때문에 10분 넘게 걸릴 수 있다.

> 버전을 고정해서 깔려면 `requirements.txt` 대신 `requirements.lock` 을 쓴다(명령은 README 의 "파이썬 환경" 절).
> 개발 이미지와 CI 가 쓰는 버전이고 컨테이너 안에서 테스트가 통과했다. 다만 카메라·마이크로 직접 돌려 본
> 노트북은 `requirements.txt` 로 깐 환경이라, 잠금 파일로 깐 노트북은 처음 켤 때 몇 턴 더 확인한다.

### 3. `.env` 채우기

```bash
copy .env.example .env      # 맥: cp .env.example .env
```

`.env` 를 메모장으로 열어 **맨 위 "꼭 채울 값"만** 채운다.

| 값 | 무엇 |
|---|---|
| `OPENAI_API_KEY` | 팀 공용 키. 팀 노션에 있다 |
| `JARVISEO_MIC` · `JARVISEO_SPEAKER` | 번호나 장치 이름 일부(`USB`, `Buds`). 비우면 노트북 기본 장치. 이름 목록은 5번 점검이 보여 준다 |
| `JARVISEO_CAMERA_INDEX` | 카메라 번호. 내장 카메라가 보통 0, USB 카메라는 1 부터 |
| `JARVISEO_ENABLE_RECALL` | 기억 검색. `true` 그대로 |
| `JARVISEO_AUTO_UPDATE` | 켤 때 새 버전 받기. 확인한 버전에 고정하려면 `false` (아래 "버전 고정") |

장치 **번호**는 USB 를 꽂는 순서마다 바뀐다. **이름**을 적는 편이 안전하다.

### 4. 모델 받기

```bash
python scripts/download_models.py
```

| 받는 것 | 어디서 | 크기 · 걸린 시간(10/22) |
|---|---|---|
| 호출어 모델 `data/models/jarviseo.onnx` | 레포 GitHub 릴리스 `models-v1` | 205KB |
| 손 관절 모델 `data/models/hand_landmarker.task` | MediaPipe 공식 | 7.8MB · 8초 |
| openWakeWord 특징 모델 2개 | openWakeWord 공식 | 2.4MB · 7초 |
| EasyOCR 한국어·영어 `~/.EasyOCR/model` | EasyOCR | 99MB · 86초 |
| `yolov8n.pt` (레포 맨 위) | ultralytics | 6.5MB · 6초 |

이미 있는 파일은 건너뛴다. 몇 번을 돌려도 된다.
이 파일들은 처음 쓸 때 저절로 받기도 하지만, 그러면 인터넷이 느린 곳에서 첫 질문이 그만큼 늦어진다.

### 5. 점검

```bash
python scripts/check_setup.py
```

파이썬·패키지·`.env`·모델·마이크·스피커·카메라·OpenAI 연결·DB 를 한 번에 보고,
`[통과]` / `[주의]`(없어도 켜지지만 기능 하나가 빠짐) / `[실패]`(고쳐야 켜짐)로 알려 준다.
실패마다 무엇을 하면 되는지 `→` 로 한 줄씩 나온다. 아래 "문제 해결" 표도 본다.

### 6. 바탕화면 아이콘 만들기

- **윈도우**: 탐색기에서 `JARVISEO\scripts\start.bat` 를 오른쪽 클릭 → **추가 옵션 표시** → **보내기** → **바탕 화면에 바로 가기 만들기**. 바로 가기 이름은 "자비서 시작" 으로 바꾼다.
- **맥**: Finder 에서 `JARVISEO/scripts/start.command` 를 option+command 를 누른 채 바탕화면으로 끌어 별칭을 만든다.
  처음 열 때 "확인되지 않은 개발자" 창이 뜨면 오른쪽 클릭 → **열기**. 카메라·마이크 권한 창이 뜨면 **터미널**에 허용한다.

---

## 매번 — 켜기

바탕화면의 **자비서 시작** 을 더블클릭한다. 창 하나가 열리고 차례로 이렇게 한다.

```
1/5 새 버전 받기   main 의 새 버전을 받는다 (git pull)
2/5 패키지         받은 새 버전에서 requirements.txt 가 바뀌었을 때만 pip install
3/5 모델           없는 모델만 받는다
4/5 점검           check_setup.py 와 같은 점검. 실패가 있으면 여기서 멈추고 할 일을 보여 준다
5/5 켜기           assistant.py
```

`'자비서'를 부르고 이어서 말하세요` 가 나오면 준비가 끝난 것이다. 끝낼 때는 그 창에서 **Ctrl+C**.
끝나면 단계별 지연 중앙값을 찍고 "아무 키나 누르십시오" 에서 기다린다(그동안 결과를 읽을 수 있다).

10/22 이 노트북(RAM 7.7GB)에서 더블클릭부터 호출어를 기다리기까지 **29초** 였다(점검 끝까지 23초).
노트북을 막 켠 뒤 처음이면 디스크에서 패키지를 읽느라 더 걸린다.

새 버전 받기는 이럴 때 **건너뛰고 지금 버전으로 켠다**.

- 인터넷이 없거나 60초 안에 끝나지 않을 때
- `main` 이 아닌 브랜치일 때(개발하는 노트북), 고친 파일이 있을 때
- `.env` 에 `JARVISEO_AUTO_UPDATE=false` 일 때

명령으로 켜도 같다. 인자는 `assistant.py` 와 같다.

```bash
python scripts/start.py                  # .env 의 장치
python scripts/start.py USB Buds         # 이번만 다른 마이크·스피커
python scripts/start.py --no-update      # 이번만 새 버전 받기 건너뜀
```

### 버전 고정

켤 때마다 새 버전을 받으면, 누가 `main` 에 막 머지한 코드 때문에 잘 되던 노트북이 갑자기 안 켜질 수 있다.
꼭 켜져야 하는 날 전에는 버전을 고정한다.

1. 한 번 켜서 새 버전을 받는다.
2. 잘 도는지 확인한다.
3. `.env` 의 `JARVISEO_AUTO_UPDATE=false` 로 바꾼다. 이제 켤 때마다 확인한 그 버전이다. 다시 받으려면 `true`.

### 고친 코드가 노트북에 들어가는 길

```
고침 → PR → CI(윈도우·맥·리눅스에서 린트·테스트) → 리뷰 → main 에 머지
     → 다음에 아이콘으로 켤 때 1/5 에서 받는다
```

명령으로 개발하는 팀원은 지금처럼 `git pull` 하면 된다. 호출어 모델을 바꿨을 때만
`data/models/jarviseo.onnx` 를 지우고 `python scripts/download_models.py` 를 다시 돌린다.

### 나중에 Postgres 로 바꿀 때

지금은 SQLite(`data/assistant.db`)를 쓴다. 로컬 Postgres(도커)로 바꿀 때는 코드를 고치지 않는다.

```bash
docker compose up -d db
python scripts/start.py --db postgresql+psycopg://jarviseo:jarviseo@localhost:5432/jarviseo
```

바탕화면 바로 가기로 켜려면 바로 가기 **속성 → 대상** 끝에 ` --db postgresql+psycopg://...` 를 붙인다.
점검(4/5)이 같은 DB 에 붙어 보고, 도커가 꺼져 있으면 거기서 알려 준다.

---

## 문제 해결

| 증상 | 할 일 |
|---|---|
| 아이콘을 눌렀더니 "conda 환경 JARVISEO 를 찾지 못했습니다" | 2번을 다시 한다. 다른 곳에 깔았으면 시스템 환경변수 `JARVISEO_PYTHON` 에 그 `python.exe` 경로를 적는다 |
| `[실패] 패키지` / `jarviseo 패키지` | `conda activate JARVISEO` 뒤 `pip install -r requirements.txt` · `pip install -e .` |
| `[실패] OPENAI_API_KEY` · `OpenAI — 키가 틀림` | `.env` 의 키를 다시 붙여 넣는다. 앞뒤 공백·따옴표가 없어야 한다 |
| `[실패] OpenAI — 연결 안 됨` | 와이파이·핫스팟을 확인한다 |
| `[실패] 호출어 모델` · "릴리스에 jarviseo.onnx 가 없음" | 레포 Releases 에 `models-v1` 이 있는지 본다. 급하면 공용 Drive 의 `jarviseo/jarviseo-r6.onnx` 를 `data/models/jarviseo.onnx` 로 복사한다 |
| `[실패] 마이크` · `스피커` — "이름에 '…' 가 들어간 …가 없습니다" | 같은 줄에 나온 "고를 수 있는 것" 중 하나의 일부를 `.env` 에 적는다. USB 를 다시 꽂았으면 이름은 그대로 두면 된다 |
| `[실패] 카메라` | `JARVISEO_CAMERA_INDEX` 를 0·1·2 로 바꿔 본다. 맥은 시스템 설정 → 개인정보 보호 → 카메라에서 터미널 허용 |
| 마이크는 통과인데 "자비서"에 반응이 없음 | 윈도우 설정 → 개인 정보 → 마이크에서 데스크톱 앱 허용. 맥은 마이크 권한. `JARVISEO_MIC` 가 다른 마이크를 가리키는지 시작할 때 찍힌 `마이크 …` 줄로 확인 |
| `[실패] DB` (Postgres) | 도커를 켜고 `docker compose up -d db`, 아니면 `--db` 를 빼고 SQLite 로 |
| `[주의] zxing-cpp` | `pip install zxing-cpp` (없으면 바코드를 잘 못 읽는다) |
| `[주의] 푸드QR` | 바코드 조회 없이 성분표 OCR 로 간다. 키는 portal.foodqr.kr |
| 1/5 가 "건너뜀 — Not possible to fast-forward" | 이 노트북에서 커밋을 했다. `git status` 로 보고 팀에 물어본다 |

## 호출어 모델을 바꿀 때 (담당자)

1. 새 모델을 GitHub 레포 **Releases → Draft a new release** 로 올린다. 태그는 `models-v2` 처럼 새로, 파일 이름은 `jarviseo.onnx`.
2. `scripts/download_models.py` 의 `RELEASE_URL` 태그와 `WAKEWORD_SHA256` 을 바꾸고, `data/README.md` 의 호출어 모델 절도 고친다.
3. PR 로 머지한다. 각 노트북은 `data/models/jarviseo.onnx` 를 지우고 `download_models.py` 를 다시 돌린다.

옛 릴리스는 지우지 않는다. 새 모델이 나쁘면 태그만 되돌리면 된다.
