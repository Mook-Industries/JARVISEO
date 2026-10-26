# 노트북 설치·실행

자비서 본체(`scripts/assistant.py`: 호출어 → 받아쓰기 → VLM → 음성 대답)는 **노트북에서 직접** 돈다.
카메라·마이크·스피커가 노트북에 꽂혀 있고, 맥·윈도우의 Docker 는 이 장치를 컨테이너로 넘기지 못하기 때문이다.
대화 기록은 노트북의 SQLite(`data/assistant.db`)에 남는다. 클라우드 대시보드로 보내는 일은 #143·#144 에서 한다.

이 문서는 세 부분이다.

- **처음 한 번 — 설치**: 새 노트북에서 30분쯤. 인터넷이 필요하다.
- **매번 — 켜기**: 바탕화면 아이콘 더블클릭. 이 노트북에서 30초쯤 뒤 "자비서"를 기다린다.
- **클라우드 서버(AWS)**: 휴대폰으로 여는 대시보드. main 에 머지되면 저절로 새 버전이 뜬다.

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

---

## 클라우드 서버 (AWS)

휴대폰에서 `https://<도메인>` 으로 여는 **팀 웹**(`api/main.py`: 대시보드 `/` · 물어보기 `/assistant` ·
성분 확인 `/ingredient`)이다(#179). 서버가 OpenAI · Google Vision · 푸드QR 키로 질문과 성분 확인을 처리하므로
**접속 비밀번호**를 건다. 호출어로 부르는 본체(`assistant.py`)는 여전히 노트북에서 돈다.
대시보드의 "자비서" 호출은 서버가 아니라 **브라우저**가 듣는다(#218, `src/jarviseo/api/README.md` "브라우저 호출어").
서버는 호출어 모델 3개(약 2.6MB)를 내려주기만 한다.

```
휴대폰 ──https──▶ <이름>.duckdns.org ──▶ EC2 (서울 ap-northeast-2, t3.small 2GB, 고정 IP)
                                          ├─ caddy   80·443, https 인증서 자동, 접속 비밀번호(/health 만 열림)
                                          ├─ api     팀 웹(Dockerfile.web) — ECR 에서 받아 온다
                                          └─ db      pgvector/pgvector:pg16, 바깥에 안 열림
GitHub main 머지 ─▶ Actions(deploy.yml) ─OIDC─▶ AWS 역할 ─▶ 이미지 빌드 → ECR ─SSM─▶ 서버에서 deploy/update.sh
                                                         키 · 비밀번호: Parameter Store /jarviseo/
```

| 파일 | 무엇 |
|---|---|
| `deploy/compose.yml` | 서버에서 띄우는 세 컨테이너. DB 포트는 열지 않는다. api 는 메모리 상한 1400MB · 사진은 tmpfs |
| `deploy/Caddyfile` | 도메인으로 온 요청에 접속 비밀번호를 확인하고 api 로. 인증서는 Caddy 가 알아서 받는다 |
| `deploy/update.sh` | 서버에서 돈다: 키 받기 → ECR 에서 이미지 받기 → 다시 띄우기 → 스키마 맞추기 → `/health` 확인 |
| `deploy/bootstrap.sh` | 서버에 AWS CLI 가 없으면 `update.sh` 가 처음 한 번 깐다 |
| `deploy/schema.sql` | `scripts/schema_sql.py` 가 모델에서 뽑은 스키마. 모델을 바꾸면 다시 뽑는다 |
| `deploy/env.example` | 서버의 `/opt/jarviseo/.env` 모양(도메인 · DB 비밀번호). 실제 값은 서버에만 |
| `Dockerfile.web` | 팀 웹 이미지(3.85GB, 받는 크기 917MB). YOLO · 손 관절 모델, 브라우저에 내려줄 호출어 모델 3개를 미리 넣는다 |
| `scripts/put_cloud_secrets.py` | 키 · 접속 비밀번호를 Parameter Store 에 넣는다 |
| `.github/workflows/deploy.yml` | main 머지 때 이미지를 빌드해 ECR 에 올리고 서버에 새 버전을 띄운다 |

### AWS 에 만든 것

전부 서울 리전이고 `Project=jarviseo` 태그가 붙어 있다. 같은 계정의 다른 자원은 건드리지 않는다.

| 자원 | 이름 | 내용 |
|---|---|---|
| EC2 | `jarviseo` | **t3.small(2GB)**, Ubuntu 24.04, 디스크 20GB gp3(암호화), 첫 부팅에 Docker · compose · 스왑 1GB |
| Elastic IP | `jarviseo` | 서버에 붙은 고정 IP. DuckDNS 도메인이 이 IP 를 가리킨다 |
| 보안 그룹 | `jarviseo-web` | 80 · 443 만 연다. **SSH(22)는 열지 않는다** — 서버 명령은 SSM 으로 보낸다 |
| IAM 역할 | `jarviseo-ec2-ssm` | 서버 권한: SSM 명령 받기 · Parameter Store 읽기(`AmazonSSMManagedInstanceCore`), ECR 에서 이미지 받기(`AmazonEC2ContainerRegistryReadOnly`) |
| ECR 저장소 | `jarviseo-web` | 팀 웹 이미지. 커밋마다 태그 + `latest`, 최근 5개만 남긴다 |
| Parameter Store | `/jarviseo/*` | SecureString: `OPENAI_API_KEY` · `GOOGLE_VISION_API_KEY` · `JARVISEO_FOOD_QR_API_KEY` · `BASIC_AUTH_PASSWORD` |
| OIDC 제공자 + IAM 역할 | `jarviseo-github-deploy` | GitHub Actions 가 키 없이 빌리는 역할. `Mook-Industries/JARVISEO` 의 main 에서만, ECR `jarviseo-web` 에 올리기와 그 서버 한 대에 SSM 명령만 |
| 예산 | `jarviseo-monthly` | `Project=jarviseo` 비용 월 $10, 실제 80% · 예상 100% 에서 메일(크레딧 포함 금액) |

GitHub 저장소 변수(Settings → Secrets and variables → Actions → **Variables**)에 `AWS_DEPLOY_ROLE_ARN` · `EC2_INSTANCE_ID` 를 넣어 두었다.
비밀값이 아니라 위치 정보다. GitHub 에는 AWS 키도, API 키도, 서버 비밀번호도 없다.

### 키 · 접속 비밀번호

서버가 쓰는 API 키와 접속 비밀번호는 AWS **Parameter Store**(`/jarviseo/`)에 암호화해 둔다.
`update.sh` 가 배포할 때마다 서버의 IAM 역할로 받아 `/opt/jarviseo/secrets.env`(root 만 읽음)에 쓴다.
Git · GitHub · SSM 명령 기록 어디에도 값이 남지 않는다.

```bash
python scripts/put_cloud_secrets.py                 # 이 노트북 .env 의 키를 넣고, 접속 비밀번호가 없으면 만든다
python scripts/put_cloud_secrets.py --new-password  # 접속 비밀번호 바꾸기 — 새 값을 이 화면에만 보여 준다
```

넣거나 바꾼 값은 **다음 배포부터** 서버에 반영된다(Actions → Deploy → Run workflow).
접속 비밀번호를 다시 보려면 AWS 콘솔 → Systems Manager → Parameter Store → `/jarviseo/BASIC_AUTH_PASSWORD` → Show decrypted value.
아이디는 `jarviseo` 다.

접속 비밀번호는 **임시**다(Caddy 의 브라우저 기본 로그인 창). 웹 담당과 정한 로그인이 팀 웹에 들어오면 바꾼다.
`/health` 만 비밀번호 없이 열린다(배포 확인용).

### 서버 크기

10/24 로컬 Docker 에서 팀 웹을 재 보고(`scripts/measure_web_memory.py`) t3.small(2GB)로 정했다.

| 상황 | 메모리 최고 | 응답 |
|---|---|---|
| 켠 직후 대기 | 128MiB | — |
| 질문(`/api/ask`) | 546MiB | 1.8~1.9초(첫 질문 7.9초 — YOLO · 손 관절을 이때 불러온다) |
| 성분 확인(Google Vision) | 580MiB | 4.3~4.6초 |
| 성분 확인(EasyOCR 로 떨어지면) | 3,108MiB | 10분 넘어도 안 끝남 |

10/24 서버(t3.small)에 올린 뒤 같은 사진으로 로컬(이 노트북, `python -m jarviseo.api`)과 5번씩 비교했다
(`scripts/compare_web_latency.py`). 서버가 노트북 CPU 보다 빨라 오히려 짧았다.

| | 질문 중앙값 | 성분 확인(Vision) 중앙값 |
|---|---|---|
| 로컬(노트북) | 4.0초 (첫 질문 23.5초) | 6.8초 |
| 클라우드(t3.small, https · 접속 비밀번호 포함) | **1.9초** (첫 질문 12.1초) | **3.1초** |

그때 서버 메모리: 팀 웹 663MiB(상한 1.37GiB) · db 31MiB · caddy 13MiB, 서버 전체 1.9GB 중 1.1GB 사용,
스왑 9MB. 디스크 20GB 중 8.4GB.

EasyOCR 은 서버를 키워도 쓸 수 없을 만큼 느려서, 서버에서는 **꺼 둔다**(`JARVISEO_OCR_LOCAL_FALLBACK=false`,
`deploy/compose.yml`). Vision 이 실패하면 "지금은 성분표를 읽을 수 없어요" 라고 답한다. 그래도 메모리가 튀면
api 컨테이너만 1400MB 에서 죽고 다시 뜬다(db · caddy 는 그대로).

크기를 바꿀 때는 인스턴스를 **중지 → 인스턴스 유형 변경 → 시작**한다(1~2분 멈춘다). 고정 IP · 도메인 · DB · 인증서는 그대로다.

```bash
aws ec2 stop-instances --instance-ids <EC2_INSTANCE_ID> --profile jarviseo --region ap-northeast-2
aws ec2 wait instance-stopped --instance-ids <EC2_INSTANCE_ID> --profile jarviseo --region ap-northeast-2
aws ec2 modify-instance-attribute --instance-id <EC2_INSTANCE_ID> --instance-type Value=t3.small --profile jarviseo --region ap-northeast-2
aws ec2 start-instances --instance-ids <EC2_INSTANCE_ID> --profile jarviseo --region ap-northeast-2
```

PowerShell 에서 `--instance-type '{"Value":"t3.small"}'` 처럼 JSON 으로 쓰면 따옴표가 빠져 실패한다. 위처럼 `Value=` 로 쓴다.

### 받은 사진

질문 · 성분 확인 때 받은 사진은 팀 웹이 `data/captures/web/` 에 남긴다. 서버에서는 이 폴더를 **tmpfs(메모리, 200MB)**로
두어 디스크에 쌓이지 않고 컨테이너를 다시 띄우면(배포 · 재시작마다) 사라진다. 얼굴 · 집 안이 찍힐 수 있어서다.
10/24 확인: 질문 · 성분 확인 10번 뒤 tmpfs 에 사진 15장(1.3MB), 디스크에는 없다. 대화 기록(턴)은 서버 Postgres 에 남는다.

### 노트북 대화 기록 보내기 (#144)

노트북 `assistant.py` 는 턴을 노트북 DB(`data/assistant.db`)에 **먼저** 쓰고, 그 뒤 다른 스레드가 서버
`POST /api/turns` 로 복사본을 보낸다(`src/jarviseo/memory/uploader.py`). 대답은 보내기를 기다리지 않는다.
대시보드 `/` 의 "대화 내역"이 3초마다 `GET /api/turns` 로 새 턴을 가져와 위에 쌓는다(노트북 턴은 "노트북", 웹에서 물은 턴은 "웹").

```
assistant.py ─턴 기록─▶ data/assistant.db ─▶ TurnUploader(스레드) ─https · 토큰─▶ POST /api/turns ─▶ 서버 Postgres
                                                                                          대시보드 ◀─ GET /api/turns(3초마다)
```

| 보낸다 | 보내지 않는다 |
|---|---|
| 질문 · 대답 · 받아쓴 원문 · 물은 시각 · 말 시작/끝 시각 | **사진**(`data/captures/`) · 사진 경로 — 얼굴 · 집 안이 찍힐 수 있다 |
| 단계별 지연(stt · route · vision · llm · tts · total) | 녹음 경로(`audio_path`) |
| 고른 대상 · 확신도 · 점수차 · 되물음 · 끼어들기 · "잠깐만요" 여부 | 가리킴 후보 목록 · 성분 판정 행 · 대화 기억(임베딩) |
| 성분표에서 읽은 글자(`ocr_text`) | 의도(DB 에 열이 없다) |

- **켜기**: 노트북 `.env` 에 `JARVISEO_CLOUD_URL=https://<도메인>` 과 `JARVISEO_CLOUD_TOKEN` 을 둔다. 둘 중 하나라도 비면 보내지 않는다.
  토큰은 `python scripts/put_cloud_secrets.py` 가 만들어 `.env` 에 적고 Parameter Store `/jarviseo/JARVISEO_CLOUD_TOKEN` 에도 넣는다
  (화면에 찍지 않는다). 서버에는 다음 배포 때 들어간다. 켜지면 `assistant.py` 첫 화면에 "클라우드로 보내기 켜짐"이 찍힌다.
- **토큰**: `POST /api/turns` 는 접속 비밀번호(Caddy)를 거치지 않고, 팀 웹이 `Authorization: Bearer <토큰>` 을 확인한다.
  틀리면 401, 서버에 토큰이 없으면 503 으로 아무것도 받지 않는다. 읽기(`GET /api/turns`)는 접속 비밀번호를 거친다.
- **못 보낸 턴**: 마지막으로 보낸 턴 번호를 `data/cloud_upload.json` 에 남긴다. 인터넷이 끊기면 2초부터 60초까지 늘려 가며
  다시 보내고, 다음 턴 · 다시 연결 · 다음에 켤 때 밀린 턴을 순서대로 보낸다. 서버는 같은 턴(같은 노트북 세션 · 시각 · 질문)을 두 번 남기지 않는다.
- **처음 켤 때**: 그때까지 노트북 DB 에 있던 턴(10/14~10/22 측정 턴 98개)은 보내지 않고, 켠 뒤의 턴부터 보낸다.
  예전 턴은 옛 코드로 잰 것이고 32개가 빈 대답 · "모르겠어요"라 대시보드 지연 · 정확도가 흐려진다. 노트북 DB 에는 그대로 남는다.
  예전 턴도 보내려면 `data/cloud_upload.json` 의 `last_turn_id` 를 그 앞 번호(전부면 0)로 고친다. 파일을 지우면 다시 "지금부터"가 된다.
- `assistant.py` 를 끝내면 "클라우드 보내기: 다 보냄, 턴 끝 → 서버가 받기까지 중앙 …ms" 를 찍는다.
  10/24 로컬(같은 PC 의 팀 웹)로 10턴: 턴 끝 → 서버 목록에 보이기까지 중앙 37ms(`notify` 자체는 0.02ms). 대시보드에는 여기에 3초 주기가 더해진다.

### 배포 역할이 믿는 GitHub 토큰 모양

Actions 가 AWS 에 내미는 OIDC 토큰의 `sub` 에 "어느 저장소의 어느 브랜치에서 돌았는지"가 들어 있고,
역할 `jarviseo-github-deploy` 는 그 값이 정확히 맞을 때만 빌려준다. 이 저장소는 GitHub 의
**immutable subject**(바꿀 수 없는 subject)를 쓰고 있어서 `sub` 에 이름 대신 숫자 ID 가 붙는다.

| | `sub` |
|---|---|
| 이 저장소의 실제 값 | `repo:Mook-Industries@330743458/JARVISEO@1375536158:ref:refs/heads/main` |
| 흔히 보는 예시(이름만) | `repo:Mook-Industries/JARVISEO:ref:refs/heads/main` — **이걸로는 거절된다** |

숫자는 조직 ID(`330743458`)와 저장소 ID(`1375536158`)다. 저장소 이름이 바뀌거나 누가 같은 이름으로
저장소를 다시 만들어도 바뀌지 않아서 이름만 쓰는 것보다 안전하다.
10/23 첫 자동 배포가 이 차이로 `Not authorized to perform sts:AssumeRoleWithWebIdentity` 에 걸렸다.

역할을 다시 만들 때는 저장소의 지금 형식을 먼저 확인하고 그 `sub_claim_prefix` 뒤에 `:ref:refs/heads/main` 을 붙인다.
(저장소 관리자 권한이 필요하다. GitHub CLI 가 있으면 `gh api repos/Mook-Industries/JARVISEO/actions/oidc/customization/sub`)

```bash
curl -H "Authorization: Bearer <GitHub 토큰>" https://api.github.com/repos/Mook-Industries/JARVISEO/actions/oidc/customization/sub
# {"use_default": true, "use_immutable_subject": true, "sub_claim_prefix": "repo:Mook-Industries@330743458/JARVISEO@1375536158"}
```

역할의 신뢰 정책(trust policy) 조건은 이렇게 둔다.

```json
"Condition": {"StringEquals": {
  "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
  "token.actions.githubusercontent.com:sub": "repo:Mook-Industries@330743458/JARVISEO@1375536158:ref:refs/heads/main"
}}
```

```bash
aws iam update-assume-role-policy --role-name jarviseo-github-deploy --policy-document file://gh-trust.json --profile jarviseo --region ap-northeast-2
```

### 비용 (서울, AWS 요금 API)

| 항목 | 단가 | 한 달 |
|---|---|---|
| EC2 t3.small | $0.026/시간 | $18.98 |
| 공인 IPv4(고정 IP) | $0.005/시간 | $3.65 |
| 디스크 gp3 20GB | $0.0912/GB-월 | $1.82 |
| ECR 이미지 보관(약 1GB × 최근 5개) | $0.10/GB-월 | 약 $0.5 |
| **합계(서버)** | | **약 $25** (데이터 전송은 이 규모면 무료 범위) |

여기에 질문 · 성분 확인 수만큼 **OpenAI**(VLM · 임베딩) 사용료와 **Google Vision**(월 1,000건까지 무료, 넘으면
1,000건당 약 $1.5)이 따로 나간다. 10/23 기준 이 계정 사용액은 크레딧으로 상쇄되고 있다(결제 콘솔 → Credits).

예산은 태그로 걸러서 본다. 태그는 **결제 콘솔에서 비용 분류 태그로 켜야** 예산에 잡히는데, 자원에 태그를 단 뒤
청구 데이터에 나타나기까지 최대 24시간이 걸린다. 그 전에는 켜기가 `Tag keys not found` 로 실패한다.

```bash
aws ce update-cost-allocation-tags-status --cost-allocation-tags-status TagKey=Project,Status=Active --region us-east-1 --profile jarviseo
```

### AWS CLI 로그인

액세스 키를 만들지 않고 콘솔 로그인으로 임시 자격 증명을 받는다(AWS CLI 2.32 이상의 `aws login`).
브라우저가 열리면 로그인하고, 터미널에 `Updated profile jarviseo ...` 가 나올 때까지 그 터미널을 닫지 않는다.

```bash
aws login --profile jarviseo --region ap-northeast-2
aws sts get-caller-identity --profile jarviseo
```

자격 증명은 시간이 지나면 만료된다. 명령이 `ExpiredToken` 으로 실패하면 다시 로그인한다.

### 처음 한 번 — 서버 `.env` 와 첫 배포

DuckDNS 에서 도메인을 만들고 current ip 에 고정 IP 를 넣은 뒤(update ip), 서버에 `.env` 를 만든다.
DB 비밀번호는 서버에서 만들어 서버에만 둔다(root 만 읽음). 명령은 JSON 파일로 넘긴다. 셸 따옴표가 덜 꼬인다.
`env-cmd.json` (도메인만 바꾼다):

```json
{"commands": [
  "set -e",
  "mkdir -p /opt/jarviseo",
  "if [ -f /opt/jarviseo/.env ]; then echo 'already exists'; else umask 077; printf 'DOMAIN=%s\nDB_PASSWORD=%s\n' jarviseo.duckdns.org $(openssl rand -hex 24) > /opt/jarviseo/.env; echo created; fi",
  "grep '^DOMAIN=' /opt/jarviseo/.env"
]}
```

```bash
aws ssm send-command --profile jarviseo --region ap-northeast-2 --instance-ids <EC2_INSTANCE_ID> --document-name AWS-RunShellScript --parameters file://env-cmd.json --query Command.CommandId --output text
```

이미 `.env` 가 있으면 건드리지 않는다. DB 를 만든 뒤 비밀번호를 바꾸면 기존 DB 와 맞지 않으니 바꾸지 않는다.

키 · 접속 비밀번호를 Parameter Store 에 넣는다(위 "키 · 접속 비밀번호").

그다음 GitHub **Actions → Deploy → Run workflow**(main)로 첫 배포를 한다. 이후로는 main 에 머지될 때마다
(`src/jarviseo/**` · `deploy/**` · `Dockerfile.web` · `requirements.lock` · 동의어 사전이 바뀌었으면) 저절로 돈다.
이미지를 빌드해 올리느라 처음에는 10분쯤, 캐시가 있으면 몇 분 걸린다.

### 상태 보기 · 로그

SSH 대신 SSM 으로 명령을 보낸다. 결과는 `get-command-invocation` 으로 읽는다.

```bash
aws ssm send-command --profile jarviseo --region ap-northeast-2 --instance-ids <EC2_INSTANCE_ID> --document-name AWS-RunShellScript --parameters 'commands=["cd /opt/jarviseo/repo/deploy && docker compose --env-file /opt/jarviseo/.env --env-file /opt/jarviseo/secrets.env ps && docker compose --env-file /opt/jarviseo/.env --env-file /opt/jarviseo/secrets.env logs --tail 30 api caddy"]' --query Command.CommandId --output text
aws ssm get-command-invocation --profile jarviseo --region ap-northeast-2 --instance-id <EC2_INSTANCE_ID> --command-id <위에서 나온 ID> --query StandardOutputContent --output text
```

### 끄기 · 지우기

- **잠깐 끄기**: EC2 콘솔에서 인스턴스 **중지**. 서버 요금($18.98)은 멈추지만, 고정 IP 는 붙은 채 쉬어도 시간당 $0.005, 디스크는 그대로 나간다(월 약 $5.5).
  다시 **시작**하면 IP 는 그대로고 컨테이너는 `restart: unless-stopped` 라 저절로 뜬다.
- **완전히 지우기**(발표가 끝나고 더 안 쓸 때): 인스턴스 종료(디스크도 같이 지워짐 → **DB 데이터도 사라진다**) →
  Elastic IP 해제 → 보안 그룹 `jarviseo-web` · 역할 `jarviseo-ec2-ssm`(인스턴스 프로필 포함) · `jarviseo-github-deploy` ·
  OIDC 제공자(다른 프로젝트가 안 쓰면) · ECR 저장소 `jarviseo-web` · Parameter Store `/jarviseo/*` ·
  예산 `jarviseo-monthly` 삭제 → GitHub 저장소 변수 두 개 삭제.
  남길 기록이 있으면 먼저 `docker compose exec db pg_dump` 로 받아 둔다.

### 문제 해결 (클라우드)

| 증상 | 할 일 |
|---|---|
| `https://<도메인>` 이 안 열림, Caddy 로그에 인증서 오류 | DuckDNS 의 current ip 가 고정 IP 와 같은지, 보안 그룹에 80 · 443 이 열려 있는지 본다. 인증서는 80 번으로 확인받는다 |
| `/health` 는 200 인데 화면이 안 뜨거나 500 | `logs api` 를 본다. DB 연결 오류면 db 가 healthy 인지, `.env` 의 `DB_PASSWORD` 를 처음 만든 뒤 바꾸지 않았는지 확인한다(바꾸면 기존 DB 와 안 맞는다) |
| 로그인 창이 계속 뜸 | 아이디 `jarviseo`, 비밀번호는 Parameter Store `/jarviseo/BASIC_AUTH_PASSWORD`. 바꿨으면 Deploy 를 한 번 돌려야 반영된다 |
| Deploy 가 `Parameter Store 에 /jarviseo/BASIC_AUTH_PASSWORD 가 없다` 로 멈춤 | `python scripts/put_cloud_secrets.py` 를 돌린다 |
| 질문이 "키가 없다"고 답함 | Parameter Store 에 `OPENAI_API_KEY` 가 없다. 넣고 Deploy 를 다시 돌린다 |
| 성분 확인이 "지금은 성분표를 읽을 수 없어요" | Google Vision 이 실패했다(무료 한도 초과 · 키 오류 · 네트워크). `logs api` 에 Vision 실패 이유가 찍힌다 |
| Actions 의 Deploy 가 빨간불 | 그 실행의 로그에 서버에서 돈 `update.sh` 출력이 그대로 나온다 |
| Deploy 가 `configure-aws-credentials` 에서 `Not authorized to perform sts:AssumeRoleWithWebIdentity` | 역할이 믿는 `sub` 가 토큰과 다르다. 위 "배포 역할이 믿는 GitHub 토큰 모양"대로 저장소의 `sub_claim_prefix` 를 확인해 신뢰 정책을 맞춘다. main 이 아닌 브랜치에서 돌렸거나 저장소 변수 `AWS_DEPLOY_ROLE_ARN` 이 틀려도 같은 오류다 |
| `aws` 명령이 `NoRegion` | 프로필 `jarviseo` 에 리전이 없다. IAM 명령에도 `--region ap-northeast-2` 를 붙인다 |
| 윈도우 Git Bash 에서 `/jarviseo/` 경로가 이상하게 바뀜 | Git Bash 가 `/` 로 시작하는 인자를 윈도우 경로로 바꾼다. 앞에 `MSYS_NO_PATHCONV=1` 을 붙이거나 PowerShell 에서 돌린다 |
| 이 노트북에서 ECR 로 `docker login` 이 `error storing credentials` | 윈도우 자격 증명 관리자가 긴 ECR 토큰을 못 저장한다. 이미지는 Deploy 워크플로가 올리므로 보통은 손으로 올릴 일이 없다 |
| Deploy 가 아예 안 돎 | 바뀐 파일이 위 경로에 없으면 안 돈다. Actions 탭에서 Run workflow 로 돌린다 |
| SSM 명령이 `Undeliverable` | 서버가 꺼져 있거나 SSM 에 아직 안 붙었다. 켠 뒤 1~2분 기다린다 |
