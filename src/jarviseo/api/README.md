# api — FastAPI · 대시보드 · 담당 권용현

웹 관제실(JARVISEO 대시보드)의 백엔드와 화면.

| 파일 | 하는 일 |
|---|---|
| `main.py` | FastAPI 진입점 (`/`, `/assistant`, `/ingredient`, `/api/*`) |
| `__main__.py` | `python -m jarviseo.api` — 서버를 켜고 브라우저를 연다 |
| `assistant_web.py` | 웹 질문 하나를 팀 그래프(workflow)로 돌린다 |
| `marks.py` | 성분 판정에서 찾은 자리를 사진에 그린다 |
| `pages/` | 실제 API 를 붙인 화면 (대시보드 · 물어보기 · 성분 확인) |
| `static/` | 디자인 목업 원본. 직접 고치지 않는다 |

## 각자 PC 에서 대시보드 쓰기

서버는 **각자 자기 PC 에서** 켠다. 성분표 글자는 Google Vision 으로 읽는다.

**1. 설치** — 레포 README 의 [시작하기 1번](../../../README.md#1-파이썬-환경)
(venv → `pip install -r requirements.txt` → `pip install -e .`). 한 번만 하면 된다.

**2. `.env`** — 레포 맨 위 폴더의 `.env` 에 이것만 채운다.

| 항목 | 값 | 없으면 |
|---|---|---|
| `OPENAI_API_KEY` | **각자 본인 키** | 대답 · 기억 검색이 가짜로 돈다 |
| `GOOGLE_VISION_API_KEY` | Google Cloud API 키 (Cloud Vision API 사용 설정) | 아래 Colab · 이 PC 로 읽는다 |
| `JARVISEO_FOOD_QR_API_KEY` | 푸드QR 키 | 바코드로 성분을 못 가져오고 OCR 로만 읽는다 |
| `JARVISEO_OCR_REMOTE_URL` · `_TOKEN` | (선택) 권용현이 켜 둔 Colab OCR 주소 · 토큰 | 필요 없다. 구글이 실패할 때만 쓴다 |

키 · 토큰은 레포나 채팅방에 올리지 않는다.

**2-1. 손 모델(선택)** — 손끝 방향으로 가리킨 물건을 고르려면(가리킴 v2)
`data/models/hand_landmarker.task` 가 있어야 한다. git 에 없으니 각자 받는다
(받는 명령은 [pointing README](../pointing/README.md)). 없으면 서버 로그에 경고만 남고
화면 가운데에 가까운 물건을 고른다(v1). 응답의 `pointing` 이 `v2` 면 손을 쓴 것이다.

**3. 실행**

```bash
python -m jarviseo.api          # http://localhost:8000 이 열린다. 끄려면 Ctrl+C
```

카메라 · 마이크는 브라우저가 켠다. 처음에 뜨는 "카메라 · 마이크 허용"을 누른다.
`localhost` 로 열어야 켜진다(브라우저는 localhost · https 에서만 카메라를 허락한다).

### 팀 음성 모드

대시보드를 열면 서버가 이 PC 의 마이크 · 스피커를 열고 안경(`scripts/assistant.py`)과 같은 팀 부품으로 돈다
(`api/voice_loop.py`): 호출어 모델 → 시작음 → OpenAI 받아쓰기 → 그래프(가리킴 · 되묻기 · 성분 · 기억)
→ "잠깐만요" → OpenAI 목소리 → 끼어들기. 대답은 **PC 스피커**로 나온다. 카메라는 대시보드 탭이 잡고
장면을 서버로 보낸다. 입력창 질문도 같은 부품으로 돈다(받아쓰기만 건너뛴다).

- 필요한 것: `OPENAI_API_KEY`, 호출어 모델 `data/models/jarviseo.onnx`(받는 법은 `data/README.md`),
  손 모델(선택, 위 2-1). 마이크 · 스피커는 `.env` 의 `JARVISEO_MIC` · `JARVISEO_SPEAKER`(비면 기본 장치).
- 키나 마이크가 없으면 예전처럼 브라우저 받아쓰기 · 읽기로 돈다(콘솔에 `[voice] 팀 음성 모드 꺼짐`).

### 브라우저 호출어 (#218)

서버에 마이크가 없을 때(클라우드 `jarviseo.duckdns.org`) 대시보드 탭이 직접 "자비서"를 듣는다(`pages/wakeword.js`).
노트북과 같은 모델 3개를 onnxruntime-web 1.20.1(jsDelivr)로 돌린다. 평소 소리는 서버로 가지 않는다.

```
노트북 마이크(유선 카메라) → 브라우저: melspectrogram → embedding → jarviseo.onnx (80ms 마다 점수)
  "자비서" 감지 → 시작음 → 브라우저 받아쓰기 → POST /api/ask (사진 + 질문) → 서버 그래프 → 브라우저가 읽음
```

- 서버는 모델을 `GET /models/wakeword/<이름>` 으로 내려주고, `GET /api/wakeword` 가 켤 수 있는지와 임계값
  (config 의 `WAKEWORD_THRESHOLD` · `WAKEWORD_PLAYBACK_THRESHOLD`, 지금 둘 다 0.97)을 알려 준다.
  웹 이미지(`Dockerfile.web`)가 빌드 때 세 모델을 `data/models/` 에 넣는다. 노트북은 `scripts/download_models.py` 가 받은 것을 쓴다.
- 2단계 확인(노트북 #149 와 같은 `voice/wakecheck.py`): 깨어나면 감지까지 직전 1.6초(`JARVISEO_WAKE_CHECK_SEC`)를
  `POST /api/wake-check`(16kHz 모노 int16 PCM, 5초까지)로 보낸다. 서버가 OpenAI 로 받아써 "자비서"가 없으면 거절 →
  브라우저는 듣기를 멈추고 아무 말 없이 다시 기다린다. 질문을 듣는 동안 같이 돌아 진짜 호출은 늦어지지 않는다.
  키가 없거나 받아쓰기가 실패 · 3초 넘게 걸리면 통과. 키는 브라우저로 가지 않는다. 콘솔에 `[wake] 2단계 확인 통과 · 거절`.
- 대답을 읽는 중에도 듣는다. "자비서"가 들리면 읽기를 끊고 새 질문을 받는다. 그 턴은 `trigger_type=BARGE_IN`.
- 마이크는 브라우저 기본 마이크. 유선 카메라 마이크를 쓰려면 주소창 왼쪽 사이트 설정 → 마이크에서 고른다.
  잡음 제거 · 자동 음량은 끄고 에코 제거는 켠다(주소 끝에 `?aec=0` 이면 에코 제거도 끈다). 콘솔에 `[wake] 브라우저 호출어 켜짐 · <마이크 이름>`.
- 켜지지 않는 경우(콘솔 `[wake] 브라우저 호출어 꺼짐: <이유>`): 휴대폰, https 가 아님, 마이크 권한 거부, 서버에 모델 없음.
  그때 화면 문구는 "호출 버튼이나 스페이스바로 불러 주세요."
- 노트북 파이썬과 점수가 같은지는 `scripts/wakeword/compare_browser.py` 로 잰다(Node + onnxruntime-web).

### 질문 하나가 도는 곳

| 단계 | 어디서 |
|---|---|
| 장면 고르기(말 시작 뒤 3장 중 흔들림 · 반사가 가장 적은 장) | 내 PC |
| 가리킴 · 크롭 · 바코드 · 알레르기 판정 · 기록 | 내 PC |
| VLM 대답 · 기억 임베딩 | OpenAI (내 키) |
| 성분표 글자 읽기 | Google Vision (크롭 사진만 보낸다) |

성분표 OCR 은 Google Vision → (실패하면) Colab → 이 PC 의 EasyOCR(CPU) 순으로 읽는다.
구글 키가 틀렸거나 사용량을 넘으면 서버 로그에 경고가 남고, 한 장에 5~10초씩 걸리면 그 신호다.
평가 세트 21장 비교는 `docs/experiments.md` 의 "성분표 OCR — Google Vision".
알레르기 등록 · 대화 기록은 각자 PC 의 `data/assistant.db` 에만 남는다.

## 알아둘 것

- **로컬에서 실행한다.** 카메라 · 마이크가 이 노트북에 물려 있어서
  클라우드에 올리면 지연만 늘어난다.
- `static/` 은 디자인 원본이다. 화면을 바꿀 때는 `pages/` 의 복사본을 고친다.
