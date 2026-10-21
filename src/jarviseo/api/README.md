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

서버는 **각자 자기 PC 에서** 켠다. 같이 쓰는 건 성분표 글자를 읽는 Colab OCR 하나뿐이다.

**1. 설치** — 레포 README 의 [시작하기 1번](../../../README.md#1-파이썬-환경)
(venv → `pip install -r requirements.txt` → `pip install -e .`). 한 번만 하면 된다.

**2. `.env`** — 레포 맨 위 폴더의 `.env` 에 이것만 채운다.

| 항목 | 값 | 없으면 |
|---|---|---|
| `OPENAI_API_KEY` | **각자 본인 키** | 대답 · 기억 검색이 가짜로 돈다 |
| `JARVISEO_OCR_REMOTE_URL` | 권용현이 켜 둔 Colab 주소 (고정, 따로 전달) | 이 PC CPU 로 읽는다 (한 장 5~10초) |
| `JARVISEO_OCR_REMOTE_TOKEN` | 위 주소의 토큰 (따로 전달) | Colab 이 요청을 거절한다 |
| `JARVISEO_FOOD_QR_API_KEY` | 푸드QR 키 | 바코드로 성분을 못 가져오고 OCR 로만 읽는다 |

키 · 토큰은 레포나 채팅방에 올리지 않는다.

**3. 실행**

```bash
python -m jarviseo.api          # http://localhost:8000 이 열린다. 끄려면 Ctrl+C
```

카메라 · 마이크는 브라우저가 켠다. 처음에 뜨는 "카메라 · 마이크 허용"을 누른다.
`localhost` 로 열어야 켜진다(브라우저는 localhost · https 에서만 카메라를 허락한다).

### 질문 하나가 도는 곳

| 단계 | 어디서 |
|---|---|
| 가리킴 · 크롭 · 바코드 · 알레르기 판정 · 기록 | 내 PC |
| VLM 대답 · 기억 임베딩 | OpenAI (내 키) |
| 성분표 글자 읽기 | 같이 쓰는 Colab GPU (크롭 사진만 보낸다) |

Colab 이 꺼져 있으면 화면 위에 "OCR 서버 꺼짐"이 뜨고 이 PC 의 CPU 로 대신 읽는다.
알레르기 등록 · 대화 기록은 각자 PC 의 `data/assistant.db` 에만 남는다.

## 알아둘 것

- **로컬에서 실행한다.** 카메라 · 마이크가 이 노트북에 물려 있어서
  클라우드에 올리면 지연만 늘어난다.
- `static/` 은 디자인 원본이다. 화면을 바꿀 때는 `pages/` 의 복사본을 고친다.
