"""프로젝트 전역 설정.

경로와 환경변수를 여기 한 곳에서만 읽는다. 각 모듈이 제멋대로
``os.getenv`` 를 부르면, 맥에서는 되는데 윈도우에서는 안 되는 문제가
생겼을 때 어디를 봐야 할지 알 수 없게 된다.

경로 규칙
---------
모든 경로는 ``pathlib.Path`` 로 만든다. 문자열 결합("/" 이어붙이기) 금지.
윈도우는 구분자가 역슬래시라서 문자열로 만들면 팀원 노트북에서 깨진다.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# src/jarviseo/config.py → src/jarviseo → src → 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"  # 학습 데이터 (공용 Drive 에서 받음)
MODELS_DIR = DATA_DIR / "models"  # 학습된 가중치 (Git 에 커밋하지 않음)
VIDEOS_DIR = DATA_DIR / "videos"  # 테스트용 녹화 영상
DOCS_DIR = PROJECT_ROOT / "docs"

# .env 는 각자 로컬에만 두고 커밋하지 않는다. .env.example 을 복사해서 만든다.
load_dotenv(PROJECT_ROOT / ".env")


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _get_int(key: str, default: int) -> int:
    raw = _get(key)
    return int(raw) if raw else default


def _get_float(key: str, default: float) -> float:
    raw = _get(key)
    return float(raw) if raw else default


def _get_bool(key: str, default: bool) -> bool:
    raw = _get(key).lower()
    return raw in ("1", "true", "yes", "on") if raw else default


# --- 프레임 입력 ----------------------------------------------------------
# "usbcam" | "webcam" | "video" | "folder"
FRAME_SOURCE = _get("JARVISEO_FRAME_SOURCE", "webcam")
CAMERA_INDEX = _get_int("JARVISEO_CAMERA_INDEX", 0)
FRAME_WIDTH = _get_int("JARVISEO_FRAME_WIDTH", 1920)
FRAME_HEIGHT = _get_int("JARVISEO_FRAME_HEIGHT", 1080)
# FRAME_SOURCE 가 video/folder 일 때 읽을 경로
FRAME_PATH = _get("JARVISEO_FRAME_PATH")

# --- 프레임 품질 ----------------------------------------------------------
FRAME_BUFFER_SIZE = _get_int("JARVISEO_FRAME_BUFFER_SIZE", 30)
SHARPNESS_THRESHOLD = _get_float("JARVISEO_SHARPNESS_THRESHOLD", 100.0)

# --- 지시 대상 특정 -------------------------------------------------------
# 1위와 2위 점수차가 이 값보다 작으면 단정하지 않고 되묻는다.
# 값을 올리면 되묻는 횟수가 늘고, 내리면 틀린 단정이 늘어난다.
# ablation 실험으로 정할 값이므로 지금은 출발점이다.
CLARIFY_MARGIN_THRESHOLD = _get_float("JARVISEO_CLARIFY_MARGIN", 0.15)
# 손 관절 모델(MediaPipe HandLandmarker). git 에 안 올라가서 각자 받는다(pointing/README.md).
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
# 물체 후보 검출기 가중치. 파일 경로나 ultralytics 모델 이름. 기본은 사전학습 COCO yolov8n 이다.
# 학습한 가중치(data/models/detector_rN.pt, #167)는 시험할 때만 .env 로 바꾼다.
# 기본값 교체는 회차 비교로 나아진 것을 확인한 뒤에 한다.
DETECTOR_WEIGHTS = _get("JARVISEO_DETECTOR_WEIGHTS") or "yolov8n.pt"

# --- 기능 플래그 ----------------------------------------------------------
# 아직 안 끝난 기능은 브랜치에 들고 있지 말고, **꺼둔 채로 main 에 머지한다.**
# 브랜치가 오래 살수록 충돌 비용이 커지기 때문이다.
#
#     if config.ENABLE_GAZE_CUE:
#         scores += score_gaze(...)
#
# 규칙: 기능이 완성되면 플래그와 if 분기를 **같이 지운다.**
# 다 만든 기능의 플래그를 남겨두면 경우의 수만 늘어나고 아무도 안 지운다.
ENABLE_GAZE_CUE = _get_bool("JARVISEO_ENABLE_GAZE_CUE", False)
ENABLE_LANGUAGE_CUE = _get_bool("JARVISEO_ENABLE_LANGUAGE_CUE", False)
ENABLE_RECALL = _get_bool("JARVISEO_ENABLE_RECALL", False)

# --- 외부 API -------------------------------------------------------------
# 팀 공용 OpenAI 키 1개를 STT · VLM · TTS · 임베딩이 함께 쓴다. 월 사용 한도 설정 필수.
# 이 키만 JARVISEO_ 를 붙이지 않는다. OpenAI SDK 가 찾는 이름 그대로 쓴다.
# 시스템 환경변수에 OPENAI_API_KEY 가 이미 있으면 .env 보다 그쪽이 이긴다.
OPENAI_API_KEY = _get("OPENAI_API_KEY")
# 모델 이름은 여기서만 정한다. 모듈마다 기본값을 따로 적어두면 결정이 바뀔 때
# 한쪽만 고쳐져서 어긋난다.
VLM_MODEL = _get("JARVISEO_VLM_MODEL", "gpt-6-sol")
STT_MODEL = _get("JARVISEO_STT_MODEL", "gpt-transcribe")
# gpt-4o-mini-tts 는 첫 조각이 0.7초쯤 빠르지만 읽는 도중 자주 멈춰서 기본으로 두지 않는다.
# 짧은 대답 10번 중 6번이 5.8~10.2초 멈췄고 tts-1 은 0번이었다(10/21). 멈추면 폴백까지 4초를
# 기다리고 처음부터 다시 읽어서, 첫 조각 0.7초를 아끼는 것보다 훨씬 오래 걸렸다.
TTS_MODEL = _get("JARVISEO_TTS_MODEL", "tts-1")
# TTS_MODEL 합성이 멈춰서(읽기 타임아웃) 한 번 더 요청할 때 쓰는 모델. 기본은 같은 tts-1 로
# 다시 보낸다. gpt-4o-mini-tts 는 긴 대답 8번 중 5번이 30초 넘게 멈췄다(#118, 10/20).
TTS_FALLBACK_MODEL = _get("JARVISEO_TTS_FALLBACK_MODEL", "tts-1")
# 기억 검색용 문장 임베딩. 차원(1536)이 observation.embedding 의 VECTOR(1536) 과 같아야 한다.
EMBED_MODEL = _get("JARVISEO_EMBED_MODEL", "text-embedding-3-small")
# 기억 검색 점수(코사인 유사도)가 이보다 낮으면 근거로 넘기지 않는다.
# 기억은 지난 질문·대답 기록이다(memory/dialog.py, #66). 평가셋(data/eval/recall_set.csv
# 대화 20·질문 30, scripts/eval_recall.py, 10/15): 5등 안에 든 맞는 기록은 0.258~0.605 인데
# 기록에 없는 것을 물은 질문의 1등도 0.295~0.385 다. 0.25 까지는 정답을 안 버리지만 기록 없는
# 질문을 하나도 못 거르고, 0.30 부터 정답을 버린다. 장면 설명 방식(#65)도 0.20 까지 같았다.
# 그래서 맞는 것을 버리지 않게 낮게 둔다. 상관없는 질문은 라우터가 RECALL 로 안 보내는 것으로,
# 기록에 없는 것은 VLM 이 기록만 근거로 "기억에 없다"고 답하는 것(vlm/client.py 의 MEMORY_RULE)
# 으로 막는다.
MEMORY_MIN_SCORE = _get_float("JARVISEO_MEMORY_MIN_SCORE", 0.15)

# --- 식품 성분 판정 ------------------------------------------
# 푸드QR 알레르기정보 (식약처). 바코드로 포장에 표시된 알레르기 유발물질 목록을 그대로 준다.
# 원재료가 상위 분류('면류, 분말스프')로 뭉개지지 않아 판정 근거로 바로 쓴다.
# 캐시 다음에 묻는다. 키가 없으면 건너뛴다. (portal.foodqr.kr 에서 발급)
# 외부 API 가 느리면 사용자는 그동안 침묵을 듣는다. 짧게 끊고 OCR 로 넘어간다.
FOOD_QR_API_KEY = _get("JARVISEO_FOOD_QR_API_KEY") or _get("FOOD_QR_API_KEY")
FOOD_QR_BASE_URL = _get("JARVISEO_FOOD_QR_BASE_URL", "https://foodqr.kr/openapi/service")
FOOD_QR_TIMEOUT_SEC = _get_float("JARVISEO_FOOD_QR_TIMEOUT_SEC", 3.0)

# OCR 평균 신뢰도가 이보다 낮으면 검증 실패로 본다 (Validate Agent).
OCR_CONF_THRESHOLD = _get_float("JARVISEO_OCR_CONF_THRESHOLD", 0.8)
# 기획 05-workflow (2026-10-02 반영):
#   검증 실패 → 같은 사진에서 영역 재선택·보정 후 재처리 (사진당 1회)
#   그래도 실패 → "성분표를 더 가까이 가리켜 주세요" 재촬영 (요청당 최대 2회)
#   → 최대 사진 3장 × 사진당 OCR 2회 = 검증 최대 6회
OCR_REPROCESS_PER_PHOTO = _get_int("JARVISEO_OCR_REPROCESS_PER_PHOTO", 1)
OCR_MAX_RETAKE = _get_int("JARVISEO_OCR_MAX_RETAKE", 2)
# 원재료명을 못 찾으면 90°·270° 로 돌려서도 읽는다. 캔·병은 성분표를 옆으로 인쇄하는 일이 있다.
# 그런 사진에서만 OCR 이 최대 3번 돈다. 정방향 사진은 추가 비용이 없다.
OCR_TRY_ROTATION = _get_bool("JARVISEO_OCR_TRY_ROTATION", True)
# 성분표 OCR 을 Colab GPU 서버(scripts/colab_ocr_server.py)에 맡긴다. 비우면 이 PC 의 EasyOCR(CPU).
# 노트북 CPU 로는 한 장에 수십 초가 걸린다. Colab T4 + PaddleOCR 이 훨씬 빠르고 정확하다.
# 서버를 띄울 때 출력되는 주소와 토큰을 그대로 적는다. Colab 세션이 바뀌면 주소도 바뀐다.
OCR_REMOTE_URL = _get("JARVISEO_OCR_REMOTE_URL")
OCR_REMOTE_TOKEN = _get("JARVISEO_OCR_REMOTE_TOKEN")
# 첫 요청은 모델을 데우느라 느릴 수 있어 넉넉히 둔다. 넘기면 이 PC 에서 대신 읽는다.
OCR_REMOTE_TIMEOUT_SEC = _get_float("JARVISEO_OCR_REMOTE_TIMEOUT_SEC", 15.0)
# Google Vision · Colab 이 실패했을 때 이 컴퓨터의 EasyOCR(CPU)로 대신 읽을지. 노트북은 켜 둔다
# (느려도 끝까지 읽는 편이 낫다). 클라우드 서버(t3.small, 2GB)는 끈다(deploy/compose.yml):
# 10/24 로컬 Docker 에서 EasyOCR 로 읽으면 메모리가 3.1GB 까지 오르고 한 장에 10분을 넘겼다(#179).
# 끄면 읽을 엔진이 없을 때 "지금은 성분표를 읽을 수 없다"고 안내한다(nutrition/service.py).
OCR_LOCAL_FALLBACK = _get_bool("JARVISEO_OCR_LOCAL_FALLBACK", True)
# 성분표 영역 자동 찾기 (nutrition/region.py, #75). 사진 전체를 한 번 더 OCR 해서 '함유'·원재료명
# 같은 단서로 성분표만 잘라 크게 읽는다. 평가 세트 21장 재현율 사람 영역 81% / 자동 76%.
# OCR 이 한 번 늘어서 CPU 에서는 느리다. GPU 서버에서 확인한 뒤 켠다.
OCR_AUTO_REGION = _get_bool("JARVISEO_OCR_AUTO_REGION", False)

# OCR 전 사진 품질 검사 (nutrition/quality.py). 흐리거나 반사가 심하면 OCR 을 돌리지 않고
# 다시 보여 달라고 한다. 평가 세트 21장으로 정한 임시값 — 실제 카메라 프레임으로 다시 맞춘다.
#   선명도 비율: 한 번 더 흐리게 했을 때 경계가 줄어드는 비율. 원본 5.6~42, 흐린 사진 3 이하
#   경계 세기: 라플라시안 분산. 원본 77 이상, 심하게 흐린 사진 15 이하
#   반사: 하얗게 날아간 픽셀 비율. 대부분 0.03 이하, 반사가 심한 사진 0.21
QUALITY_MIN_SHARPNESS = _get_float("JARVISEO_QUALITY_MIN_SHARPNESS", 4.0)
QUALITY_MIN_EDGE = _get_float("JARVISEO_QUALITY_MIN_EDGE", 40.0)
QUALITY_MAX_GLARE = _get_float("JARVISEO_QUALITY_MAX_GLARE", 0.15)

# 동의어 사전("탈지분유" → "우유"). 판정 recall 이 이 파일 품질에 달려 있다.
ALLERGEN_SYNONYMS_PATH = DATASETS_DIR / "allergen_synonyms.json"
# 성분표 검출 YOLO 가중치. 학습 전에는 파일이 없고, 그때는 크롭 전체를 OCR 한다.
PANEL_DETECTOR_WEIGHTS = MODELS_DIR / "ingredient_panel.pt"

# --- 관계형 DB (Postgres) -------------------------------------------------
# docker compose 로 띄운 Postgres 에 붙는다.
#   네이티브 실행   → localhost:5432
#   컨테이너 안     → db:5432  (compose 가 환경변수로 덮어쓴다)
#
# compose 의 environment 가 .env 보다 우선한다. load_dotenv 는 이미 설정된
# 환경변수를 덮어쓰지 않기 때문이다(override=False 가 기본값).
# 그래서 같은 .env 를 쓰면서도 컨테이너 안에서만 호스트명이 db 로 바뀐다.
DB_HOST = _get("JARVISEO_DB_HOST", "localhost")
DB_PORT = _get_int("JARVISEO_DB_PORT", 5432)
DB_NAME = _get("JARVISEO_DB_NAME", "jarviseo")
DB_USER = _get("JARVISEO_DB_USER", "jarviseo")
DB_PASSWORD = _get("JARVISEO_DB_PASSWORD", "jarviseo")

# 통째로 지정하고 싶으면 이 값을 쓴다. 위 항목들보다 우선한다.
# SQLite 로 되돌리려면 여기에 이렇게 넣으면 된다:
#   JARVISEO_DATABASE_URL=sqlite:///data/jarviseo.db
# 발표 데모에서 맥북 메모리가 빠듯하면 이 방법으로 컨테이너 없이 돌릴 수 있다.
DATABASE_URL = _get("JARVISEO_DATABASE_URL") or (
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
# Postgres 연결을 기다리는 최대 시간(초). 없으면 윈도우에서 컨테이너가 꺼져 있을 때
# psycopg 가 닫힌 포트를 알아채지 못하고 끝없이 기다린다(#131).
# localhost 는 주소 두 개(127.0.0.1, ::1)를 차례로 시도해서 실제로는 이것의 2배까지 걸린다.
DB_CONNECT_TIMEOUT_SEC = _get_int("JARVISEO_DB_CONNECT_TIMEOUT_SEC", 3)

# --- 음성 ----------------------------------------------------------------
WAKE_WORD = _get("JARVISEO_WAKE_WORD", "자비서")
# 마이크·스피커. sounddevice 번호나 장치 이름 일부("USB", "Buds"). 비우면 윈도우·맥 기본 장치.
# 번호는 노트북마다, USB 를 꽂는 순서마다 바뀌어서 이름을 적는 편이 낫다(voice/devices.py).
# assistant.py 에 번호를 직접 주면 이 값 대신 그것을 쓴다.
MIC_DEVICE = _get("JARVISEO_MIC")
SPEAKER_DEVICE = _get("JARVISEO_SPEAKER")
# TTS 재생이 끝난 뒤 이만큼 더 마이크를 막는다.
# 스피커 소리를 마이크가 다시 듣고 자기 응답에 반응하는 되먹임 루프를 막는 장치다.
TTS_MIC_GATE_SEC = _get_float("JARVISEO_TTS_MIC_GATE_SEC", 0.3)
# 호출어 모델. .onnx 경로나 openWakeWord 기성 모델 이름(hey_jarvis 등).
# 기본은 "자비서"로 학습한 커스텀 모델이다(6회차, data/README.md 에서 받는 법). Git 에는 없다.
WAKEWORD_MODEL = _get("JARVISEO_WAKEWORD_MODEL") or str(MODELS_DIR / "jarviseo.onnx")
# 호출어 점수(0~1)가 이 값을 넘으면 깨어난다. 0.97 은 평가셋에서 FRR 조용 3.8%·소음 16.7%,
# 비슷한 발음 FAR 5% 였다(TV 30분 3회는 기준 미달, docs/experiments.md 학습 기록 6회차).
# 재생 중 임계값은 따로 둔다. 자기 목소리(TTS 대답)는 실제 재생 6개·약 2분 반에서 최고 0.001 이라
# 평소와 같은 0.97 로 둔다. 0.99 로는 재생 중 붙여 말한 "자비서"를 자주 놓쳤다(잡힌 점수 0.974~).
WAKEWORD_THRESHOLD = _get_float("JARVISEO_WAKEWORD_THRESHOLD", 0.97)
WAKEWORD_PLAYBACK_THRESHOLD = _get_float("JARVISEO_WAKEWORD_PLAYBACK_THRESHOLD", 0.97)
# 호출어 감지는 "자비서"가 끝나고 조금 뒤에 된다(실제 녹음 37개에서 중앙 0.12초, 최대 0.33초).
# listen(since=감지 시각)은 이만큼 앞에서부터 읽어, 호출어에 바로 이어 말한 질문 앞부분이
# 잘리지 않게 한다. 호출어("자비서" 길이 중앙 0.57초)가 통째로 들어가야 받아쓴 뒤
# transcript.clean 이 지운다. 0.4초로는 끝 "서"만 들어가 "그래서"·"여기서"로 받아써졌다.
WAKE_LOOKBACK_SEC = _get_float("JARVISEO_WAKE_LOOKBACK_SEC", 1.0)
# 시작음("자비서"를 알아들음)·종료음(질문을 다 들음) 크기. 0~1 로 파일 소리에 곱한다. 0 이면 끔.
CHIME_VOLUME = _get_float("JARVISEO_CHIME_VOLUME", 1.0)
# VLM 답이 이 시간(초) 안에 안 오면 "잠깐만요"를 먼저 읽는다. 안경에는 화면이 없어서
# 아무 소리가 없으면 사용자는 못 들은 줄 안다. gpt-6-sol 왕복은 3번 재서 1.7~2.8초였다(10/14).
FILLER_AFTER_SEC = _get_float("JARVISEO_FILLER_AFTER_SEC", 1.5)

# VLM 답을 이 시간(초)까지만 기다리고, 넘으면 안내 문장으로 턴을 끝낸다. 클라이언트 타임아웃이
# 10초 × 재시도 1번이라 그대로 두면 최악 20초를 기다린다. spike 3 에서 45번 중 1번(6.5초)만 넘었다.
VLM_DEADLINE_SEC = _get_float("JARVISEO_VLM_DEADLINE_SEC", 5.0)

# --- 클라우드 대시보드로 보내기 (#144) -------------------------------------
# assistant.py 가 끝난 턴을 이 서버(https://….duckdns.org)로 보낸다(memory/uploader.py).
# 둘 중 하나라도 비면 보내지 않는다.
# 토큰은 서버의 같은 이름 값(Parameter Store /jarviseo/JARVISEO_CLOUD_TOKEN)과 같아야 받는다.
CLOUD_URL = _get("JARVISEO_CLOUD_URL")
CLOUD_TOKEN = _get("JARVISEO_CLOUD_TOKEN")

# --- 켜기 (scripts/start.py) ---------------------------------------------
# 켤 때 main 의 새 버전을 받는다(git pull). 확인한 버전 그대로 쓰려면 false 로 고정한다.
# 누가 main 에 막 머지한 코드를 모르고 받아 잘 되던 노트북이 갑자기 안 켜지는 일을 막는다.
AUTO_UPDATE = _get_bool("JARVISEO_AUTO_UPDATE", True)
