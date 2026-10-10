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
# 팀 공용 OpenAI 키 1개를 STT · VLM · TTS 가 함께 쓴다. 월 사용 한도 설정 필수.
# 이 키만 JARVISEO_ 를 붙이지 않는다. OpenAI SDK 가 찾는 이름 그대로 쓴다.
# 시스템 환경변수에 OPENAI_API_KEY 가 이미 있으면 .env 보다 그쪽이 이긴다.
OPENAI_API_KEY = _get("OPENAI_API_KEY")
# 모델 이름은 여기서만 정한다. 모듈마다 기본값을 따로 적어두면 결정이 바뀔 때
# 한쪽만 고쳐져서 어긋난다.
VLM_MODEL = _get("JARVISEO_VLM_MODEL", "gpt-6-sol")
STT_MODEL = _get("JARVISEO_STT_MODEL", "gpt-transcribe")
TTS_MODEL = _get("JARVISEO_TTS_MODEL", "gpt-4o-mini-tts")

# --- 식품 성분 판정 ------------------------------------------
# 식품안전나라 OpenAPI. C005(바코드 → 품목제조보고번호) → C002/C006(원재료명).
# 발급 전에는 "sample" 로 두면 샘플 데이터 몇 건만 응답한다.
# C005 바코드 데이터는 2018년 이후 갱신이 멈춰 있어, 못 찾으면 OCR 로 넘어간다.
# .env 에 빈 값으로 적혀 있으면 os.getenv 가 "" 를 돌려주므로 or 로 한 번 더 받는다.
MFDS_API_KEY = _get("JARVISEO_MFDS_API_KEY") or "sample"
MFDS_BASE_URL = _get("JARVISEO_MFDS_BASE_URL", "http://openapi.foodsafetykorea.go.kr/api")
# 외부 API 가 느리면 사용자는 그동안 침묵을 듣는다. 짧게 끊고 OCR 로 넘어간다.
MFDS_TIMEOUT_SEC = _get_float("JARVISEO_MFDS_TIMEOUT_SEC", 3.0)

# OCR 평균 신뢰도가 이보다 낮으면 검증 실패로 본다 (Validate Agent).
OCR_CONF_THRESHOLD = _get_float("JARVISEO_OCR_CONF_THRESHOLD", 0.8)
# 기획 05-workflow (2026-10-02 반영):
#   검증 실패 → 같은 사진에서 영역 재선택·보정 후 재처리 (사진당 1회)
#   그래도 실패 → "성분표를 더 가까이 가리켜 주세요" 재촬영 (요청당 최대 2회)
#   → 최대 사진 3장 × 사진당 OCR 2회 = 검증 최대 6회
OCR_REPROCESS_PER_PHOTO = _get_int("JARVISEO_OCR_REPROCESS_PER_PHOTO", 1)
OCR_MAX_RETAKE = _get_int("JARVISEO_OCR_MAX_RETAKE", 2)

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

# --- 음성 ----------------------------------------------------------------
WAKE_WORD = _get("JARVISEO_WAKE_WORD", "자비서")
# TTS 재생이 끝난 뒤 이만큼 더 마이크를 막는다.
# 스피커 소리를 마이크가 다시 듣고 자기 응답에 반응하는 되먹임 루프를 막는 장치다.
TTS_MIC_GATE_SEC = _get_float("JARVISEO_TTS_MIC_GATE_SEC", 0.3)
