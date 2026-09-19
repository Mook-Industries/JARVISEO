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
ENABLE_BELONGING = _get_bool("JARVISEO_ENABLE_BELONGING", False)
ENABLE_RECALL = _get_bool("JARVISEO_ENABLE_RECALL", False)

# --- 외부 API -------------------------------------------------------------
# 팀 공용 키 1개를 팀장이 관리한다. 월 사용 한도 설정 필수.
LLM_API_KEY = _get("JARVISEO_LLM_API_KEY")
VLM_MODEL = _get("JARVISEO_VLM_MODEL", "claude-sonnet-5")

# --- 저장소 ---------------------------------------------------------------
SQLITE_PATH = Path(_get("JARVISEO_SQLITE_PATH") or (DATA_DIR / "jarviseo.db"))
CHROMA_PATH = Path(_get("JARVISEO_CHROMA_PATH") or (DATA_DIR / "chroma"))

# --- 음성 ----------------------------------------------------------------
WAKE_WORD = _get("JARVISEO_WAKE_WORD", "자비서")
# TTS 재생이 끝난 뒤 이만큼 더 마이크를 막는다.
# 스피커 소리를 마이크가 다시 듣고 자기 응답에 반응하는 되먹임 루프를 막는 장치다.
TTS_MIC_GATE_SEC = _get_float("JARVISEO_TTS_MIC_GATE_SEC", 0.3)
