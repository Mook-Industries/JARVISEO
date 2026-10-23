"""클라우드 대시보드 앱(api/cloud.py) — 헬스 체크, 무거운 모듈을 안 불러오는지, 이미지 의존성 버전.

앱 테스트는 fastapi 가 있어야 해서 로컬에서만 돈다(CI 는 requirements-dev.txt 만 깐다).
버전 비교 테스트는 CI 에서도 돈다.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def pins(path: Path) -> dict[str, str]:
    """`이름==버전` 줄을 {소문자 이름: 버전} 으로. [standard] 같은 추가 기능 표시는 뗀다."""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^\s;]+)", line)
        if m:
            found[m.group(1).lower()] = m.group(2)
    return found


def test_api_requirements_match_lock():
    """requirements-api.txt 의 버전이 requirements.lock 과 같다(노트북과 클라우드가 같은 버전)."""
    api = pins(ROOT / "requirements-api.txt")
    lock = pins(ROOT / "requirements.lock")

    assert api, "requirements-api.txt 에 고정한 버전이 없다"
    assert {name: lock.get(name) for name in api} == api


def test_cloud_app_does_not_import_heavy_modules():
    """클라우드 앱은 cv2 · numpy · torch 같은 무거운 모듈을 안 불러온다(작은 이미지에 없다)."""
    pytest.importorskip("fastapi")
    code = (
        "import sys, jarviseo.api.cloud;"
        "print([m for m in ('cv2', 'numpy', 'torch', 'ultralytics', 'easyocr', 'mediapipe',"
        " 'openai', 'langgraph') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "[]"


def test_health_and_db_check(tmp_path):
    """/health 는 늘 200, /health/db 는 DB 에 붙으면 200 · 못 붙으면 503 과 이유를 준다."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from jarviseo.api.cloud import create_app

    ok = TestClient(create_app(f"sqlite:///{tmp_path / 'cloud.db'}"))
    assert ok.get("/health").json() == {"status": "ok"}
    assert ok.get("/health/db").json()["status"] == "ok"

    missing = tmp_path / "없는 폴더" / "cloud.db"
    bad = TestClient(create_app(f"sqlite:///{missing}")).get("/health/db")
    assert bad.status_code == 503
    assert bad.json()["status"] == "error" and bad.json()["reason"]
