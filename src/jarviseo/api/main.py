"""FastAPI 진입점.  담당: 권용현

로컬에서 실행한다. 실시간 파이프라인을 클라우드에 올리지 않는다.
카메라와 마이크가 이 노트북에 물려 있어서, 클라우드에 올리면 지연만 늘어난다.

Docker 주의
-----------
macOS 의 Docker 는 USB 장치 접근이 사실상 불가능하다.
그래서 캡처 프로세스는 네이티브로 실행하고, 처리 서버만 컨테이너에 넣어
HTTP/WebSocket 으로 잇는다. 이 파일은 후자에 해당한다.

대시보드 레이아웃: 실시간 영상 / 현재 대화 / 채팅 이력 / 단계별 지연 그래프

지금 되는 것 (성분 확인)
------------------------
    uvicorn jarviseo.api.main:app --port 8000      # 레포 맨 위 폴더에서
    http://localhost:8000/ingredient                # 사진을 올리면 성분을 분석한다

    GET  /api/allergens      등록된 알레르기
    PUT  /api/allergens      알레르기 목록 바꾸기   {"allergens": ["계란", "땅콩"]}
    POST /api/ingredient     본문 = 사진 바이트(JPEG/PNG). ?allergens=계란,땅콩 이 없으면 등록된 것

알레르기는 assistant.py · register_allergens.py 와 같은 DB · 같은 사용자(owner)에 둔다.
성분표 OCR 은 .env 에 JARVISEO_OCR_REMOTE_URL 이 있으면 Colab GPU 에서 돈다.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from jarviseo.memory import MemoryStore
from jarviseo.nutrition.service import IngredientChecker, make_checker

# assistant.py 의 기본 DB 와 같아야 등록한 알레르기를 앱과 같이 쓴다.
API_DB = os.environ.get("JARVISEO_API_DB", "sqlite:///data/assistant.db")
PAGES = Path(__file__).parent / "pages"
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 휴대폰 원본 사진도 들어가는 크기
SOURCE_NAMES = {"barcode": "바코드 · 푸드QR", "ocr": "성분표 OCR", "retake": "다시 찍기 안내"}


class Allergens(BaseModel):
    allergens: list[str]


def create_app(
    store: MemoryStore | None = None, checker: IngredientChecker | None = None
) -> FastAPI:
    """앱을 만든다. 테스트에서는 가짜 store · checker 를 넣는다."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store or MemoryStore(database_url=API_DB)
        app.state.store.init_schema()
        app.state.user_id = app.state.store.ensure_user()
        # OCR 모델 · 동의어 사전 · 푸드QR 연결을 한 번만 만든다.
        app.state.checker = checker or make_checker(app.state.store)
        app.state.lock = threading.Lock()  # OCR 엔진 하나를 여러 요청이 동시에 쓰지 않게
        yield
        app.state.checker.close()

    app = FastAPI(title="자비서 (JARVISEO)", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        """서버가 살아 있는지 확인한다."""
        return {"status": "ok"}

    @app.get("/ingredient", include_in_schema=False)
    def ingredient_page() -> FileResponse:
        """사진을 올려 성분을 확인하는 페이지."""
        return FileResponse(PAGES / "ingredient.html")

    @app.get("/api/allergens")
    def get_allergens(request: Request) -> Allergens:
        s = request.app.state
        return Allergens(allergens=s.store.get_user_allergens(s.user_id))

    @app.put("/api/allergens")
    def put_allergens(body: Allergens, request: Request) -> Allergens:
        s = request.app.state
        names = list(dict.fromkeys(a.strip() for a in body.allergens if a.strip()))
        s.store.set_user_allergens(s.user_id, names)
        return Allergens(allergens=s.store.get_user_allergens(s.user_id))

    @app.post("/api/ingredient")
    async def analyze(request: Request, allergens: str | None = None) -> dict:
        """사진 한 장을 분석한다. 본문은 사진 바이트 그대로(JPEG/PNG)."""
        data = await request.body()
        if not 0 < len(data) <= MAX_IMAGE_BYTES:
            raise HTTPException(413 if data else 400, "사진이 없거나 너무 큽니다 (15MB 까지)")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "사진을 읽을 수 없습니다 (JPEG/PNG)")
        s = request.app.state
        names = [a.strip() for a in (allergens or "").split(",") if a.strip()]
        names = names or s.store.get_user_allergens(s.user_id)
        return await _run_in_thread(lambda: _check(s, image, names))

    return app


async def _run_in_thread(fn):
    # OCR 은 수 초 걸린다. 그동안 다른 요청(페이지 · 알레르기 조회)이 멈추지 않게 스레드에서 돌린다.
    return await anyio.to_thread.run_sync(fn)


def _check(s, image: np.ndarray, allergens: list[str]) -> dict:
    t0 = time.perf_counter()
    with s.lock:
        r = s.checker.check(image, allergens)
    lookup, product = r.barcode, r.judgement.product
    if r.retake:
        source = "retake"
    elif lookup is not None and lookup.hit and not r.ocr_text:
        source = "barcode"
    else:
        source = "ocr"
    return {
        "sentence": r.retake or r.message,
        "verdict": r.judgement.verdict.value,
        "source": source,
        "source_name": SOURCE_NAMES[source],
        "retake": bool(r.retake),
        "occlusion": r.occlusion,
        "barcode": lookup.barcode.code if lookup and lookup.barcode else None,
        "product": product.product_name if product and product.product_name else None,
        "allergens": allergens,
        "timings_ms": r.timings_ms,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
    }


app = create_app()
