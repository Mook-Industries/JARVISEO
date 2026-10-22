"""FastAPI 진입점.  담당: 권용현

로컬에서 실행한다. 실시간 파이프라인을 클라우드에 올리지 않는다.
카메라와 마이크가 이 노트북에 물려 있어서, 클라우드에 올리면 지연만 늘어난다.

Docker 주의
-----------
macOS 의 Docker 는 USB 장치 접근이 사실상 불가능하다.
그래서 캡처 프로세스는 네이티브로 실행하고, 처리 서버만 컨테이너에 넣어
HTTP/WebSocket 으로 잇는다. 이 파일은 후자에 해당한다.

대시보드 레이아웃: 실시간 영상 / 현재 대화 / 채팅 이력 / 단계별 지연 그래프

대시보드 (사용자 목업 디자인)
----------------------------
    http://localhost:8000/                          # 모니터: 호출 → 질문 → 팀 그래프 → 대답
    api/static/ 의 목업은 그대로 두고(직접 고치지 않는다), 디자인을 복사한
    pages/dashboard.html · dashboard.js 에 실제 API 를 붙였다. CSS 는 /static/jarviseo.css 원본.

지금 되는 것 (성분 확인)
------------------------
    uvicorn jarviseo.api.main:app --port 8000      # 레포 맨 위 폴더에서
    http://localhost:8000/ingredient                # 사진을 올리면 성분을 분석한다

    GET  /api/ocr-status     OCR 이 켜져 있는지. Google Vision 이면 늘 켜짐,
                             Colab 이 꺼져 있으면 페이지에 "Colab 열기" 버튼
    GET  /api/allergens      등록된 알레르기
    PUT  /api/allergens      알레르기 목록 바꾸기   {"allergens": ["계란", "땅콩"]}
    POST /api/ingredient     본문 = 사진 바이트(JPEG/PNG). ?allergens=계란,땅콩 이 없으면 등록된 것
                             알레르기에 '모두' · '테스트'를 넣으면 법정 알레르기 전체(시험용)

자비서 전체 흐름 (팀 그래프를 웹으로)
-----------------------------------
    http://localhost:8000/assistant                 # 사진 + 질문 → 자비서 대답
    POST /api/ask            {"question": "저거 뭐야?", "image": "data:image/jpeg;base64,..."}
    의도 · 가리킴 · VLM · 기억 · 성분 · 기록을 workflow/graph.py 그대로 돈다(api/assistant_web.py).

알레르기는 assistant.py · register_allergens.py 와 같은 DB · 같은 사용자(owner)에 둔다.
성분표 OCR 은 .env 에 JARVISEO_OCR_REMOTE_URL 이 있으면 Colab GPU 에서 돈다.
무료 · 유료 Colab 모두 밖에서 노트북을 켜는 방법이 없어서, 꺼져 있으면 페이지가 알려 주고
JARVISEO_OCR_NOTEBOOK_URL(서버 노트북 주소)을 여는 버튼을 보여 준다. 켜면 페이지가 알아서 바뀐다.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from jarviseo import config
from jarviseo.api.assistant_web import WebAssistant, draw_boxes
from jarviseo.api.marks import jpeg_data_url, mark_findings
from jarviseo.memory import MemoryStore
from jarviseo.nutrition.allergen import KNOWN_ALLERGENS, find_allergens, normalize_term
from jarviseo.nutrition.ocr import google_vision_key
from jarviseo.nutrition.service import IngredientChecker, make_checker
from jarviseo.types import IngredientSource

# assistant.py 의 기본 DB 와 같아야 등록한 알레르기를 앱과 같이 쓴다.
API_DB = os.environ.get("JARVISEO_API_DB", "sqlite:///data/assistant.db")
PAGES = Path(__file__).parent / "pages"
STATIC = Path(__file__).parent / "static"  # 사용자 목업(디자인 원본). 읽기만 한다
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 휴대폰 원본 사진도 들어가는 크기
OCR_NOTEBOOK_URL = os.environ.get("JARVISEO_OCR_NOTEBOOK_URL", "")
OCR_STATUS_TTL_SEC = 5.0  # 여러 탭이 동시에 물어봐도 Colab 에는 5초에 한 번만 묻는다
SOURCE_NAMES = {"barcode": "바코드 · 푸드QR", "ocr": "성분표 OCR", "retake": "다시 찍기 안내"}
# 알레르기 칸에 이 말을 넣으면 법정 알레르기 전체로 판정한다. 인식 성능을 볼 때 쓴다.
ALL_KEYWORDS = {"모두", "테스트", "전체"}
# 근거 구간 이름. 원재료·알레르기 표시에서 찾으면 함유, 혼입 문장에서만 찾으면 혼입 가능이다.
SECTIONS = (("ingredients", "원재료"), ("notice", "알레르기 표시"), ("cross", "같은 제조시설"))


class Allergens(BaseModel):
    allergens: list[str]


class Ask(BaseModel):
    question: str
    image: str | None = None  # data URL 또는 base64 JPEG/PNG. 없으면 사진 없이 묻는다


def create_app(
    store: MemoryStore | None = None,
    checker: IngredientChecker | None = None,
    assistant: WebAssistant | None = None,
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
        app.state.assistant = assistant or WebAssistant(
            app.state.store, app.state.checker, app.state.user_id
        )
        yield
        app.state.assistant.close()
        app.state.checker.close()

    app = FastAPI(title="자비서 (JARVISEO)", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        """서버가 살아 있는지 확인한다."""
        return {"status": "ok"}

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard_page() -> FileResponse:
        """대시보드(모니터 · 관리자 · 환경설정). 목업 디자인에 실제 API 를 붙였다."""
        return FileResponse(PAGES / "dashboard.html")

    @app.get("/dashboard.js", include_in_schema=False)
    def dashboard_js() -> FileResponse:
        return FileResponse(PAGES / "dashboard.js", media_type="text/javascript")

    @app.get("/ingredient", include_in_schema=False)
    def ingredient_page() -> FileResponse:
        """사진을 올려 성분을 확인하는 페이지."""
        return FileResponse(PAGES / "ingredient.html")

    @app.get("/assistant", include_in_schema=False)
    def assistant_page() -> FileResponse:
        """사진과 질문으로 자비서 전체 흐름을 돌려 보는 페이지."""
        return FileResponse(PAGES / "assistant.html")

    @app.post("/api/ask")
    async def ask(body: Ask, request: Request) -> dict:
        """질문 한 번을 팀 그래프로 돌린다. 사진이 있으면 가리킴 · 성분 · VLM 이 그 사진을 본다."""
        question = body.question.strip()
        if not question:
            raise HTTPException(400, "질문이 비었습니다")
        image = _decode_image(body.image) if body.image else None
        s = request.app.state
        return await _run_in_thread(lambda: _ask(s, question, image))

    @app.get("/api/ocr-status")
    def ocr_status(request: Request) -> dict:
        """Colab OCR 서버 상태. remote=False 면 원래부터 이 PC CPU 로 읽는 설정이다."""
        s = request.app.state
        now = time.monotonic()
        cached = getattr(s, "ocr_status", None)
        if cached is None or now - cached[0] > OCR_STATUS_TTL_SEC:
            cached = (now, _probe_ocr(config.OCR_REMOTE_URL))
            s.ocr_status = cached
        return {**cached[1], "notebook_url": OCR_NOTEBOOK_URL}

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
        names = expand_allergens(names or s.store.get_user_allergens(s.user_id))
        return await _run_in_thread(lambda: _check(s, image, names))

    return app


def _probe_ocr(url: str | None) -> dict:
    """OCR 서버의 /health 를 짧게 물어본다. 꺼져 있으면 ngrok 이 안내 페이지(HTML)를 준다.

    Google Vision 키가 있으면 Colab 을 쓰지 않으므로 묻지 않고 켜짐으로 알린다.
    """
    if google_vision_key():
        return {"remote": True, "up": True, "device": "google-vision"}
    if not url:
        return {"remote": False, "up": False, "device": None}
    try:
        r = httpx.get(f"{url.rstrip('/')}/health", timeout=3.0)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    except (httpx.HTTPError, ValueError):
        body = {}
    up = bool(body.get("ok"))
    return {"remote": True, "up": up, "device": body.get("device") if up else None}


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
        "evidence": (ev := evidence(r, allergens, getattr(s.checker, "synonyms", {}) or {})),
        "barcode_steps": barcode_steps(r),
        "where": jpeg_data_url(where_image(image, r, ev)),
        "ocr_text": r.ocr_text,
        "reason": r.judgement.reason,
        "timings_ms": r.timings_ms,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
    }


def _decode_image(data: str) -> np.ndarray:
    raw = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        buf = base64.b64decode(raw, validate=False)
    except ValueError:
        raise HTTPException(400, "사진을 읽을 수 없습니다") from None
    if len(buf) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "사진이 너무 큽니다 (15MB 까지)")
    image = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "사진을 읽을 수 없습니다 (JPEG/PNG)")
    return image


def _ask(s, question: str, image: np.ndarray | None) -> dict:
    t0 = time.perf_counter()
    with s.lock:  # 성분 판정기(OCR)를 성분 페이지와 같이 쓴다
        state, spoken = s.assistant.ask(question, image)
    response = state.get("response")
    target = state.get("target")
    chosen = target.chosen if target is not None else None
    result = {
        "question": question,
        "heard": state["utterance"].text if state.get("utterance") else None,
        "intent": getattr(state.get("intent"), "value", state.get("intent")),
        "answer": response.text if response is not None else None,
        "spoken": spoken,
        "target": chosen.detection.label if chosen is not None else None,
        # 확신도: 고른 대상의 점수, 1위 - 2위 점수차(후보가 하나면 1.0)
        "target_score": round(chosen.total_score, 3) if chosen is not None else None,
        "margin": round(target.margin, 3) if target is not None else None,
        "candidates": len(target.candidates) if target is not None else 0,
        "boxes": jpeg_data_url(draw_boxes(state)),
        "crop": jpeg_data_url(state.get("crop")),
        "memories": [
            {"text": m.text, "score": round(m.score, 3)} for m in state.get("memories") or []
        ],
        "vlm_real": s.assistant.vlm_real,
        "pointing": getattr(s.assistant, "pointing", "v1"),  # v2 = 손끝 방향까지 본다
        "turn_id": state.get("turn_id"),
        # stt · tts · total 은 말하고 듣는 시간이라 웹(가짜 STT · TTS)에서는 뜻이 없다
        "latency_ms": {
            k: round(v)
            for k, v in (state.get("latency_ms") or {}).items()
            if k not in {"stt", "tts", "total"}
        },
        "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        "ingredient": None,
        "retried_whole": s.assistant.retried_whole,
    }
    check = state.get("ingredient")
    if check is not None:
        allergens = s.store.get_user_allergens(s.user_id)
        ev = evidence(check, allergens, getattr(s.checker, "synonyms", {}) or {})
        judged = s.assistant.judged_image  # 판정기에 실제로 들어간 사진(크롭 또는 사진 전체)
        result["ingredient"] = {
            "evidence": ev,
            "barcode_steps": barcode_steps(check),
            "where": jpeg_data_url(where_image(judged, check, ev)) if judged is not None else None,
            "ocr_text": check.ocr_text,
            "reason": check.judgement.reason,
        }
    return result


def where_image(image: np.ndarray, r, ev: list[dict]) -> np.ndarray | None:
    """판정에 쓴 사진에 찾은 글줄 · 바코드 자리를 그린다(api/marks.py)."""
    read = r.barcode.barcode if r.barcode is not None else None
    return mark_findings(
        image,
        getattr(r, "ocr_lines", []) or [],
        ev,
        barcode=read.bbox if read is not None else None,
        barcode_code=read.code if read is not None else None,
    )


def expand_allergens(names: list[str]) -> list[str]:
    """'모두' · '테스트'가 있으면 법정 알레르기 전체를 더한다. 나머지 이름은 그대로 둔다."""
    if not any(n.strip() in ALL_KEYWORDS for n in names):
        return names
    rest = [n for n in names if n.strip() not in ALL_KEYWORDS]
    return list(dict.fromkeys([*KNOWN_ALLERGENS, *rest]))


def evidence(r, allergens: list[str], synonyms: dict[str, list[str]]) -> list[dict]:
    """알레르기마다 어디서(원재료 · 알레르기 표시 · 같은 제조시설) 어떤 글자로 찾았는지.

    판정(allergen.judge_allergens)과 같은 동의어 사전 · 같은 찾기 함수를 쓴다.
    status: contains(함유) / may(혼입 가능) / not_found(못 찾음). 못 찾음은 '안 들어 있음'이 아니다.
    ingredient_only: '~ 함유' 알레르기 표시가 읽혔는데 원재료에서만 찾았다. 판정은 그대로
    함유로 두지만(놓치는 것보다 낫다), 오독일 수 있어서 화면에 따로 알린다
    (크런키 실측: '코코아분말'을 '코코아분밀'로 읽어 밀).
    """
    p = r.judgement.product
    texts = {
        "ingredients": p.raw_ingredients if p else "",
        "notice": p.allergen_notice if p else "",
        "cross": p.cross_contamination if p else "",
    }
    canonical = {normalize_term(t): a for a, ts in synonyms.items() for t in ts}
    table = dict(synonyms)
    wanted: dict[str, str] = {}  # 표준명 → 사용자가 쓴 이름
    for a in allergens:
        key = canonical.get(normalize_term(a), normalize_term(a))
        if key and key not in wanted:
            wanted[key] = a.strip()
            table.setdefault(key, [key])
    hits = {k: find_allergens(t, table) for k, t in texts.items()}
    notice_read = bool(hits["notice"])  # 함유 표시에서 알레르기 이름을 하나라도 읽었나
    out = []
    for key, label in wanted.items():
        where = [{"section": name, "terms": hits[k][key]} for k, name in SECTIONS if key in hits[k]]
        found = {w["section"] for w in where}
        if found & {"원재료", "알레르기 표시"}:
            status = "contains"
        elif found:
            status = "may"
        else:
            status = "not_found"
        only = status == "contains" and notice_read and found == {"원재료"}
        out.append({"allergen": label, "status": status, "where": where, "ingredient_only": only})
    return out


def barcode_steps(r) -> list[str]:
    """바코드로 무엇을 시도했는지 순서대로. 화면에 '바코드 → 조회 → 성분표' 단계로 보여 준다."""
    lookup = r.barcode
    if lookup is None:
        return ["바코드 확인 안 함"]
    if lookup.barcode is None:
        return ["바코드를 찾지 못함", "성분표를 읽음"] if r.ocr_text else ["바코드를 찾지 못함"]
    if lookup.barcode.symbology == "TEXT":
        # 줄무늬는 못 읽고, 성분표 OCR 글자에 찍힌 바코드 숫자로 찾았다(service._from_ocr)
        steps = [
            "바코드 줄무늬를 못 읽음",
            "성분표를 읽음",
            f"글자에서 바코드 번호 {lookup.barcode.code} 찾음",
        ]
        if lookup.hit:
            where = "저장된 정보" if lookup.product.source is IngredientSource.CACHE else "푸드QR"
            return [*steps, f"{where}에서 제품을 찾음"]
        reasons = {"not_found": "푸드QR에 제품 정보가 없음", "api_error": "푸드QR 조회 실패"}
        no_panel = r.judgement.reason == "no_ingredients"
        last = "원재료명을 찾지 못함" if no_panel else "읽은 성분표로 판정"
        return [*steps, reasons.get(lookup.reason, "제품을 찾지 못함"), last]
    steps = [f"바코드 {lookup.barcode.code} 읽음"]
    if lookup.hit:
        where = "저장된 정보" if lookup.product.source is IngredientSource.CACHE else "푸드QR"
        steps.append(f"{where}에서 제품을 찾음")
        if r.ocr_text:
            steps.append("알레르기 표시가 없어 성분표도 읽음")
        return steps
    reasons = {
        "not_found": "푸드QR에 제품 정보가 없음",
        "no_ingredients": "원재료 정보가 없음",
        "api_error": "푸드QR 조회 실패",
    }
    steps.append(reasons.get(lookup.reason, "제품을 찾지 못함"))
    if r.ocr_text:
        steps.append("성분표를 읽음")
    return steps


app = create_app()
