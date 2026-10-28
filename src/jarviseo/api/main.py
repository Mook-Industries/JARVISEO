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
    GET · PUT /api/trace     미량 혼입 가능성 알림 {"on": true}. 끄면 함유만 말한다
    POST /api/ingredient     본문 = 사진 바이트(JPEG/PNG). ?allergens=계란,땅콩 이 없으면 등록된 것
                             알레르기에 '모두' · '테스트'를 넣으면 법정 알레르기 전체(시험용)

노트북 대화 기록 (#144)
----------------------
    POST /api/turns          노트북 assistant.py 가 끝난 턴을 보낸다(memory/uploader.py).
                             Authorization: Bearer <JARVISEO_CLOUD_TOKEN>.
                             서버에 토큰이 없으면 안 받는다
    GET  /api/turns          대시보드 채팅 이력. ?after=<turn_id> 면 그 뒤 턴만(3초마다 묻는다)

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

import asyncio
import base64
import contextlib
import hmac
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC
from pathlib import Path
from typing import Literal

import anyio
import cv2
import httpx
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from jarviseo import config
from jarviseo.api.assistant_web import WebAssistant, draw_boxes
from jarviseo.api.browser_voice import FRAME, HEADER, BrowserVoice
from jarviseo.api.marks import jpeg_data_url, mark_findings
from jarviseo.api.report import report
from jarviseo.api.trace import TraceChecker, get_trace, set_trace
from jarviseo.api.voice_loop import VoiceLoop
from jarviseo.memory import MemoryStore, SessionTurn
from jarviseo.memory.store import UPLOAD_TITLE
from jarviseo.nutrition.allergen import (
    expand_allergens,
    find_allergens,
    normalize_term,
)
from jarviseo.nutrition.ocr import google_vision_key
from jarviseo.nutrition.service import IngredientChecker, make_checker
from jarviseo.types import IngredientSource

# assistant.py 의 기본 DB 와 같아야 등록한 알레르기를 앱과 같이 쓴다.
API_DB = os.environ.get("JARVISEO_API_DB", "sqlite:///data/assistant.db")
PAGES = Path(__file__).parent / "pages"
STATIC = Path(__file__).parent / "static"  # 사용자 목업(디자인 원본). 읽기만 한다
NO_CACHE = {"Cache-Control": "no-cache"}  # 고친 페이지가 새로고침 한 번에 보이게
MAX_MORE_FRAMES = 4  # 말 시작 뒤 잇달아 찍는 장면 수 상한(대시보드는 2장을 더 보낸다)
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 휴대폰 원본 사진도 들어가는 크기
OCR_NOTEBOOK_URL = os.environ.get("JARVISEO_OCR_NOTEBOOK_URL", "")
OCR_STATUS_TTL_SEC = 5.0  # 여러 탭이 동시에 물어봐도 Colab 에는 5초에 한 번만 묻는다
SOURCE_NAMES = {"barcode": "바코드 · 푸드QR", "ocr": "성분표 OCR", "retake": "다시 찍기 안내"}
# 알레르기 칸에 이 말을 넣으면 법정 알레르기 전체로 판정한다. 인식 성능을 볼 때 쓴다.
# 근거 구간 이름. 원재료·알레르기 표시에서 찾으면 함유, 혼입 문장에서만 찾으면 혼입 가능이다.
SECTIONS = (("ingredients", "원재료"), ("notice", "알레르기 표시"), ("cross", "같은 제조시설"))
# 노트북이 턴을 보낼 때 함께 보내는 토큰(#144). 비어 있으면 이 서버는 턴을 받지 않는다.
CLOUD_TOKEN = os.environ.get("JARVISEO_CLOUD_TOKEN", "").strip()
MAX_UPLOAD_TURNS = 100  # 노트북은 20개씩 보낸다(memory/uploader.BATCH)
MAX_FEED_TURNS = 50
# 브라우저 호출어 2단계 확인에 받는 구간 상한(5초). 보통은 1.6초 = 51,200바이트
MAX_WAKE_CLIP_BYTES = 16000 * 2 * 5
# 브라우저 호출어(#218)가 받는 모델. 공용 특징 모델 2개 + 팀 모델(config.WAKEWORD_MODEL)
WAKEWORD_FEATURES = ("melspectrogram.onnx", "embedding_model.onnx")
WAKEWORD_FILE = "jarviseo.onnx"


class Allergens(BaseModel):
    allergens: list[str]


class VoiceSettings(BaseModel):
    """환경설정 AI 음성. 없는 칸은 그대로 둔다. voice 는 voice/tts.py VOICES 중 하나."""

    voice: str | None = None
    speed: float | None = Field(None, ge=0.5, le=2.0)  # 말하기 속도 배수
    volume: float | None = Field(None, ge=0.0, le=1.0)  # 소리 크기


class Trace(BaseModel):
    """환경설정 '미량 혼입 가능성 알림'. on = 받기(혼입 가능까지 말함), off = 함유만."""

    on: bool


class Ask(BaseModel):
    question: str
    image: str | None = None  # data URL 또는 base64 JPEG/PNG. 없으면 사진 없이 묻는다
    # 말을 시작한 뒤 잇달아 찍은 장면들과 찍은 시각(말 시작으로부터 ms). 가장 선명한 장면을 쓴다.
    more: list[str] = []
    more_ms: list[int] = []
    # 대답을 읽는 중 "자비서"로 끼어들어 물었으면 BARGE_IN(브라우저 호출어, #218).
    # session_turn.trigger_type 에 남는다.
    trigger: Literal["WAKEWORD", "BARGE_IN"] = "WAKEWORD"


class UploadedTurn(BaseModel):
    """노트북이 보낸 턴 하나(memory/store.turn_payload). 열 이름은 ERD 그대로다."""

    local_turn_id: int
    local_session_id: int
    turn: dict
    voice: dict | None = None
    inference: dict | None = None


class TurnBatch(BaseModel):
    source: str = Field(min_length=1, max_length=40)  # 노트북마다 정한 임의 이름
    turns: list[UploadedTurn] = Field(max_length=MAX_UPLOAD_TURNS)


def create_app(
    store: MemoryStore | None = None,
    checker: IngredientChecker | None = None,
    assistant: WebAssistant | None = None,
    cloud_token: str | None = None,
    wake_check=None,
    openai_client=None,
) -> FastAPI:
    """앱을 만든다. 테스트에서는 가짜 store · checker 와 토큰, 호출어 확인(WakeCheck)을 넣는다.

    openai_client 는 테스트에서 브라우저 음성(/api/voice/ws)의 받아쓰기 · 읽기를 가짜로 바꿀 때만.
    """
    token = CLOUD_TOKEN if cloud_token is None else cloud_token

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store or MemoryStore(database_url=API_DB)
        app.state.store.init_schema()
        app.state.user_id = app.state.store.ensure_user()
        # OCR 모델 · 동의어 사전 · 푸드QR 연결을 한 번만 만든다.
        # '미량 혼입 가능성 알림'을 끄면 함유만 말한다.
        # 판정기 하나를 감싸 모든 길(웹 · 음성 · 성분 페이지)에 같게 한다(api/trace.py).
        app.state.checker = TraceChecker(
            checker or make_checker(app.state.store), app.state.store, app.state.user_id
        )
        app.state.lock = threading.Lock()  # OCR 엔진 하나를 여러 요청이 동시에 쓰지 않게
        app.state.assistant = assistant or WebAssistant(
            app.state.store, app.state.checker, app.state.user_id
        )
        # 브라우저 호출어 2단계 확인(#218). OpenAI 키가 있을 때 처음 확인할 때 만든다
        app.state.wake_check = wake_check
        app.state.voice = VoiceLoop(
            app.state.store, app.state.checker, app.state.user_id, app.state.lock, _voice_result
        )
        yield
        app.state.voice.stop()
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
        return FileResponse(PAGES / "dashboard.html", headers=NO_CACHE)

    @app.get("/dashboard.js", include_in_schema=False)
    def dashboard_js() -> FileResponse:
        return FileResponse(PAGES / "dashboard.js", media_type="text/javascript", headers=NO_CACHE)

    @app.get("/wakeword.js", include_in_schema=False)
    def wakeword_js() -> FileResponse:
        return FileResponse(PAGES / "wakeword.js", media_type="text/javascript", headers=NO_CACHE)

    @app.get("/voicelink.js", include_in_schema=False)
    def voicelink_js() -> FileResponse:
        return FileResponse(PAGES / "voicelink.js", media_type="text/javascript", headers=NO_CACHE)

    @app.get("/api/wakeword")
    def wakeword_info() -> dict:
        """브라우저 호출어를 켤 수 있는지(모델 3개가 다 있는지)와 임계값. 노트북과 같은 값이다."""
        return {
            "available": all(
                wakeword_file(n) is not None for n in (*WAKEWORD_FEATURES, WAKEWORD_FILE)
            ),
            "threshold": config.WAKEWORD_THRESHOLD,
            "playback_threshold": config.WAKEWORD_PLAYBACK_THRESHOLD,
            # 2단계 확인: 깨어나면 직전 check_sec 초를 POST /api/wake-check 로 보낸다
            "check": config.WAKE_CHECK,
            "check_sec": config.WAKE_CHECK_SEC,
        }

    @app.post("/api/wake-check")
    async def wake_check_clip(request: Request) -> dict:
        """브라우저 호출어가 깨어난 직전 구간(16kHz 모노 int16 PCM)을 받아써 호출어였는지 본다.

        노트북 2단계 확인(voice/wakecheck.py, #149)과 같은 코드다. 키가 브라우저에 가지 않게
        받아쓰기는 서버가 한다. 받아쓰지 못하면(키 없음 · 오류) 통과로 본다.
        진짜 호출을 놓치지 않게.
        """
        data = await request.body()
        if len(data) % 2 or len(data) > MAX_WAKE_CLIP_BYTES:
            raise HTTPException(400, f"16비트 PCM, {MAX_WAKE_CLIP_BYTES} 바이트까지입니다")
        audio = np.frombuffer(data, dtype="<i2")
        return await _run_in_thread(lambda: _wake_check(request.app.state, audio))

    @app.get("/models/wakeword/{name}", include_in_schema=False)
    def wakeword_model(name: str) -> FileResponse:
        """브라우저 호출어 모델 3개(약 2.6MB). 대시보드가 켤 때 한 번 받는다.

        no-cache: 릴리스로 모델을 바꿔 배포하면(#232) 다음 새로고침에 새 파일을 받게 한다.
        그대로면 ETag 로 304 만 오가서 다시 받지 않는다.
        """
        path = wakeword_file(name)
        if path is None:
            raise HTTPException(404, "없는 호출어 모델입니다")
        return FileResponse(path, media_type="application/octet-stream", headers=NO_CACHE)

    @app.get("/sounds/{name}", include_in_schema=False)
    def sound(name: str) -> FileResponse:
        """호출을 받은 시작음 · 질문이 끝난 종료음. 안경(voice/chime.py)과 같은 파일을 쓴다."""
        from jarviseo.voice.chime import END_WAV, START_WAV

        path = {"start.wav": START_WAV, "end.wav": END_WAV}.get(name)
        if path is None or not path.is_file():
            raise HTTPException(404, "없는 소리입니다")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/ingredient", include_in_schema=False)
    def ingredient_page() -> FileResponse:
        """사진을 올려 성분을 확인하는 페이지."""
        return FileResponse(PAGES / "ingredient.html", headers=NO_CACHE)

    @app.get("/assistant", include_in_schema=False)
    def assistant_page() -> FileResponse:
        """사진과 질문으로 자비서 전체 흐름을 돌려 보는 페이지."""
        return FileResponse(PAGES / "assistant.html", headers=NO_CACHE)

    @app.post("/api/ask")
    async def ask(body: Ask, request: Request) -> dict:
        """질문 한 번을 팀 그래프로 돌린다. 사진이 있으면 가리킴 · 성분 · VLM 이 그 사진을 본다."""
        question = body.question.strip()
        if not question:
            raise HTTPException(400, "질문이 비었습니다")
        image = _decode_image(body.image) if body.image else None
        if len(body.more) != len(body.more_ms) or len(body.more) > MAX_MORE_FRAMES:
            raise HTTPException(
                400, f"more 와 more_ms 의 개수가 같아야 합니다({MAX_MORE_FRAMES}장까지)"
            )
        more = [
            (_decode_image(m), ms / 1000) for m, ms in zip(body.more, body.more_ms, strict=True)
        ]
        s = request.app.state
        if s.voice.running:
            # 팀 음성 모드: 입력 질문도 같은 팀 부품으로(서버 스피커로 읽고, 장면은 카메라 버퍼에서)
            return await _run_in_thread(lambda: s.voice.ask_text(question, image))
        return await _run_in_thread(
            lambda: _ask(s, question, image, more if image is not None else [], body.trigger)
        )

    @app.post("/api/turns")
    def receive_turns(body: TurnBatch, request: Request, authorization: str = Header("")) -> dict:
        """노트북(assistant.py)이 보낸 끝난 턴을 이 서버 DB 에 남긴다. 토큰이 맞아야 받는다.

        이 길은 Caddy 접속 비밀번호를 거치지 않는다(deploy/Caddyfile). 토큰이 그 대신이다.
        같은 턴을 다시 보내면 건너뛴다(store.import_turns).
        """
        if not token:
            raise HTTPException(503, "이 서버는 턴을 받지 않습니다 (JARVISEO_CLOUD_TOKEN 없음)")
        if not hmac.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
            raise HTTPException(401, "토큰이 틀립니다")
        turns = [t.model_dump() for t in body.turns]
        return request.app.state.store.import_turns(body.source, turns)

    @app.get("/api/turns")
    def turn_feed(request: Request, after: int | None = None, limit: int = 30) -> dict:
        """대시보드 채팅 이력. 노트북에서 받은 턴과 이 서버에서 물은 턴(웹)을 같이 준다.

        after 가 없으면 최근 limit 개를 새것부터, 있으면 그 뒤 턴을 들어온 순서로 준다.
        """
        limit = min(max(limit, 1), MAX_FEED_TURNS)
        return {"turns": [_feed_item(t) for t in request.app.state.store.turn_feed(after, limit)]}

    # ---- 팀 음성 모드(voice_loop.py): 마이크 · 스피커 · 웹캠을 서버가 열고 팀 부품으로 돈다 ----

    @app.get("/api/report")
    def performance_report(request: Request) -> dict:
        """성능 리포트: 실제 턴 지연 · 평가 실행 · 팀 실측 결과(api/report.py)."""
        return report(request.app.state.store)

    @app.post("/api/voice/start")
    async def voice_start(request: Request) -> dict:
        """팀 음성 모드를 켠다. 처음엔 모델을 읽느라 10~20초 걸린다. 못 켜면 on=False 와 이유."""
        voice = request.app.state.voice
        try:
            await _run_in_thread(voice.start)
        except Exception as e:  # 장치 · 모델 · 키가 없으면 브라우저 방식으로 돌아간다
            voice.reason = str(e)
            return {"on": False, "reason": voice.reason}
        return {"on": True, "pointing": voice.pointing}

    @app.post("/api/voice/stop")
    async def voice_stop(request: Request) -> dict:
        await _run_in_thread(request.app.state.voice.stop)
        return {"on": False}

    @app.post("/api/voice/call")
    def voice_call(request: Request) -> dict:
        """화면의 호출 버튼 = "자비서"라고 부른 것."""
        request.app.state.voice.call()
        return {"ok": True}

    @app.post("/api/voice/settings")
    def voice_settings(body: VoiceSettings, request: Request) -> dict:
        """환경설정 AI 음성을 팀 음성 모드(PC 스피커)에 쓴다. 켜기 전에 보내면 켤 때 쓴다."""
        voice = request.app.state.voice
        voice.configure_voice(**body.model_dump())
        return {"on": voice.running, **voice.voice_settings}

    @app.get("/api/voice/events")
    def voice_events(request: Request, after: int = 0) -> dict:
        voice = request.app.state.voice
        events, last = voice.events_after(after)
        return {"on": voice.running, "events": events, "last": last}

    @app.post("/api/voice/frame")
    async def voice_frame(request: Request) -> dict:
        """대시보드 카메라의 장면(JPEG 바이트)을 팀 음성 모드의 프레임 버퍼에 넣는다."""
        data = await request.body()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "사진이 너무 큽니다")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "사진을 읽지 못했습니다")
        request.app.state.voice.push_frame(image)
        return {"ok": True}

    @app.websocket("/api/voice/ws")
    async def voice_ws(ws: WebSocket) -> None:
        """브라우저 마이크 · 스피커로 팀 음성 부품을 돌린다(api/browser_voice.py, #218).

        서버에 마이크가 없을 때(클라우드) 대시보드가 연다. 받아쓰기 · 읽기 · "잠깐만요" · 되묻기 ·
        끼어들기가 노트북과 같은 코드로 돈다. 키가 없으면 off 를 보내고 닫는다.
        """
        await ws.accept()
        s = ws.app.state
        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue[bytes | dict] = asyncio.Queue()

        def send(msg: bytes | dict) -> None:  # 아무 스레드에서나 부른다. 보내는 줄은 하나다
            # 연결이 끝난 뒤에 나는 소리(시작음 스레드 등)는 버린다
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(outbox.put_nowait, msg)

        async def sender() -> None:
            with contextlib.suppress(Exception):  # 끊긴 뒤 남은 것은 버린다
                while True:
                    msg = await outbox.get()
                    if isinstance(msg, bytes):
                        await ws.send_bytes(msg)
                    else:
                        await ws.send_json(msg)

        voice = BrowserVoice(
            s.store, s.checker, s.user_id, s.lock, s.assistant, send, _voice_result, openai_client
        )
        task = asyncio.create_task(sender())
        try:
            try:
                await _run_in_thread(voice.start)
            except Exception as e:
                await ws.send_json({"type": "off", "reason": str(e)})
                return
            while True:
                m = await ws.receive()
                if m["type"] == "websocket.disconnect":
                    break
                if m.get("bytes"):
                    data = m["bytes"]
                    if data[0] == FRAME and len(data) > HEADER.size:
                        # 장면은 풀고 선명도를 재느라 몇십 ms 걸린다. 소리를 받는 줄을 막지 않게
                        await _run_in_thread(lambda d=data: voice.on_binary(d))
                    else:
                        voice.on_binary(data)
                elif m.get("text"):
                    with contextlib.suppress(ValueError):
                        voice.on_message(json.loads(m["text"]))
        except WebSocketDisconnect:
            pass
        finally:
            await _run_in_thread(voice.close)
            task.cancel()
            with contextlib.suppress(Exception):
                await ws.close()

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
        trace = get_trace(s.store, s.user_id)  # 새로 더한 알레르기도 같은 알림 설정을 따르게
        s.store.set_user_allergens(s.user_id, names)
        set_trace(s.store, s.user_id, trace)
        return Allergens(allergens=s.store.get_user_allergens(s.user_id))

    @app.get("/api/trace")
    def get_trace_setting(request: Request) -> Trace:
        """환경설정 '미량 혼입 가능성 알림'. true = 받기(혼입 가능까지 말함)."""
        s = request.app.state
        return Trace(on=get_trace(s.store, s.user_id))

    @app.put("/api/trace")
    def put_trace_setting(body: Trace, request: Request) -> Trace:
        s = request.app.state
        set_trace(s.store, s.user_id, body.on)
        return Trace(on=get_trace(s.store, s.user_id))

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


def _feed_item(t: SessionTurn) -> dict:
    title = (t.session.title if t.session is not None else None) or ""
    asked = t.asked_at.replace(tzinfo=UTC) if t.asked_at.tzinfo is None else t.asked_at
    return {
        "id": t.turn_id,
        "asked_at": asked.isoformat(),  # 브라우저가 자기 시각대로 바꿔 보여 준다
        "where": "노트북" if title.startswith(UPLOAD_TITLE) else "웹",
        "question": t.question_text,
        "answer": t.answer_text,
        "target": t.inference.target_label if t.inference is not None else None,
        "total_ms": t.total_ms,
    }


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
    source_name = SOURCE_NAMES[source]
    if source == "barcode" and lookup.product.source is IngredientSource.CACHE:
        # 푸드QR 을 다시 부르지 않고 저장해 둔 바코드 정보를 썼다
        source_name = "바코드 · 저장된 정보"
    return {
        "sentence": r.retake or r.message,
        "verdict": r.judgement.verdict.value,
        "source": source,
        "source_name": source_name,
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


def _voice_result(state, t0: float, loop) -> dict:
    """팀 음성 모드 · 브라우저 음성의 턴 결과를 /api/ask 와 같은 모양으로. 대답은 이미 읽었다."""
    from types import SimpleNamespace

    crop, frame = state.get("crop"), state.get("frame")
    shim = SimpleNamespace(
        store=loop.store,
        user_id=loop.user_id,
        checker=loop.checker,
        assistant=SimpleNamespace(
            vlm_real=True,
            pointing=loop.pointing,
            retried_whole=getattr(loop, "retried_whole", False),
            picked_index=0,
            judged_image=_judged(loop, crop, frame),
        ),
    )
    response = state.get("response")
    spoken = [response.text] if response is not None and response.text else []
    result = _result(shim, state["utterance"].text, state, spoken, t0)
    result["voice"] = True
    result["filler"] = bool(state.get("filler_sent"))
    result["interrupted"] = bool(state.get("interrupted"))
    return result


def _judged(loop, crop, frame):
    """성분 판정에 실제로 넣은 사진. 브라우저 음성(browser_voice.py)은 따로 적어 둔다."""
    judged = getattr(loop, "judged_image", None)
    if judged is not None:
        return judged
    return crop if crop is not None else (frame.image if frame else None)


def wakeword_file(name: str) -> Path | None:
    """브라우저 호출어 모델 파일. 없으면 None.

    팀 모델은 config.WAKEWORD_MODEL(.onnx 경로일 때만). 공용 특징 모델은 data/models/
    (웹 이미지가 빌드 때 넣는다)를 먼저 보고, 없으면 노트북의 openwakeword 패키지 폴더
    (scripts/download_models.py).
    """
    if name == WAKEWORD_FILE:
        path = Path(config.WAKEWORD_MODEL)
        return path if path.suffix == ".onnx" and path.is_file() else None
    if name not in WAKEWORD_FEATURES:
        return None
    path = config.MODELS_DIR / name
    if path.is_file():
        return path
    try:
        import openwakeword  # 리눅스(서버 이미지)에는 없다
    except ImportError:
        return None
    path = Path(openwakeword.__file__).parent / "resources" / "models" / name
    return path if path.is_file() else None


def _wake_check(s, audio: np.ndarray) -> dict:
    if s.wake_check is None:
        if not config.OPENAI_API_KEY:
            return {"passed": True, "text": None, "ms": 0}
        from jarviseo.voice.wakecheck import WakeCheck  # openai 를 쓸 때만 읽는다

        s.wake_check = WakeCheck()
    from jarviseo.voice.transcript import has_wake_word

    t0 = time.perf_counter()
    text = s.wake_check.check(audio)  # 실패하면 None(통과). 노트북과 같은 로그를 남긴다
    ms = round((time.perf_counter() - t0) * 1000)
    return {"passed": text is None or has_wake_word(text), "text": text, "ms": ms}


def _ask(s, question: str, image: np.ndarray | None, more=None, trigger="WAKEWORD") -> dict:
    t0 = time.perf_counter()
    with s.lock:  # 성분 판정기(OCR)를 성분 페이지와 같이 쓴다
        state, spoken = s.assistant.ask(question, image, more or None, trigger=trigger)
    return _result(s, question, state, spoken, t0)


def _result(s, question: str, state, spoken: list[str], t0: float) -> dict:
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
        "picked_frame": getattr(s.assistant, "picked_index", 0),  # 0 = 말 시작 순간
    }
    check = state.get("ingredient")
    if check is not None:
        allergens = expand_allergens(s.store.get_user_allergens(s.user_id))
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
