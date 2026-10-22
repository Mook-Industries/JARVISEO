"""클라우드에 올리는 대시보드 서버.  담당: 문태현 (배포)

노트북의 ``api/main.py`` 는 웹에서 자비서를 돌리느라(가리킴 · 성분표 OCR · VLM)
cv2 · YOLO · torch 를 불러온다. 클라우드 서버는 기록을 받아 보여 주기만 해서 그것들이 필요 없다.
그래서 이 앱은 FastAPI · DB 드라이버만 쓰는 작은 이미지(``Dockerfile.api``)로
뜬다. **여기서 cv2 · numpy · torch 나 jarviseo 의 무거운 모듈을 불러오지 않는다**
(tests/api/test_cloud.py 가 본다).

지금은 서버와 DB 가 살아 있는지만 본다. 노트북이 턴을 보내는 길(``POST /api/turns``, #144)과
휴대폰으로 보는 화면(#143)은 웹 담당과 모양을 정한 뒤 여기에 붙인다.

    uvicorn jarviseo.api.cloud:app --host 0.0.0.0 --port 8000
    docker compose up api        # 같은 것을 컨테이너로 (http://localhost:8000/health)

    GET /health       서버가 떠 있으면 200
    GET /health/db    DB(JARVISEO_DATABASE_URL 또는 JARVISEO_DB_*)에 붙으면 200, 못 붙으면 503
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, make_url, text
from sqlalchemy.engine import Engine

from jarviseo import config


def _engine(url: str) -> Engine:
    # Postgres 가 꺼져 있을 때 끝없이 기다리지 않게 연결 시간을 자른다(#131 과 같은 값).
    args = (
        {"connect_timeout": config.DB_CONNECT_TIMEOUT_SEC} if url.startswith("postgresql") else {}
    )
    return create_engine(url, pool_pre_ping=True, connect_args=args)


def create_app(database_url: str | None = None) -> FastAPI:
    """앱을 만든다. 테스트에서는 SQLite 주소를 넘긴다."""
    url = database_url or config.DATABASE_URL
    engine = _engine(url)
    app = FastAPI(title="자비서 대시보드 (클라우드)", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        """서버가 살아 있는지 확인한다. 클라우드의 헬스 체크가 이 주소를 부른다."""
        return {"status": "ok"}

    @app.get("/health/db")
    def health_db() -> JSONResponse:
        """DB 에 붙는지 확인한다. 주소의 비밀번호는 가리고 돌려준다."""
        where = make_url(url).render_as_string(hide_password=True)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:  # noqa: BLE001 — 어떤 이유든 "못 붙음"으로 알린다
            reason = str(getattr(e, "orig", None) or e).strip().splitlines()[0]
            return JSONResponse({"status": "error", "db": where, "reason": reason}, 503)
        return JSONResponse({"status": "ok", "db": where})

    return app


app = create_app()
