"""FastAPI 진입점.  담당: C

로컬에서 실행한다. 실시간 파이프라인을 클라우드에 올리지 않는다.
카메라와 마이크가 이 노트북에 물려 있어서, 클라우드에 올리면 지연만 늘어난다.

Docker 주의
-----------
macOS 의 Docker 는 USB 장치 접근이 사실상 불가능하다.
그래서 캡처 프로세스는 네이티브로 실행하고, 처리 서버만 컨테이너에 넣어
HTTP/WebSocket 으로 잇는다. 이 파일은 후자에 해당한다.

대시보드 레이아웃: 실시간 영상 / 현재 대화 / 채팅 이력 / 단계별 지연 그래프
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="자비서 (JARVISEO)", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """서버가 살아 있는지 확인한다."""
    return {"status": "ok"}


# TODO(C): 아래 엔드포인트들
# GET  /              대시보드 페이지
# WS   /ws/stream     실시간 영상 프레임 + 현재 상태 전송
# GET  /api/history   대화 이력 조회 (이것만 선택적으로 서버 배포 가능)
# GET  /api/latency   단계별 지연 통계 (p50/p95)
# POST /api/ask       텍스트로 질문 (카메라 없이 테스트할 때)
