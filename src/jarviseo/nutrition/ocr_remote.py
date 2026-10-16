"""성분표 OCR 을 Colab GPU 서버에 맡긴다.

노트북 CPU 로 EasyOCR 을 돌리면 성분표 한 장에 수십 초가 걸려 대답을 기다릴 수 없다.
Colab T4 GPU 의 PaddleOCR(PP-OCRv5 server_det + korean_PP-OCRv5_mobile_rec)은 훨씬 빠르고,
평가 세트 21장 CER 도 EasyOCR 0.664 → 0.365 로 낫다(docs/experiments.md).

서버는 scripts/colab_ocr_server.py 를 Colab 에서 띄운다. 주소(trycloudflare.com)와 토큰을
.env 의 JARVISEO_OCR_REMOTE_URL / JARVISEO_OCR_REMOTE_TOKEN 에 적으면 ``ocr._get_reader`` 가
이 클래스를 쓴다. ``readtext`` 모양이 easyocr.Reader 와 같아서 나머지 코드는 그대로다.

Colab 은 세션이 끊기면 주소가 바뀐다. 서버에 닿지 않으면 노트북 EasyOCR(CPU)로 대신 읽는다.
느리지만 대답은 한다.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import cv2
import httpx
import numpy as np

from jarviseo import config

__all__ = ["RemoteOcrError", "RemoteOcrReader"]

log = logging.getLogger(__name__)


class RemoteOcrError(RuntimeError):
    """OCR 서버가 이상한 응답을 줬다."""


class RemoteOcrReader:
    """Colab OCR 서버 클라이언트. easyocr.Reader.readtext 와 같은 모양으로 돌려준다.

    Args:
        url: 서버 주소 (예: https://xxxx.trycloudflare.com).
        token: 서버가 시작할 때 출력한 토큰. 주소만 알면 누구나 쓰지 못하게 막는다.
        fallback: 서버에 닿지 않을 때 대신 쓸 OCR 을 만드는 함수(처음 실패할 때 한 번 만든다).
    """

    def __init__(
        self,
        url: str,
        token: str = "",
        timeout: float | None = None,
        fallback: Callable[[], object] | None = None,
        transport: httpx.BaseTransport | None = None,  # 테스트에서 가짜 응답을 끼운다
    ) -> None:
        self.url = url.rstrip("/")
        self._http = httpx.Client(
            timeout=timeout or config.OCR_REMOTE_TIMEOUT_SEC,
            transport=transport,
            headers={"X-Token": token} if token else {},
        )
        self._make_fallback = fallback
        self._fallback = None

    def close(self) -> None:
        self._http.close()

    def readtext(self, image: np.ndarray, **_) -> list:
        """[(꼭짓점 4개, 글자, 신뢰도), ...]. 서버가 안 되면 대체 OCR 로 읽는다."""
        try:
            return self._remote(image)
        except (httpx.HTTPError, RemoteOcrError, ValueError) as e:
            if self._make_fallback is None:
                raise
            log.warning("OCR 서버(%s)에 닿지 않아 이 PC 에서 읽는다: %s", self.url, e)
            if self._fallback is None:
                self._fallback = self._make_fallback()
            return self._fallback.readtext(image, detail=1, paragraph=False)

    def _remote(self, image: np.ndarray) -> list:
        # JPEG 95: PNG 보다 몇 배 작고 글자 경계는 OCR 결과가 달라지지 않을 만큼 남는다.
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RemoteOcrError("이미지 인코딩 실패")
        resp = self._http.post(
            f"{self.url}/ocr", content=buf.tobytes(), headers={"Content-Type": "image/jpeg"}
        )
        resp.raise_for_status()
        data = resp.json()
        if "lines" not in data:
            raise RemoteOcrError(f"응답에 lines 가 없다: {str(data)[:200]}")
        return [(pts, str(text), float(conf)) for pts, text, conf in data["lines"]]
