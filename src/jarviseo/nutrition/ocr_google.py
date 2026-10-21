"""성분표 OCR 을 Google Cloud Vision 에 맡긴다.

평가 세트 21장(정답 알레르기 140개)에서 지금 쓰던 Colab PaddleOCR 은 54개를 놓쳤고,
Google Vision(DOCUMENT_TEXT_DETECTION)은 7개를 놓쳤다. 한 장 평균 약 2초로 더 빠르고,
Colab 을 켜 둘 필요도 없다(docs/experiments.md).

.env 에 GOOGLE_VISION_API_KEY 가 있으면 ``ocr._get_reader`` 가 이 클래스를 쓴다.
``readtext`` 모양이 easyocr.Reader 와 같아서 나머지 코드는 그대로다.
요청이 실패하면(키 오류 · 네트워크 · 할당량) 대체 OCR(Colab 또는 이 PC 의 EasyOCR)로 읽는다.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable

import cv2
import httpx
import numpy as np

__all__ = ["GoogleVisionError", "GoogleVisionReader", "lines_from_annotation"]

log = logging.getLogger(__name__)

ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"
# 줄이 끝나는 표시. HYPHEN 은 줄 끝에서 단어가 잘린 경우다.
_LINE_ENDS = {"EOL_SURE_SPACE", "LINE_BREAK", "HYPHEN"}
_SPACES = {"SPACE", "SURE_SPACE"}


class GoogleVisionError(RuntimeError):
    """Google Vision 이 오류를 돌려줬다."""


def _box(vertices: list[dict]) -> tuple[int, int, int, int]:
    xs = [v.get("x", 0) for v in vertices] or [0]  # 0 인 좌표는 응답에서 빠진다
    ys = [v.get("y", 0) for v in vertices] or [0]
    return min(xs), min(ys), max(xs), max(ys)


def lines_from_annotation(annotation: dict) -> list:
    """fullTextAnnotation → [(꼭짓점 4개, 한 줄 글자, 신뢰도), ...] (easyocr.readtext 모양).

    Vision 은 블록 · 문단 · 단어 · 글자 단위로 돌려준다.
    글자마다 붙은 줄바꿈 표시로 줄을 다시 묶는다.
    """
    lines: list = []
    text, boxes, confs = "", [], []

    def flush() -> None:
        nonlocal text, boxes, confs
        if text.strip():
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes)
            y2 = max(b[3] for b in boxes)
            conf = sum(confs) / len(confs) if confs else 1.0
            lines.append(([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], text.strip(), float(conf)))
        text, boxes, confs = "", [], []

    for page in annotation.get("pages", []):
        for block in page.get("blocks", []):
            for para in block.get("paragraphs", []):
                for word in para.get("words", []):
                    for sym in word.get("symbols", []):
                        text += sym.get("text", "")
                        boxes.append(_box(sym.get("boundingBox", {}).get("vertices", [])))
                        confs.append(sym.get("confidence", word.get("confidence", 1.0)))
                        brk = sym.get("property", {}).get("detectedBreak", {}).get("type")
                        if brk in _SPACES:
                            text += " "
                        elif brk in _LINE_ENDS:
                            flush()
                flush()  # 문단이 끝나면 줄도 끝난다
    return lines


class GoogleVisionReader:
    """Google Vision OCR 클라이언트. easyocr.Reader.readtext 와 같은 모양으로 돌려준다.

    Args:
        api_key: Google Cloud API 키(Cloud Vision API 사용 설정 필요).
        fallback: Vision 이 실패할 때 대신 쓸 OCR 을 만드는 함수(처음 실패할 때 한 번 만든다).
    """

    def __init__(
        self,
        api_key: str,
        timeout: float = 20.0,
        fallback: Callable[[], object] | None = None,
        transport: httpx.BaseTransport | None = None,  # 테스트에서 가짜 응답을 끼운다
    ) -> None:
        self._key = api_key
        self._http = httpx.Client(timeout=timeout, transport=transport)
        self._make_fallback = fallback
        self._fallback = None

    def close(self) -> None:
        self._http.close()

    def readtext(self, image: np.ndarray, **_) -> list:
        """[(꼭짓점 4개, 글자, 신뢰도), ...]. Vision 이 안 되면 대체 OCR 로 읽는다."""
        try:
            return self._vision(image)
        except (httpx.HTTPError, GoogleVisionError, ValueError, KeyError) as e:
            if self._make_fallback is None:
                raise
            log.warning("Google Vision 이 실패해 대체 OCR 로 읽는다: %s", e)
            if self._fallback is None:
                self._fallback = self._make_fallback()
            return self._fallback.readtext(image, detail=1, paragraph=False)

    def _vision(self, image: np.ndarray) -> list:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise GoogleVisionError("이미지 인코딩 실패")
        body = {
            "requests": [
                {
                    "image": {"content": base64.b64encode(buf.tobytes()).decode()},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                    "imageContext": {"languageHints": ["ko"]},
                }
            ]
        }
        # 키는 주소에 붙지만 로그에 남지 않게 params 로만 넘긴다
        resp = self._http.post(ENDPOINT, params={"key": self._key}, json=body)
        data = resp.json()
        if resp.status_code != 200 or "error" in data:
            message = data.get("error", {}).get("message", resp.text[:200])
            raise GoogleVisionError(f"{resp.status_code}: {message}")
        first = data["responses"][0]
        if "error" in first:
            raise GoogleVisionError(first["error"].get("message", "알 수 없는 오류"))
        return lines_from_annotation(first.get("fullTextAnnotation", {}))
