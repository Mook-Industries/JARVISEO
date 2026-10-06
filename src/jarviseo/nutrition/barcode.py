"""바코드 → 제품 정보 조회.  (Barcode Lookup Agent)

흐름: 크롭 이미지 → 바코드 번호 읽기 → (다음 커밋) product 캐시 → 식품안전나라 API
      여기서 원재료명을 못 얻으면 호출한 쪽이 성분표 OCR 로 넘어간다.

바코드 읽기에 pyzbar 대신 OpenCV 내장 검출기를 쓰는 이유
-----------------------------------------------------
pyzbar 는 윈도우에서 Visual C++ 2013 런타임이 없으면 DLL 로드가 실패한다.
팀원 노트북마다 설치 여부가 달라서 "내 PC 에서만 안 됨"이 생긴다.
OpenCV(cv2.barcode)는 이미 의존성에 있고 EAN-13/EAN-8/UPC 를 읽는다.

LLM 을 쓰지 않는다. 바코드 번호는 체크섬으로 검증하고, 제품 정보는 공공 API 그대로다.
"""

from __future__ import annotations

import cv2
import numpy as np

from jarviseo.types import BarcodeRead, BBox

__all__ = [
    "is_valid_gtin",
    "read_barcodes",
]

# 식품 포장에 쓰이는 소매용 바코드만 받는다. QR·Code128 등은 제품 번호가 아니다.
_RETAIL_TYPES = {"EAN_13", "EAN_8", "UPC_A", "UPC_E"}
# cv2.barcode 는 막대 굵기 2px 안팎에서 가장 잘 읽고, 막대가 너무 굵거나 가늘면 놓친다.
# 그래서 긴 변을 아래 크기들로 맞춰 가며 시도한다(None = 원본 크기).
#   - 크롭이 바코드에 딱 맞게 잘린 경우 → 320 에서 거의 다 읽힌다
#   - 큰 크롭 속 작은 바코드 → 800 이상에서 읽힌다
# 한 번에 10ms 안팎이라 최악(7번)에도 0.1초 수준이다.
_TARGET_LONG_SIDES: tuple[int | None, ...] = (None, 320, 480, 640, 800, 960, 1280)

_detector: cv2.barcode.BarcodeDetector | None = None


# --------------------------------------------------------------------------
# 1. 이미지에서 바코드 번호 읽기
# --------------------------------------------------------------------------


def _get_detector() -> cv2.barcode.BarcodeDetector:
    # 검출기 생성이 가볍지 않아 프로세스당 하나만 만든다.
    global _detector
    if _detector is None:
        _detector = cv2.barcode.BarcodeDetector()
    return _detector


def is_valid_gtin(code: str) -> bool:
    """GTIN(EAN-8/UPC-A/EAN-13/GTIN-14) 체크섬 검증.

    검출기가 막대 하나를 잘못 읽으면 엉뚱한 제품이 조회된다.
    마지막 자리(체크 디지트)로 걸러낸다.
    """
    if not code.isdigit() or len(code) not in (8, 12, 13, 14):
        return False
    digits = [int(c) for c in code]
    body, check = digits[:-1], digits[-1]
    # 체크 디지트 바로 왼쪽부터 3, 1, 3, 1 ... 가중치
    total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return (10 - total % 10) % 10 == check


def _prepare(gray: np.ndarray, long_side: int | None) -> tuple[np.ndarray, float, int]:
    """검출기에 넣을 이미지로 바꾼다. (이미지, 배율, 여백) 을 돌려준다.

    - 크기 맞춤: 위 _TARGET_LONG_SIDES 설명 참고
    - 여백: 크롭이 바코드 끝에 딱 붙어 있으면 검출기가 영역을 못 잡는다
    - 약한 블러: 경계가 지나치게 날카로우면(인쇄물 스캔 등) 오히려 놓친다
    """
    scale = 1.0 if long_side is None else long_side / max(gray.shape[:2])
    if scale != 1.0:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
    pad = max(16, int(0.08 * max(gray.shape[:2])))
    gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return cv2.GaussianBlur(gray, (3, 3), 0), scale, pad


def _points_to_bbox(pts: np.ndarray, scale: float, pad: int) -> BBox:
    xs, ys = (pts[:, 0] - pad) / scale, (pts[:, 1] - pad) / scale
    return BBox(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def read_barcodes(image: np.ndarray) -> list[BarcodeRead]:
    """크롭 이미지에서 소매용 바코드를 모두 읽는다. 못 읽으면 빈 목록.

    Args:
        image: BGR 또는 grayscale. ``pointing`` 이 넘겨준 타깃 크롭.

    Returns:
        체크섬을 통과한 바코드만, 중복 없이. 좌표는 입력 이미지 기준.
    """
    if image is None or image.size == 0:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    found: dict[str, BarcodeRead] = {}
    for long_side in _TARGET_LONG_SIDES:
        img, scale, pad = _prepare(gray, long_side)
        ok, infos, types, points = _get_detector().detectAndDecodeWithType(img)
        if not ok:
            continue
        for i, (code, kind) in enumerate(zip(infos, types, strict=False)):
            code = (code or "").strip()
            if kind not in _RETAIL_TYPES or code in found or not is_valid_gtin(code):
                continue
            bbox = _points_to_bbox(points[i], scale, pad) if points is not None else None
            found[code] = BarcodeRead(code=code, symbology=kind, bbox=bbox)
        if found:
            break  # 한 크기에서 읽혔으면 나머지는 볼 필요 없다
    return list(found.values())
