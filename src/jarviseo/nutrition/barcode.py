"""바코드 → 제품 정보 조회.  (Barcode Lookup Agent)

흐름: 크롭 이미지 → 바코드 번호 읽기 → product 캐시 → 푸드QR → 캐시 저장
      여기서 원재료명을 못 얻으면 호출한 쪽이 성분표 OCR 로 넘어간다.

바코드 읽기에 pyzbar 대신 OpenCV 내장 검출기를 쓰는 이유
-----------------------------------------------------
pyzbar 는 윈도우에서 Visual C++ 2013 런타임이 없으면 DLL 로드가 실패한다.
팀원 노트북마다 설치 여부가 달라서 "내 PC 에서만 안 됨"이 생긴다.
OpenCV(cv2.barcode)는 이미 의존성에 있고 EAN-13/EAN-8/UPC 를 읽는다.

LLM 을 쓰지 않는다. 바코드 번호는 체크섬으로 검증하고, 제품 정보는 공공 API 그대로다.
"""

from __future__ import annotations

import html
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import httpx
import numpy as np

from jarviseo import config
from jarviseo.types import BarcodeRead, BBox, IngredientSource, ProductInfo

__all__ = [
    "NO_ALLERGEN_NOTICE",
    "BarcodeLookup",
    "FoodQrClient",
    "FoodQrError",
    "is_valid_gtin",
    "find_gtin",
    "lookup_code",
    "lookup_product",
    "make_food_qr_client",
    "read_barcodes",
]

log = logging.getLogger(__name__)

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
    if not found:
        # cv2.barcode 는 작고 기울고 휘거나 반사광이 지나간 바코드를 자주 놓친다.
        # zxing-cpp 로 한 번 더 본다.
        # 웹캠 실측: 손에 든 크런키 바코드를 cv2 는 못 읽고 zxing 은 읽었다.
        for read in _read_zxing(image):
            found.setdefault(read.code, read)
    return list(found.values())


# zxing-cpp 의 형식 이름 → 이 모듈의 이름(cv2.barcode 와 같게)
_ZXING_TYPES = {"EAN13": "EAN_13", "EAN8": "EAN_8", "UPCA": "UPC_A", "UPCE": "UPC_E"}


def _read_zxing(image: np.ndarray) -> list[BarcodeRead]:
    """zxing-cpp 로 소매용 바코드를 읽는다. 설치돼 있지 않으면 빈 목록(cv2 결과만 쓴다)."""
    try:
        import zxingcpp
    except ImportError:
        return []
    out = []
    for r in zxingcpp.read_barcodes(image):
        kind = _ZXING_TYPES.get(str(r.format).split(".")[-1].replace("-", "").upper())
        code = (r.text or "").strip()
        if kind is None or not is_valid_gtin(code):
            continue
        pos = r.position
        pts = [pos.top_left, pos.top_right, pos.bottom_right, pos.bottom_left]
        xs, ys = [float(q.x) for q in pts], [float(q.y) for q in pts]
        out.append(
            BarcodeRead(code=code, symbology=kind, bbox=BBox(min(xs), min(ys), max(xs), max(ys)))
        )
    return out


# --------------------------------------------------------------------------
# 2. 푸드QR 알레르기정보 (식약처, portal.foodqr.kr)
# --------------------------------------------------------------------------


# 푸드QR 에 등록됐지만 알레르기 표시가 없는 제품. 판정 쪽은 이 칸이 비었는지만 본다.
NO_ALLERGEN_NOTICE = "알레르기 유발물질 표시 없음"


class FoodQrError(RuntimeError):
    """푸드QR 이 정상 응답 코드가 아닌 것을 돌려줬다(인증 실패·트래픽 초과 등)."""


class FoodQrClient:
    """푸드QR 조회. 포장 표시 내용을 항목별로 받아 ProductInfo 하나로 묶는다.

    - 알레르기정보(qr1009/F009): 표시된 알레르기 유발물질 ("대두", "밀", "조개류(굴)")
    - 주의사항(qr1004/F004): 혼입 문구 ("메밀, 땅콩, … 을 사용한 제품과 같은 시설에서 제조")
    - 원재료(qr1007/F007): 원재료 표시 전체 (HTML 링크가 섞여 와서 글자만 남긴다)
    - 영양표시(qr1008/F008): 위가 모두 비었을 때 등록 여부만 본다

    원재료가 상위 분류('면류, 분말스프')로 뭉개지지 않아서 판정 근거로 바로 쓴다.
    바코드는 앞에 0을 붙인 14자리(GTIN-14)로 등록돼 있다("08801045121086").
    서비스마다 인증키 권한이 따로라, 주의사항·원재료가 401 이면 그 칸만 비우고 계속한다.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,  # 테스트에서 가짜 응답을 끼운다
    ) -> None:
        self.api_key = api_key or config.FOOD_QR_API_KEY
        self.base_url = (base_url or config.FOOD_QR_BASE_URL).rstrip("/")
        self._http = httpx.Client(
            timeout=timeout or config.FOOD_QR_TIMEOUT_SEC, transport=transport
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> FoodQrClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _items(self, path: str, code: str) -> list[dict]:
        resp = self._http.get(
            f"{self.base_url}/{path}",
            params={"accessKey": self.api_key, "_type": "json", "numOfRows": 100, "brcdNo": code},
        )
        resp.raise_for_status()  # 401: 권한 없는 키
        data = resp.json().get("response") or {}
        header = data.get("header") or {}
        if str(header.get("resultCode")) != "0000":
            raise FoodQrError(f"{path}: {header.get('resultCode')} {header.get('resultMsg')}")
        items = (data.get("body") or {}).get("items") or []
        if isinstance(items, dict):
            items = items.get("item") or []
        if isinstance(items, dict):  # 한 건이면 목록이 아니라 dict 로 온다
            items = [items]
        # 다른 바코드가 섞여 오지 않게 한 번 더 거른다(검색 조건을 무시하는 경우 대비).
        return [i for i in items if str(i.get("brcdNo", "")).zfill(14) == code]

    def _optional(self, path: str, code: str) -> list[dict]:
        """권한이 없거나(401) 서버가 오류를 내도 빈 목록. 있으면 좋은 정보라서다."""
        try:
            return self._items(path, code)
        except (httpx.HTTPError, FoodQrError, ValueError) as e:
            log.info("푸드QR %s 를 건너뛴다: %s", path, e)
            return []

    def lookup(self, barcode: str) -> ProductInfo | None:
        """바코드 → 제품 정보. 푸드QR 에 등록이 없는 제품이면 None.

        알레르기·주의사항·원재료가 모두 비면 영양표시(F008)로 등록 여부를 한 번 더 본다.
        등록돼 있으면 '알레르기 표시가 없는 제품'이다. 이걸 미등록과 구별하지 않으면
        성분표 OCR 까지 돌게 된다(사진당 수 초~수십 초).
        """
        code = barcode.zfill(14)
        # 세 서비스를 한꺼번에 묻는다. 차례로 물으면 기다리는 시간이 세 배가 된다.
        with ThreadPoolExecutor(max_workers=3) as pool:
            allergy = pool.submit(self._items, "qr1009/F009", code)  # 알레르기는 실패하면 예외
            caution = pool.submit(self._optional, "qr1004/F004", code)
            raw = pool.submit(self._optional, "qr1007/F007", code)
            allergy, caution, raw = allergy.result(), caution.result(), raw.result()

        names: list[str] = []
        for item in allergy:
            name = str(item.get("algCsgMtrNm") or "").strip()
            if name and name not in names:
                names.append(name)
        cross = [
            t for t in (_qr_text(i.get("atentMterCn")) for i in caution) if _QR_CROSS.search(t)
        ]
        text = ", ".join(t for t in (_qr_text(i.get("prvwCn")) for i in raw) if t)

        found = allergy or caution or raw
        if not names and not cross and not text:
            found = self._items("qr1008/F008", code)
            if not found:
                return None
        notice = ", ".join(names) or NO_ALLERGEN_NOTICE
        return ProductInfo(
            barcode=barcode,
            product_name=str(found[0].get("prdctNm") or "").strip(),
            # 원재료를 못 받으면 알레르기 목록을 원재료 칸에도 둔다(비면 미스로 본다).
            raw_ingredients=text or notice,
            allergen_notice=notice,
            cross_contamination=" ".join(cross),
            source=IngredientSource.API,
        )


# 주의사항 중 혼입 문구. "~혼입가능", "~을 사용한 제품과 같은 시설에서 제조"
_QR_CROSS = re.compile(r"혼\s*입|같은\s*(?:제조)?\s*시\s*설|동일\s*(?:제조)?\s*시\s*설")


def _qr_text(value) -> str:
    """푸드QR 글자 칸 정리. 원재료 칸에는 사전 링크 HTML 이, 주의사항 칸에는 &nbsp; 가 섞여 온다."""
    text = html.unescape(re.sub(r"<[^>]+>", "", str(value or "")))
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def make_food_qr_client() -> FoodQrClient | None:
    """인증키(``config.FOOD_QR_API_KEY``)가 있으면 FoodQrClient, 없으면 None."""
    return FoodQrClient() if config.FOOD_QR_API_KEY else None


# --------------------------------------------------------------------------
# 3. 캐시 우선 조회 (Barcode Lookup Agent 본체)
# --------------------------------------------------------------------------


@dataclass
class BarcodeLookup:
    """조회 결과. ``hit`` 이 False 면 성분표 OCR 로 넘어간다.

    바코드는 읽었는데 제품을 못 찾은 경우에도 ``barcode`` 는 채워 둔다.
    OCR 로 읽은 성분을 그 바코드로 캐시에 남기기 위해서다.
    """

    barcode: BarcodeRead | None
    product: ProductInfo | None
    elapsed_ms: int = 0
    reason: str = ""  # 미스 사유: no_barcode / not_found / no_ingredients / api_error

    @property
    def hit(self) -> bool:
        return self.product is not None and bool(self.product.raw_ingredients)


def lookup_product(
    image: np.ndarray,
    store=None,
    food_qr: FoodQrClient | None = None,
) -> BarcodeLookup:
    """크롭 이미지 → 제품 정보. 캐시 → 푸드QR 순서로 찾는다.

    Args:
        image: 타깃 크롭 (BGR).
        store: ``MemoryStore``. 없으면 캐시를 건너뛴다(테스트·오프라인).
        food_qr: ``FoodQrClient``. 없으면(인증키 없음) 푸드QR 을 건너뛴다.

    외부 API 장애는 예외로 올리지 않는다. 사용자는 OCR 로 답을 받을 수 있어야 한다.
    """
    t0 = time.perf_counter()

    def done(
        read: BarcodeRead | None, product: ProductInfo | None, reason: str = ""
    ) -> BarcodeLookup:
        ms = int((time.perf_counter() - t0) * 1000)
        return BarcodeLookup(barcode=read, product=product, elapsed_ms=ms, reason=reason)

    reads = read_barcodes(image)
    if not reads:
        return done(None, None, "no_barcode")
    return _lookup_reads(reads, store, food_qr, done)


# 바코드 아래 찍힌 숫자. OCR 이 사이에 공백 · '-' · ':' · '.' 을 넣어 읽는다("8:801062-640782").
_GTIN_TEXT = re.compile(r"(?<!\d)\d(?:[\s\-:.]?\d){12}(?!\d)")
# 한국 상품 바코드 국가 코드. 같은 글자에 숫자 덩어리가 여럿이면 이쪽을 먼저 본다.
_KOREA_PREFIX = "880"


def find_gtin(text: str) -> str | None:
    """OCR 글자에서 EAN-13 바코드 번호를 찾는다. 체크섬이 맞는 것만. 없으면 None.

    바코드 줄무늬를 못 읽어도(작음 · 흐림) 그 아래 숫자는 OCR 이 읽는 일이 많다(크런키 실측:
    줄무늬는 cv2 · zxing 둘 다 못 읽고 OCR 은 "8:801062-640782" 로 읽음).
    품목보고번호 · 전화번호도 숫자라서 13자리 + 체크섬을 통과해야만 받는다.
    """
    codes = [re.sub(r"\D", "", m.group()) for m in _GTIN_TEXT.finditer(text or "")]
    codes = [c for c in codes if is_valid_gtin(c)]
    codes.sort(key=lambda c: not c.startswith(_KOREA_PREFIX))
    return codes[0] if codes else None


def lookup_code(
    code: str, store=None, food_qr: FoodQrClient | None = None, symbology: str = "TEXT"
) -> BarcodeLookup:
    """바코드 번호(성분표 글자에서 읽은 것 등)로 캐시 → 푸드QR 을 찾는다."""
    t0 = time.perf_counter()

    def done(
        read: BarcodeRead | None, product: ProductInfo | None, reason: str = ""
    ) -> BarcodeLookup:
        ms = int((time.perf_counter() - t0) * 1000)
        return BarcodeLookup(barcode=read, product=product, elapsed_ms=ms, reason=reason)

    return _lookup_reads([BarcodeRead(code=code, symbology=symbology)], store, food_qr, done)


def _lookup_reads(reads: list[BarcodeRead], store, food_qr, done) -> BarcodeLookup:
    """읽은 바코드들로 캐시 → 푸드QR 순서로 찾는다. 하나라도 찾으면 그것을 돌려준다."""
    reason = "not_found"
    for read in reads:
        if store is not None:
            cached = store.get_product(read.code)
            if cached is not None and cached.raw_ingredients:
                return done(read, cached)

        if food_qr is None:
            continue
        try:
            product = food_qr.lookup(read.code)
        except (httpx.HTTPError, FoodQrError, ValueError) as e:  # ValueError: JSON
            log.warning("푸드QR 조회 실패 (%s): %s", read.code, e)
            reason = "api_error"
            continue
        if product is None:
            continue
        if not product.raw_ingredients:
            reason = "no_ingredients"
            continue
        if store is not None:
            store.save_product(product, status="verified")
        return done(read, product)

    return done(reads[0], None, reason)
