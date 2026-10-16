"""바코드 → 제품 조회 테스트.

실제 네트워크·카메라 없이 돈다.
- 바코드 이미지는 EAN-13 규격대로 직접 그린다(사진 파일을 커밋하지 않으려고).
- 식품안전나라 API 는 httpx.MockTransport 로 가짜 응답을 준다.
- 캐시는 SQLite 인메모리 DB 를 쓴다.

실행: pytest tests/test_nutrition_barcode.py
"""

import httpx
import numpy as np
import pytest

from jarviseo.memory.store import MemoryStore
from jarviseo.nutrition.barcode import (
    MfdsClient,
    MfdsError,
    OffClient,
    is_valid_gtin,
    lookup_product,
    read_barcodes,
)
from jarviseo.types import IngredientSource

# 체크섬이 맞는 실제 형식의 번호 (880 = 대한민국)
CODE = "8801234567893"

# --- EAN-13 그리기 ---------------------------------------------------------
_L = ["0001101", "0011001", "0010011", "0111101", "0100011",
      "0110001", "0101111", "0111011", "0110111", "0001011"]  # fmt: skip
_R = ["".join("1" if b == "0" else "0" for b in p) for p in _L]
_G = [p[::-1] for p in _R]
_PARITY = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
           "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]  # fmt: skip


def draw_ean13(code: str, module: int = 4, height: int = 160) -> np.ndarray:
    """흰 바탕에 EAN-13 막대를 그린 BGR 이미지."""
    d = [int(c) for c in code]
    left = "".join((_L if p == "L" else _G)[x] for p, x in zip(_PARITY[d[0]], d[1:7], strict=True))
    bits = "101" + left + "01010" + "".join(_R[x] for x in d[7:]) + "101"
    quiet = 12 * module
    width = len(bits) * module + 2 * quiet
    img = np.full((height + 40, width), 255, np.uint8)
    for i, b in enumerate(bits):
        if b == "1":
            x = quiet + i * module
            img[20 : 20 + height, x : x + module] = 0
    return np.dstack([img] * 3)


# --- 가짜 식품안전나라 ------------------------------------------------------


def fake_mfds(rows_by_service: dict[str, list[dict]], calls: list[str] | None = None):
    """서비스 이름별로 정해 둔 row 를 돌려주는 가짜 API."""

    def handler(request: httpx.Request) -> httpx.Response:
        service = request.url.path.split("/")[-5]  # .../{key}/{서비스}/json/1/{끝}/{조건}
        if calls is not None:
            calls.append(service)
        rows = rows_by_service.get(service, [])
        body = {"total_count": str(len(rows)), "RESULT": {"CODE": "INFO-000"}}
        if rows:
            body["row"] = rows
        else:
            body["RESULT"] = {"CODE": "INFO-200", "MSG": "해당하는 데이터가 없습니다."}
        return httpx.Response(200, json={service: body})

    return MfdsClient(api_key="test", transport=httpx.MockTransport(handler))


C005_ROW = {"BAR_CD": CODE, "PRDLST_NM": "테스트 과자", "PRDLST_REPORT_NO": "1999001"}
C002_ROWS = [{"RAWMTRL_NM": "밀가루"}, {"RAWMTRL_NM": "탈지분유"}, {"RAWMTRL_NM": "설탕"}]


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(database_url="sqlite://")
    s.init_schema()
    return s


# --- 체크섬 ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "ok"),
    [
        (CODE, True),
        ("8801234567890", False),  # 마지막 자리만 틀림
        ("96385074", True),  # EAN-8
        ("036000291452", True),  # UPC-A
        ("88012345678a3", False),
        ("12345", False),
    ],
)
def test_gtin_체크섬(code, ok):
    assert is_valid_gtin(code) is ok


# --- 이미지에서 읽기 --------------------------------------------------------


def test_그린_바코드를_읽는다():
    reads = read_barcodes(draw_ean13(CODE))
    assert [r.code for r in reads] == [CODE]
    assert reads[0].symbology == "EAN_13"
    assert reads[0].bbox is not None


@pytest.mark.parametrize(("module", "height"), [(1, 40), (3, 150), (6, 300), (12, 600)])
def test_막대_굵기와_상관없이_읽는다(module, height):
    # 크롭이 바코드에 딱 맞게 잘려 막대가 아주 가늘거나 굵어도 읽어야 한다.
    reads = read_barcodes(draw_ean13(CODE, module=module, height=height))
    assert [r.code for r in reads] == [CODE]


def test_큰_크롭_속_작은_바코드도_읽는다():
    scene = np.full((900, 1200, 3), 200, np.uint8)
    bc = draw_ean13(CODE, module=3, height=150)
    scene[300 : 300 + bc.shape[0], 400 : 400 + bc.shape[1]] = bc
    reads = read_barcodes(scene)
    assert [r.code for r in reads] == [CODE]
    # 좌표는 원본 크롭 기준이어야 한다 (여백·배율을 되돌렸는지)
    box = reads[0].bbox
    assert 380 <= box.x1 <= 480 and 280 <= box.y1 <= 340


def test_바코드가_없으면_빈_목록():
    assert read_barcodes(np.full((300, 400, 3), 255, np.uint8)) == []
    assert read_barcodes(np.zeros((0, 0, 3), np.uint8)) == []


# --- API 클라이언트 ---------------------------------------------------------


def test_api_바코드로_원재료까지_가져온다():
    with fake_mfds({"C005": [C005_ROW], "C002": C002_ROWS}) as client:
        info = client.lookup(CODE)
    assert info is not None
    assert info.product_name == "테스트 과자"
    assert info.report_no == "1999001"
    assert info.raw_ingredients == "밀가루, 탈지분유, 설탕"
    assert info.source is IngredientSource.API


def test_api_c002에_없으면_c006을_본다():
    rows = {"C005": [C005_ROW], "C006": [{"RAWMTRL_NM": "돼지고기"}]}
    with fake_mfds(rows) as client:
        assert client.lookup(CODE).raw_ingredients == "돼지고기"


def test_api_등록이_없으면_none():
    with fake_mfds({}) as client:
        assert client.lookup(CODE) is None


def test_api_에러코드는_예외():
    def handler(request):
        return httpx.Response(200, json={"C005": {"RESULT": {"CODE": "ERROR-300"}}})

    with MfdsClient(api_key="bad", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MfdsError):
            client.lookup(CODE)


# --- 캐시 우선 조회 ----------------------------------------------------------


def test_첫_조회는_api_두번째는_캐시(store):
    calls: list[str] = []
    image = draw_ean13(CODE)
    with fake_mfds({"C005": [C005_ROW], "C002": C002_ROWS}, calls) as client:
        first = lookup_product(image, store=store, client=client)
        n_calls = len(calls)
        second = lookup_product(image, store=store, client=client)

    assert first.hit and first.product.source is IngredientSource.API
    assert second.hit and second.product.source is IngredientSource.CACHE
    assert len(calls) == n_calls  # 두 번째는 API 를 부르지 않았다
    assert second.product.raw_ingredients == "밀가루, 탈지분유, 설탕"


def test_api_장애여도_예외없이_ocr로_넘긴다(store):
    def handler(request):
        raise httpx.ConnectTimeout("timeout", request=request)

    with MfdsClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
        result = lookup_product(draw_ean13(CODE), store=store, client=client)
    assert not result.hit
    assert result.reason == "api_error"
    assert result.barcode.code == CODE  # OCR 결과를 이 번호로 저장할 수 있게 남긴다


def test_원재료가_비면_미스(store):
    with fake_mfds({"C005": [C005_ROW]}) as client:
        result = lookup_product(draw_ean13(CODE), store=store, client=client)
    assert not result.hit and result.reason == "no_ingredients"
    assert store.get_product(CODE) is None  # 빈 정보는 캐시에 남기지 않는다


def test_바코드가_없으면_바로_미스():
    result = lookup_product(np.full((300, 400, 3), 255, np.uint8))
    assert not result.hit and result.reason == "no_barcode" and result.barcode is None


def test_invalid_캐시는_무시한다(store):
    with fake_mfds({"C005": [C005_ROW], "C002": C002_ROWS}) as client:
        info = client.lookup(CODE)
    store.save_product(info, status="invalid")
    assert store.get_product(CODE) is None


# --- Open Food Facts (#74) -------------------------------------------------


def fake_off(product: dict | None, calls: list[str] | None = None, status: int = 200):
    """product 가 None 이면 '등록 없음' 응답(status 0)을 준다."""

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url.path)
        if status != 200:
            return httpx.Response(status, json={"status": 0})
        if product is None:
            return httpx.Response(200, json={"status": 0, "status_verbose": "product not found"})
        return httpx.Response(200, json={"status": 1, "code": CODE, "product": product})

    return OffClient(transport=httpx.MockTransport(handler))


OFF_PRODUCT = {
    "product_name": "Test Snack",
    "product_name_ko": "테스트 스낵",
    "ingredients_text_ko": "밀가루, 설탕, 탈지분유, 대두유",
    "allergens_tags": ["en:gluten", "en:milk", "en:soybeans"],
    "traces_tags": ["en:crustaceans", "ko:땅콩"],
}


def test_off_한글_원재료와_알레르기_태그를_옮긴다():
    with fake_off(OFF_PRODUCT) as off:
        info = off.lookup(CODE)
    assert info.product_name == "테스트 스낵"
    assert info.raw_ingredients == "밀가루, 설탕, 탈지분유, 대두유"
    assert info.allergen_notice == "밀, 우유, 대두"
    assert info.cross_contamination == "게, 새우, 땅콩"
    assert info.source is IngredientSource.API


@pytest.mark.parametrize("status", [200, 404])
def test_off_등록이_없으면_none(status):
    with fake_off(None, status=status) as off:
        assert off.lookup(CODE) is None


def test_off_영어_원재료만_있으면_ocr로_넘긴다(store):
    english = {"product_name": "Snack", "ingredients_text": "wheat flour, sugar, milk powder"}
    with fake_off(english) as off:
        assert off.lookup(CODE).raw_ingredients == ""
        result = lookup_product(draw_ean13(CODE), store=store, off_client=off)
    assert not result.hit and result.reason == "no_ingredients"
    assert store.get_product(CODE) is None


def test_식품안전나라에_없으면_off에서_찾아_캐시한다(store):
    off_calls: list[str] = []
    image = draw_ean13(CODE)
    with fake_mfds({}) as client, fake_off(OFF_PRODUCT, off_calls) as off:
        first = lookup_product(image, store=store, client=client, off_client=off)
        second = lookup_product(image, store=store, client=client, off_client=off)
    assert first.hit and first.product.product_name == "테스트 스낵"
    assert second.hit and second.product.source is IngredientSource.CACHE
    assert len(off_calls) == 1  # 두 번째는 캐시에서 끝났다


def test_식품안전나라에_있으면_off를_부르지_않는다(store):
    off_calls: list[str] = []
    with (
        fake_mfds({"C005": [C005_ROW], "C002": C002_ROWS}) as client,
        fake_off(OFF_PRODUCT, off_calls) as off,
    ):
        result = lookup_product(draw_ean13(CODE), store=store, client=client, off_client=off)
    assert result.hit and result.product.product_name == "테스트 과자"
    assert off_calls == []


def test_off_장애여도_예외없이_ocr로_넘긴다(store):
    def handler(request):
        raise httpx.ConnectTimeout("timeout", request=request)

    with fake_mfds({}) as client, OffClient(transport=httpx.MockTransport(handler)) as off:
        result = lookup_product(draw_ean13(CODE), store=store, client=client, off_client=off)
    assert not result.hit and result.reason == "api_error"
    assert result.barcode.code == CODE


def test_off_서버오류도_예외없이_넘긴다(store):
    with fake_off(OFF_PRODUCT, status=500) as off:
        result = lookup_product(draw_ean13(CODE), store=store, off_client=off)
    assert not result.hit and result.reason == "api_error"
