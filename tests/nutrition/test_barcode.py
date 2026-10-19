"""바코드 → 제품 조회 테스트.

실제 네트워크·카메라 없이 돈다.
- 바코드 이미지는 EAN-13 규격대로 직접 그린다(사진 파일을 커밋하지 않으려고).
- 푸드QR API 는 httpx.MockTransport 로 가짜 응답을 준다.
- 캐시는 SQLite 인메모리 DB 를 쓴다.

실행: pytest tests/test_nutrition_barcode.py
"""

import httpx
import numpy as np
import pytest

from jarviseo.memory.store import MemoryStore
from jarviseo.nutrition.barcode import (
    NO_ALLERGEN_NOTICE,
    FoodQrClient,
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
def test_gtin_checksum(code, ok):
    """GTIN 체크섬."""
    assert is_valid_gtin(code) is ok


# --- 이미지에서 읽기 --------------------------------------------------------


def test_reads_drawn_barcode():
    """그린 바코드를 읽는다."""
    reads = read_barcodes(draw_ean13(CODE))
    assert [r.code for r in reads] == [CODE]
    assert reads[0].symbology == "EAN_13"
    assert reads[0].bbox is not None


@pytest.mark.parametrize(("module", "height"), [(1, 40), (3, 150), (6, 300), (12, 600)])
def test_reads_regardless_of_bar_width(module, height):
    """막대 굵기와 상관없이 읽는다."""
    # 크롭이 바코드에 딱 맞게 잘려 막대가 아주 가늘거나 굵어도 읽어야 한다.
    reads = read_barcodes(draw_ean13(CODE, module=module, height=height))
    assert [r.code for r in reads] == [CODE]


def test_reads_small_barcode_in_large_crop():
    """큰 크롭 속 작은 바코드도 읽는다."""
    scene = np.full((900, 1200, 3), 200, np.uint8)
    bc = draw_ean13(CODE, module=3, height=150)
    scene[300 : 300 + bc.shape[0], 400 : 400 + bc.shape[1]] = bc
    reads = read_barcodes(scene)
    assert [r.code for r in reads] == [CODE]
    # 좌표는 원본 크롭 기준이어야 한다 (여백·배율을 되돌렸는지)
    box = reads[0].bbox
    assert 380 <= box.x1 <= 480 and 280 <= box.y1 <= 340


def test_no_barcode_returns_empty_list():
    """바코드가 없으면 빈 목록이다."""
    assert read_barcodes(np.full((300, 400, 3), 255, np.uint8)) == []
    assert read_barcodes(np.zeros((0, 0, 3), np.uint8)) == []


# --- 푸드QR ---------------------------------------------------------------

QR_CODE = "0" + CODE  # 푸드QR 은 14자리로 등록돼 있다
NAME = "오늘밥상 파채양념"


def qr_row(name: str, code: str = QR_CODE) -> dict:
    """foodqr.kr qr1009/F009 실제 응답 한 줄 모양."""
    return {"algCsgMtrNm": name, "brcdNo": code, "prdctNm": NAME, "verNo": 1}


def qr_text(field: str, text: str) -> dict:
    return {field: text, "brcdNo": QR_CODE, "prdctNm": NAME}


def fake_qr(
    allergy: list[dict],
    calls: list | None = None,
    status: int = 200,
    code: str = "0000",
    caution: list[dict] | None = None,  # 주의사항 F004
    raw: list[dict] | None = None,  # 원재료 F007
    nutrition: list[dict] | None = None,  # 영양표시 F008 (등록 여부 확인)
    denied: tuple[str, ...] = (),  # 권한이 없어 401 을 줄 서비스
):
    by_service = {"F009": allergy, "F004": caution, "F007": raw, "F008": nutrition}

    def handler(request: httpx.Request) -> httpx.Response:
        service = request.url.path.rsplit("/", 1)[-1]
        if calls is not None:
            calls.append((service, dict(request.url.params)))
        if status != 200 or service in denied:
            return httpx.Response(status if status != 200 else 401, json={"response": {}})
        rows = by_service.get(service) or []
        items = {"item": rows[0] if len(rows) == 1 else rows} if rows else ""
        body = {"header": {"resultCode": code, "resultMsg": "OK"}, "body": {"items": items}}
        return httpx.Response(200, json={"response": body})

    return FoodQrClient(api_key="test", transport=httpx.MockTransport(handler))


# 실제 응답에서 가져온 모양 (육개장사발면·파채양념)
CAUTION = [
    qr_text(
        "atentMterCn",
        "메밀,땅콩,새우,조개류(굴,전복,홍합포함),잣을사용한제품과같은&nbsp;시설에서제조하고있습니다.\n",
    ),
    qr_text("atentMterCn", "직사광선을피하여실온에보관하십시오.\n"),
]
RAW = [
    qr_text("prvwCn", '<a href="https://terms.naver.com/search.naver?query=설탕" class="link">'
                      "<span>설탕</span></a>,\xa0\xa0소맥분(밀가루: 미국산), 간장"),
]  # fmt: skip


def test_foodqr_copies_allergens_cross_contact_and_ingredients():
    """푸드QR 의 알레르기 · 혼입 · 원재료를 모두 옮긴다."""
    calls: list = []
    rows = [qr_row(n) for n in ("대두", "밀", "조개류(굴)", "대두")]
    with fake_qr(rows, calls, caution=CAUTION, raw=RAW) as qr:
        info = qr.lookup(CODE)
    assert info.product_name == NAME
    assert info.allergen_notice == "대두, 밀, 조개류(굴)"  # 같은 이름은 한 번만
    assert info.cross_contamination.startswith("메밀,땅콩,새우")  # 보관 방법 줄은 뺀다
    assert "보관" not in info.cross_contamination and "&nbsp;" not in info.cross_contamination
    assert info.raw_ingredients == "설탕, 소맥분(밀가루: 미국산), 간장"  # HTML 을 걷어 낸다
    assert {c[0] for c in calls} == {"F009", "F004", "F007"}  # 셋을 함께 묻고 F008 은 안 묻는다
    assert all(c[1]["brcdNo"] == QR_CODE and c[1]["accessKey"] == "test" for c in calls)


def test_foodqr_answers_with_allergens_without_caution_permission():
    """푸드QR 주의사항 · 원재료 권한이 없어도 알레르기로 답한다."""
    with fake_qr([qr_row("밀")], denied=("F004", "F007")) as qr:
        info = qr.lookup(CODE)
    assert info.allergen_notice == "밀" and info.raw_ingredients == "밀"
    assert info.cross_contamination == ""


def test_foodqr_accepts_single_dict():
    """푸드QR 결과가 한 건이면 dict 로 와도 받는다."""
    with fake_qr([qr_row("우유")]) as qr:
        assert qr.lookup(CODE).allergen_notice == "우유"


@pytest.mark.parametrize("rows", [[], [{}]])
def test_foodqr_returns_none_when_not_registered(rows):
    """푸드QR 에 등록이 없으면 None 이다."""
    with fake_qr(rows) as qr:
        assert qr.lookup(CODE) is None


def test_foodqr_registered_without_allergen_label():
    """푸드QR 에 등록됐지만 알레르기 표시가 없는 제품."""
    calls: list = []
    nutrition = [{"brcdNo": QR_CODE, "prdctNm": "생수", "cta": 0}]
    with fake_qr([], calls, nutrition=nutrition) as qr:
        info = qr.lookup(CODE)
    assert info.product_name == "생수"
    assert info.allergen_notice == NO_ALLERGEN_NOTICE  # 비우지 않는다 → OCR 로 넘어가지 않는다
    assert {c[0] for c in calls} == {"F009", "F004", "F007", "F008"}


def test_foodqr_ingredients_only_counts_as_registered():
    """푸드QR 에 원재료만 있어도 등록된 제품이다."""
    calls: list = []
    with fake_qr([], calls, raw=RAW) as qr:
        info = qr.lookup(CODE)
    assert info.allergen_notice == NO_ALLERGEN_NOTICE and "소맥분" in info.raw_ingredients
    assert "F008" not in {c[0] for c in calls}


def test_foodqr_drops_other_barcodes():
    """푸드QR 결과의 다른 바코드는 버린다."""
    with fake_qr([qr_row("땅콩", code="08800000000000")]) as qr:
        assert qr.lookup(CODE) is None


# --- 캐시 우선 조회 ----------------------------------------------------------


def test_first_lookup_api_second_cache(store):
    """첫 조회는 푸드QR, 두 번째는 캐시다."""
    calls: list = []
    image = draw_ean13(CODE)
    with fake_qr([qr_row("밀")], calls, caution=CAUTION, raw=RAW) as qr:
        first = lookup_product(image, store=store, food_qr=qr)
        n_calls = len(calls)
        second = lookup_product(image, store=store, food_qr=qr)

    assert first.hit and first.product.source is IngredientSource.API
    assert second.hit and second.product.source is IngredientSource.CACHE
    assert len(calls) == n_calls  # 두 번째는 API 를 부르지 않았다
    assert second.product.allergen_notice == "밀"
    assert "새우" in second.product.cross_contamination  # 혼입도 캐시에 남는다


def test_api_outage_falls_back_to_ocr(store):
    """푸드QR 장애여도 예외 없이 OCR 로 넘긴다."""

    def handler(request):
        raise httpx.ConnectTimeout("timeout", request=request)

    with FoodQrClient(api_key="x", transport=httpx.MockTransport(handler)) as qr:
        result = lookup_product(draw_ean13(CODE), store=store, food_qr=qr)
    assert not result.hit
    assert result.reason == "api_error"
    assert result.barcode.code == CODE  # OCR 결과를 이 번호로 저장할 수 있게 남긴다


@pytest.mark.parametrize(("status", "code"), [(401, "0000"), (200, "9999")])
def test_foodqr_error_falls_back_to_ocr(store, status, code):
    """푸드QR 이 오류 응답이면 OCR 로 넘어간다."""
    with fake_qr([], status=status, code=code) as qr:
        result = lookup_product(draw_ean13(CODE), store=store, food_qr=qr)
    assert not result.hit and result.reason == "api_error"


def test_not_registered_is_miss(store):
    """푸드QR 에 없는 제품은 미스다. 캐시에도 남기지 않는다."""
    with fake_qr([]) as qr:
        result = lookup_product(draw_ean13(CODE), store=store, food_qr=qr)
    assert not result.hit and result.reason == "not_found"
    assert store.get_product(CODE) is None


def test_no_barcode_is_immediate_miss():
    """바코드가 없으면 바로 미스다."""
    result = lookup_product(np.full((300, 400, 3), 255, np.uint8))
    assert not result.hit and result.reason == "no_barcode" and result.barcode is None


def test_invalid_cache_is_ignored(store):
    """`invalid` 캐시는 무시한다."""
    with fake_qr([qr_row("밀")]) as qr:
        info = qr.lookup(CODE)
    store.save_product(info, status="invalid")
    assert store.get_product(CODE) is None
