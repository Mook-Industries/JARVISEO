"""성분 판정 전체 흐름 테스트. 바코드 조회와 OCR 엔진은 가짜로 바꾼다.

실행: pytest tests/test_nutrition_service.py
"""

import cv2
import numpy as np
import pytest

from jarviseo.memory.store import MemoryStore
from jarviseo.nutrition import service
from jarviseo.nutrition.barcode import BarcodeLookup
from jarviseo.nutrition.quality import ImageQuality
from jarviseo.nutrition.response import RETAKE_CLOSER, RETAKE_GLARE, RETAKE_STEADY, RETAKE_WHOLE
from jarviseo.nutrition.service import IngredientChecker
from jarviseo.types import AllergenVerdict as V
from jarviseo.types import BarcodeRead, IngredientSource, ProductInfo

CROP = np.full((120, 400, 3), 255, np.uint8)
CODE = "8801234567893"

LABEL = [
    "원재료명: 밀가루, 설탕, 탈지분유, 정제소금",
    "이 제품은 땅콩을 사용한 제품과 같은 제조시설에서 제조합니다.",
]


class FakeReader:
    """readtext 를 부를 때마다 passes 에서 하나씩 꺼내 돌려준다. (줄 글자들, 신뢰도)"""

    def __init__(self, *passes):
        self.passes = list(passes)
        self.calls = 0

    def readtext(self, image, **kwargs):
        texts, conf = self.passes[min(self.calls, len(self.passes) - 1)]
        self.calls += 1
        return [([[0, i * 30], [300, i * 30], [300, i * 30 + 20], [0, i * 30 + 20]], t, conf)
                for i, t in enumerate(texts)]  # fmt: skip


def miss(code: str | None = None, reason: str = "no_barcode"):
    read = BarcodeRead(code=code, symbology="EAN_13") if code else None
    return lambda image, store=None, client=None, food_qr=None: BarcodeLookup(
        read, None, reason=reason
    )


def checker(reader, **kw) -> IngredientChecker:
    # CROP 은 단색이라 품질 검사에 걸린다. 품질 검사는 아래에서 따로 본다.
    kw.setdefault("quality_check", None)
    # FakeReader 는 이미지와 상관없이 답하므로 회전 시도를 끈다. 회전은 아래에서 따로 본다.
    kw.setdefault("try_rotation", False)
    return IngredientChecker(reader=reader, conf_threshold=0.8, **kw)


# --- 바코드로 찾은 경우 ----------------------------------------------------------


def test_barcode_hit_judges_without_ocr(monkeypatch):
    """바코드로 찾으면 OCR 없이 판정한다."""
    info = ProductInfo(
        barcode=CODE,
        product_name="버터쿠키",
        raw_ingredients="밀가루, 버터",
        source=IngredientSource.API,
    )
    monkeypatch.setattr(
        service,
        "lookup_product",
        lambda image, store=None, client=None, food_qr=None: BarcodeLookup(None, info),
    )
    reader = FakeReader((LABEL, 0.9))
    r = checker(reader).check(CROP, ["우유"])
    assert reader.calls == 0
    assert r.judgement.verdict is V.CONTAINS and r.judgement.source is IngredientSource.API
    assert r.message.startswith("주의하세요. 버터쿠키에 우유가 들어 있어요.")
    assert r.ingredients == ["밀가루", "버터"] and r.retake is None
    assert set(r.timings_ms) == {"barcode", "match"}


# --- 성분표 OCR ----------------------------------------------------------------


def test_judges_after_single_ocr_read(monkeypatch):
    """OCR 한 번에 읽히면 바로 판정한다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = FakeReader((LABEL, 0.9))
    r = checker(reader).check(CROP, ["땅콩"])
    assert reader.calls == 1 and not r.reprocessed
    assert r.judgement.verdict is V.MAY_CONTAIN and r.judgement.source is IngredientSource.OCR
    assert "땅콩을 쓰는 시설" in r.message and r.retake is None
    assert r.ocr_conf == pytest.approx(0.9) and "원재료명" in r.ocr_text
    assert set(r.timings_ms) == {"barcode", "ocr", "match"}


def test_low_confidence_reprocesses_same_photo(monkeypatch):
    """신뢰도가 낮으면 같은 사진을 재처리한다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = FakeReader((LABEL, 0.4), (LABEL, 0.9))
    r = checker(reader).check(CROP, ["땅콩"])
    assert reader.calls == 2 and r.reprocessed
    assert r.ocr_conf == pytest.approx(0.9) and r.retake is None


def test_failed_reprocess_asks_closer_retake(monkeypatch):
    """재처리도 실패하면 더 가까이 재촬영을 요청한다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(FakeReader((LABEL, 0.4))).check(CROP, ["땅콩"])
    assert r.retake == RETAKE_CLOSER and r.message == RETAKE_CLOSER
    assert r.judgement.verdict is V.UNDETERMINED and r.judgement.reason == "low_ocr_conf"


def test_missing_header_asks_full_view_retake(monkeypatch):
    """원재료명이 안 보이면 전체가 보이게 재촬영을 요청한다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(FakeReader((["밀가루, 설탕"], 0.95))).check(CROP, ["땅콩"])
    assert r.retake == RETAKE_WHOLE and r.judgement.reason == "no_ingredients"


def test_retake_limit_ends_undetermined(monkeypatch):
    """재촬영 상한이면 확인 불가로 끝낸다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(FakeReader((LABEL, 0.4))).check(CROP, ["땅콩"], retake_count=2)
    assert r.retake is None and r.retake_count == 2
    assert r.judgement.verdict is V.UNDETERMINED
    assert "정확히 읽지 못했어요" in r.message


def test_partial_read_with_contains_warns_now(monkeypatch):
    """판독이 불완전해도 함유를 찾으면 바로 경고한다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(FakeReader((LABEL, 0.4))).check(CROP, ["우유"])
    assert r.judgement.verdict is V.CONTAINS and r.retake is None
    assert r.message.startswith("주의하세요.")


def test_reprocess_off_reads_once(monkeypatch):
    """재처리를 끄면 한 번만 읽는다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = FakeReader((LABEL, 0.4))
    checker(reader, max_reprocess=0).check(CROP, ["땅콩"])
    assert reader.calls == 1


# --- 캐시 -------------------------------------------------------------------


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(database_url="sqlite://")
    s.init_schema()
    return s


def test_unknown_barcode_caches_ocr_as_unverified(monkeypatch, store):
    """바코드는 읽었는데 못 찾은 제품은 OCR 결과를 확인 전으로 캐시한다."""
    monkeypatch.setattr(service, "lookup_product", miss(CODE, "not_found"))
    checker(FakeReader((LABEL, 0.9)), store=store).check(CROP, ["땅콩"])
    cached = store.get_product(CODE)
    assert cached is not None and "탈지분유" in cached.raw_ingredients
    assert "땅콩" in cached.cross_contamination


def test_uncertain_read_is_not_cached(monkeypatch, store):
    """판독이 불확실하면 캐시하지 않는다."""
    monkeypatch.setattr(service, "lookup_product", miss(CODE, "not_found"))
    checker(FakeReader((LABEL, 0.4)), store=store).check(CROP, ["우유"])
    assert store.get_product(CODE) is None


# --- OCR 전 품질 검사 ----------------------------------------------------------


def text_image() -> np.ndarray:
    img = np.full((240, 600, 3), 235, np.uint8)
    for i in range(8):
        cv2.putText(img, "Ingredients: wheat flour, sugar, milk, soy", (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)  # fmt: skip
    return img


def bad(reason: str):
    return lambda image: ImageQuality(sharpness=1.0, edge=1.0, glare=0.0, reason=reason)


def test_blurry_photo_asks_retake_without_ocr(monkeypatch):
    """흐린 사진은 OCR 없이 재촬영을 안내한다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = FakeReader((LABEL, 0.9))
    r = checker(reader, quality_check=bad("blurry")).check(CROP, ["땅콩"])
    assert reader.calls == 0  # 30초짜리 OCR 을 돌리지 않는다
    assert r.retake == RETAKE_STEADY and r.message == RETAKE_STEADY
    assert r.judgement.verdict is V.UNDETERMINED and r.judgement.reason == "blurry"
    assert "quality" in r.timings_ms and "ocr" not in r.timings_ms


def test_glare_asks_to_turn_package(monkeypatch):
    """반사가 심하면 포장을 돌려 달라고 한다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(FakeReader((LABEL, 0.9)), quality_check=bad("glare")).check(CROP, ["땅콩"])
    assert r.retake == RETAKE_GLARE


def test_over_retake_limit_ends_undetermined(monkeypatch):
    """재촬영 상한을 넘으면 확인 불가로 끝낸다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(FakeReader((LABEL, 0.9)), quality_check=bad("blurry")).check(
        CROP, ["땅콩"], retake_count=2
    )
    assert r.retake is None
    assert r.judgement.verdict is V.UNDETERMINED
    assert "흐려서" in r.message and "포장지" in r.message


def test_barcode_hit_judges_despite_bad_quality(monkeypatch):
    """바코드로 찾으면 품질이 나빠도 판정한다."""
    info = ProductInfo(barcode=CODE, product_name="쿠키", raw_ingredients="밀가루, 버터")
    monkeypatch.setattr(
        service,
        "lookup_product",
        lambda image, store=None, client=None, food_qr=None: BarcodeLookup(None, info),
    )
    r = checker(FakeReader((LABEL, 0.9)), quality_check=bad("blurry")).check(CROP, ["우유"])
    assert r.judgement.verdict is V.CONTAINS and r.quality is None


def test_multiple_frames_use_sharpest(monkeypatch):
    """여러 프레임이면 가장 선명한 것으로 확인한다."""
    seen = []

    def lookup(image, store=None, client=None, food_qr=None):
        seen.append(image)
        return BarcodeLookup(None, None, reason="no_barcode")

    monkeypatch.setattr(service, "lookup_product", lookup)
    sharp = text_image()
    frames = [cv2.GaussianBlur(sharp, (0, 0), 3), sharp, cv2.GaussianBlur(sharp, (0, 0), 1.5)]
    checker(FakeReader((LABEL, 0.9))).check_frames(frames, ["땅콩"])
    assert seen[0] is sharp


# --- 회전 보정 ----------------------------------------------------------------


class SidewaysReader:
    """세로로 긴 이미지(= 시계 방향으로 돌린 것)일 때만 성분표를 제대로 읽는 가짜 OCR."""

    def __init__(self):
        self.shapes = []

    def readtext(self, image, **kwargs):
        self.shapes.append(image.shape[:2])
        h, w = image.shape[:2]
        texts, conf = (LABEL, 0.9) if h > w else (["ㅁ ㅇ"], 0.3)
        return [([[0, i * 30], [300, i * 30], [300, i * 30 + 20], [0, i * 30 + 20]], t, conf)
                for i, t in enumerate(texts)]  # fmt: skip


def test_rotated_panel_is_read_upright(monkeypatch):
    """성분표가 돌아가 있으면 돌려서 읽는다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = SidewaysReader()
    r = checker(reader, try_rotation=True).check(CROP, ["땅콩"])  # CROP 은 가로로 긴 이미지
    assert r.rotation in (90, 270)
    assert r.judgement.verdict is V.MAY_CONTAIN and r.retake is None
    assert len(reader.shapes) == 3  # 그대로 1번 + 90°·270°


def test_upright_panel_is_not_rotated(monkeypatch):
    """정방향이면 돌리지 않는다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = FakeReader((LABEL, 0.9))
    r = checker(reader, try_rotation=True).check(CROP, ["땅콩"])
    assert r.rotation == 0 and reader.calls == 1  # 원재료명을 찾으면 추가 OCR 없음


# --- 조회 정보만으로 '미검출'을 말하지 않는다 ------------------------------------

# 식품안전나라 실제 응답(신라면): 원재료가 상위 분류 이름뿐이다.
RAMEN = ProductInfo(
    barcode=CODE,
    product_name="신라면",
    raw_ingredients="복합조미식품, 면류, 과.채가공품",
    source=IngredientSource.API,
)


def found(info: ProductInfo, code: str = CODE):
    read = BarcodeRead(code=code, symbology="EAN_13")
    return lambda image, **_: BarcodeLookup(read, info)  # 조회 인자가 늘어도 그대로 쓴다


def test_lookup_without_allergen_label_reads_panel(monkeypatch):
    """알레르기 표시 없는 조회 정보로는 미검출이라 하지 않고 성분표를 읽는다."""
    monkeypatch.setattr(service, "lookup_product", found(RAMEN))
    reader = FakeReader((["원재료명: 소맥분(밀), 대두유, 정제염"], 0.9))
    r = checker(reader).check(CROP, ["밀"])
    assert reader.calls == 1  # 성분표를 읽었다
    assert r.judgement.verdict is V.CONTAINS and r.judgement.source is IngredientSource.OCR
    assert r.judgement.product.product_name == "신라면"  # 조회한 제품명을 이어 쓴다


def test_unreadable_panel_asks_retake(monkeypatch):
    """성분표도 못 읽으면 재촬영을 요청한다."""
    monkeypatch.setattr(service, "lookup_product", found(RAMEN))
    r = checker(FakeReader((["설탕, 정제수"], 0.95))).check(CROP, ["밀"])  # 원재료명 머리말 없음
    assert r.retake == RETAKE_WHOLE
    assert r.judgement.verdict is V.UNDETERMINED


def test_lookup_with_allergen_label_answers_from_lookup(monkeypatch):
    """알레르기 표시가 있으면 조회 정보대로 답한다."""
    info = ProductInfo(barcode=CODE, product_name="사이다", raw_ingredients="정제수, 설탕",
                       allergen_notice="알레르기 유발물질 표시 없음")  # fmt: skip
    monkeypatch.setattr(service, "lookup_product", found(info))
    reader = FakeReader((LABEL, 0.9))
    r = checker(reader).check(CROP, ["밀"])
    assert reader.calls == 0 and r.judgement.verdict is V.NOT_DETECTED


def test_lookup_hit_warns_without_label(monkeypatch):
    """조회 정보에서 찾으면 표시가 없어도 바로 경고한다."""
    info = ProductInfo(barcode=CODE, product_name="쿠키", raw_ingredients="밀가루, 버터")
    monkeypatch.setattr(service, "lookup_product", found(info))
    reader = FakeReader((LABEL, 0.9))
    r = checker(reader).check(CROP, ["밀"])
    assert reader.calls == 0 and r.judgement.verdict is V.CONTAINS


def test_no_registered_allergen_skips_panel(monkeypatch):
    """등록한 알레르기가 없으면 성분표를 읽지 않는다."""
    monkeypatch.setattr(service, "lookup_product", found(RAMEN))
    reader = FakeReader((LABEL, 0.9))
    checker(reader).check(CROP, [])
    assert reader.calls == 0


def test_verified_product_skips_ocr_next_time(monkeypatch, store):
    """한 번 성분표로 확인한 제품은 다음에 OCR 을 다시 돌리지 않는다."""
    # 첫 번째: 바코드는 읽었지만 어디에도 없어서 성분표를 읽는다(알레르기 표시 없는 제품).
    monkeypatch.setattr(service, "lookup_product", miss(CODE, "not_found"))
    plain = ["원재료명: 정제수, 설탕, 구연산"]
    checker(FakeReader((plain, 0.95)), store=store).check(CROP, ["밀"])
    cached = store.get_product(CODE)
    assert cached.allergen_notice  # 비워 두지 않는다

    # 두 번째: 캐시에서 꺼내면 바로 답한다.
    monkeypatch.setattr(service, "lookup_product", found(cached))
    reader = FakeReader((plain, 0.95))
    r = checker(reader, store=store).check(CROP, ["밀"])
    assert reader.calls == 0 and r.judgement.verdict is V.NOT_DETECTED


# --- 성분표 영역 자동 찾기 (#75) ---------------------------------------------


class ShapeReader(FakeReader):
    """FakeReader + 받은 이미지 크기 기록. 영역을 잘라 읽었는지 본다."""

    def __init__(self, *passes):
        super().__init__(*passes)
        self.shapes = []

    def readtext(self, image, **kwargs):
        self.shapes.append(image.shape[:2])
        return super().readtext(image, **kwargs)


def test_auto_region_reads_only_found_panel(monkeypatch):
    """영역을 찾으면 그 부분만 잘라 읽는다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = ShapeReader((LABEL, 0.9))
    r = checker(reader, auto_region=True).check(CROP, ["땅콩"])
    assert r.panel_region is not None and "region" in r.timings_ms
    assert reader.calls == 2  # 영역 찾기 1번 + 성분표 읽기 1번
    # 영역 찾기는 사진 전체를 키워 읽는다(긴 변 1600 목표, 최대 3배 → 400px 은 3배)
    assert reader.shapes[0] == (CROP.shape[0] * 3, CROP.shape[1] * 3)
    assert reader.shapes[1] != (CROP.shape[0] * 2, CROP.shape[1] * 2)  # 전체 2배가 아니다
    assert r.judgement.verdict is V.MAY_CONTAIN


def test_auto_region_without_anchor_reads_whole_photo(monkeypatch):
    """단서가 없으면 사진 전체를 읽는다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = ShapeReader((["밀가루, 설탕"], 0.95))
    r = checker(reader, auto_region=True, max_reprocess=0).check(CROP, ["땅콩"])
    assert r.panel_region is None
    assert reader.shapes[1] == (CROP.shape[0] * 2, CROP.shape[1] * 2)  # 기존처럼 전체 2배
    assert r.retake == RETAKE_WHOLE


def test_auto_region_off_reads_once(monkeypatch):
    """영역 찾기를 끄면 OCR 은 한 번만 돈다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = ShapeReader((LABEL, 0.9))
    r = checker(reader, auto_region=False).check(CROP, ["땅콩"])
    assert reader.calls == 1 and r.panel_region is None
