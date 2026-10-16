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
    return lambda image, store=None, client=None: BarcodeLookup(read, None, reason=reason)


def checker(reader, **kw) -> IngredientChecker:
    # CROP 은 단색이라 품질 검사에 걸린다. 품질 검사는 아래에서 따로 본다.
    kw.setdefault("quality_check", None)
    # FakeReader 는 이미지와 상관없이 답하므로 회전 시도를 끈다. 회전은 아래에서 따로 본다.
    kw.setdefault("try_rotation", False)
    return IngredientChecker(reader=reader, conf_threshold=0.8, **kw)


# --- 바코드로 찾은 경우 ----------------------------------------------------------


def test_바코드로_찾으면_ocr_없이_판정한다(monkeypatch):
    info = ProductInfo(
        barcode=CODE,
        product_name="버터쿠키",
        raw_ingredients="밀가루, 버터",
        source=IngredientSource.API,
    )
    monkeypatch.setattr(
        service, "lookup_product", lambda image, store=None, client=None: BarcodeLookup(None, info)
    )
    reader = FakeReader((LABEL, 0.9))
    r = checker(reader).check(CROP, ["우유"])
    assert reader.calls == 0
    assert r.judgement.verdict is V.CONTAINS and r.judgement.source is IngredientSource.API
    assert r.message.startswith("주의하세요. 버터쿠키에 우유가 들어 있어요.")
    assert r.ingredients == ["밀가루", "버터"] and r.retake is None
    assert set(r.timings_ms) == {"barcode", "match"}


# --- 성분표 OCR ----------------------------------------------------------------


def test_ocr_한_번에_읽히면_바로_판정한다(monkeypatch):
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = FakeReader((LABEL, 0.9))
    r = checker(reader).check(CROP, ["땅콩"])
    assert reader.calls == 1 and not r.reprocessed
    assert r.judgement.verdict is V.MAY_CONTAIN and r.judgement.source is IngredientSource.OCR
    assert "땅콩을 쓰는 시설" in r.message and r.retake is None
    assert r.ocr_conf == pytest.approx(0.9) and "원재료명" in r.ocr_text
    assert set(r.timings_ms) == {"barcode", "ocr", "match"}


def test_신뢰도가_낮으면_같은_사진을_재처리한다(monkeypatch):
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = FakeReader((LABEL, 0.4), (LABEL, 0.9))
    r = checker(reader).check(CROP, ["땅콩"])
    assert reader.calls == 2 and r.reprocessed
    assert r.ocr_conf == pytest.approx(0.9) and r.retake is None


def test_재처리도_실패하면_더_가까이_재촬영을_요청한다(monkeypatch):
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(FakeReader((LABEL, 0.4))).check(CROP, ["땅콩"])
    assert r.retake == RETAKE_CLOSER and r.message == RETAKE_CLOSER
    assert r.judgement.verdict is V.UNDETERMINED and r.judgement.reason == "low_ocr_conf"


def test_원재료명이_안_보이면_전체가_보이게_재촬영을_요청한다(monkeypatch):
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(FakeReader((["밀가루, 설탕"], 0.95))).check(CROP, ["땅콩"])
    assert r.retake == RETAKE_WHOLE and r.judgement.reason == "no_ingredients"


def test_재촬영_상한이면_확인_불가로_끝낸다(monkeypatch):
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(FakeReader((LABEL, 0.4))).check(CROP, ["땅콩"], retake_count=2)
    assert r.retake is None and r.retake_count == 2
    assert r.judgement.verdict is V.UNDETERMINED
    assert "정확히 읽지 못했어요" in r.message


def test_판독이_불완전해도_함유를_찾으면_바로_경고한다(monkeypatch):
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(FakeReader((LABEL, 0.4))).check(CROP, ["우유"])
    assert r.judgement.verdict is V.CONTAINS and r.retake is None
    assert r.message.startswith("주의하세요.")


def test_재처리를_끄면_한_번만_읽는다(monkeypatch):
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


def test_바코드는_읽었는데_못_찾은_제품은_ocr_결과를_확인_전으로_캐시한다(monkeypatch, store):
    monkeypatch.setattr(service, "lookup_product", miss(CODE, "not_found"))
    checker(FakeReader((LABEL, 0.9)), store=store).check(CROP, ["땅콩"])
    cached = store.get_product(CODE)
    assert cached is not None and "탈지분유" in cached.raw_ingredients
    assert "땅콩" in cached.cross_contamination


def test_판독이_불확실하면_캐시하지_않는다(monkeypatch, store):
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


def test_흐린_사진은_ocr_없이_재촬영을_안내한다(monkeypatch):
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = FakeReader((LABEL, 0.9))
    r = checker(reader, quality_check=bad("blurry")).check(CROP, ["땅콩"])
    assert reader.calls == 0  # 30초짜리 OCR 을 돌리지 않는다
    assert r.retake == RETAKE_STEADY and r.message == RETAKE_STEADY
    assert r.judgement.verdict is V.UNDETERMINED and r.judgement.reason == "blurry"
    assert "quality" in r.timings_ms and "ocr" not in r.timings_ms


def test_반사가_심하면_포장을_돌려_달라고_한다(monkeypatch):
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(FakeReader((LABEL, 0.9)), quality_check=bad("glare")).check(CROP, ["땅콩"])
    assert r.retake == RETAKE_GLARE


def test_재촬영_상한을_넘으면_확인_불가로_끝낸다(monkeypatch):
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(FakeReader((LABEL, 0.9)), quality_check=bad("blurry")).check(
        CROP, ["땅콩"], retake_count=2
    )
    assert r.retake is None
    assert r.judgement.verdict is V.UNDETERMINED
    assert "흐려서" in r.message and "포장지" in r.message


def test_바코드로_찾으면_품질이_나빠도_판정한다(monkeypatch):
    info = ProductInfo(barcode=CODE, product_name="쿠키", raw_ingredients="밀가루, 버터")
    monkeypatch.setattr(
        service, "lookup_product", lambda image, store=None, client=None: BarcodeLookup(None, info)
    )
    r = checker(FakeReader((LABEL, 0.9)), quality_check=bad("blurry")).check(CROP, ["우유"])
    assert r.judgement.verdict is V.CONTAINS and r.quality is None


def test_여러_프레임이면_가장_선명한_것으로_확인한다(monkeypatch):
    seen = []

    def lookup(image, store=None, client=None):
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


def test_성분표가_돌아가_있으면_돌려서_읽는다(monkeypatch):
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = SidewaysReader()
    r = checker(reader, try_rotation=True).check(CROP, ["땅콩"])  # CROP 은 가로로 긴 이미지
    assert r.rotation in (90, 270)
    assert r.judgement.verdict is V.MAY_CONTAIN and r.retake is None
    assert len(reader.shapes) == 3  # 그대로 1번 + 90°·270°


def test_정방향이면_돌리지_않는다(monkeypatch):
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


def test_알레르기_표시_없는_조회_정보로는_미검출이라_하지_않고_성분표를_읽는다(monkeypatch):
    monkeypatch.setattr(service, "lookup_product", found(RAMEN))
    reader = FakeReader((["원재료명: 소맥분(밀), 대두유, 정제염"], 0.9))
    r = checker(reader).check(CROP, ["밀"])
    assert reader.calls == 1  # 성분표를 읽었다
    assert r.judgement.verdict is V.CONTAINS and r.judgement.source is IngredientSource.OCR
    assert r.judgement.product.product_name == "신라면"  # 조회한 제품명을 이어 쓴다


def test_성분표도_못_읽으면_재촬영을_요청한다(monkeypatch):
    monkeypatch.setattr(service, "lookup_product", found(RAMEN))
    r = checker(FakeReader((["설탕, 정제수"], 0.95))).check(CROP, ["밀"])  # 원재료명 머리말 없음
    assert r.retake == RETAKE_WHOLE
    assert r.judgement.verdict is V.UNDETERMINED


def test_알레르기_표시가_있으면_조회_정보대로_답한다(monkeypatch):
    info = ProductInfo(barcode=CODE, product_name="사이다", raw_ingredients="정제수, 설탕",
                       allergen_notice="알레르기 유발물질 표시 없음")  # fmt: skip
    monkeypatch.setattr(service, "lookup_product", found(info))
    reader = FakeReader((LABEL, 0.9))
    r = checker(reader).check(CROP, ["밀"])
    assert reader.calls == 0 and r.judgement.verdict is V.NOT_DETECTED


def test_조회_정보에서_찾으면_표시가_없어도_바로_경고한다(monkeypatch):
    info = ProductInfo(barcode=CODE, product_name="쿠키", raw_ingredients="밀가루, 버터")
    monkeypatch.setattr(service, "lookup_product", found(info))
    reader = FakeReader((LABEL, 0.9))
    r = checker(reader).check(CROP, ["밀"])
    assert reader.calls == 0 and r.judgement.verdict is V.CONTAINS


def test_등록한_알레르기가_없으면_성분표를_읽지_않는다(monkeypatch):
    monkeypatch.setattr(service, "lookup_product", found(RAMEN))
    reader = FakeReader((LABEL, 0.9))
    checker(reader).check(CROP, [])
    assert reader.calls == 0


def test_한_번_성분표로_확인한_제품은_다음에_ocr을_다시_돌리지_않는다(monkeypatch, store):
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
