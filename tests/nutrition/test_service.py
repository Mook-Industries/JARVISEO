"""성분 판정 전체 흐름 테스트. 바코드 조회와 OCR 엔진은 가짜로 바꾼다.

실행: pytest tests/test_nutrition_service.py
"""

import numpy as np
import pytest

from jarviseo.memory.store import MemoryStore
from jarviseo.nutrition import service
from jarviseo.nutrition.barcode import BarcodeLookup
from jarviseo.nutrition.response import RETAKE_CLOSER, RETAKE_WHOLE
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
