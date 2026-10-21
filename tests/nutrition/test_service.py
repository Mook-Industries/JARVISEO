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
from jarviseo.nutrition.response import (
    OCCLUDED_NOTE,
    RETAKE_CLOSER,
    RETAKE_GLARE,
    RETAKE_STEADY,
    RETAKE_UNCOVER,
    RETAKE_WHOLE,
    SMALL_TEXT_NOTE,
)
from jarviseo.nutrition.service import IngredientChecker
from jarviseo.types import AllergenVerdict as V
from jarviseo.types import BarcodeRead, IngredientSource, ProductInfo

CROP = np.full((120, 400, 3), 255, np.uint8)
CODE = "8801234567893"

LABEL = [
    "원재료명: 밀가루, 설탕, 탈지분유, 정제소금",
    "이 제품은 땅콩을 사용한 제품과 같은 제조시설에서 제조합니다.",
]


def box(i: int) -> list[list[int]]:
    """i 번째 줄의 글자 상자. 사진 가장자리에 닿으면 성분표가 잘린 것으로 보므로 안쪽에 둔다."""
    y = 20 + i * 30
    return [[20, y], [300, y], [300, y + 20], [20, y + 20]]


class FakeReader:
    """readtext 를 부를 때마다 passes 에서 하나씩 꺼내 돌려준다. (줄 글자들, 신뢰도)"""

    def __init__(self, *passes):
        self.passes = list(passes)
        self.calls = 0

    def readtext(self, image, **kwargs):
        texts, conf = self.passes[min(self.calls, len(self.passes) - 1)]
        self.calls += 1
        return [(box(i), t, conf)
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
    # 가짜 글자 상자는 높이 20px(원본 10px)이라 작은 글씨에 걸린다. 작은 글씨는 아래에서 따로 본다.
    kw.setdefault("min_text_px", 0)
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
    assert r.message == "해당 제품에는 우유가 들어 있어요."
    assert r.ingredients == ["밀가루", "버터"] and r.retake is None
    assert set(r.timings_ms) == {"barcode", "match"}


def test_message_uses_registered_allergen_name(monkeypatch):
    """'계란'으로 등록한 사용자에게는 '알류'가 아니라 '계란'으로 말한다."""
    info = ProductInfo(barcode=CODE, product_name="라면", raw_ingredients="소맥분, 계란분말",
                       allergen_notice="밀, 알류")  # fmt: skip
    monkeypatch.setattr(service, "lookup_product", lambda image, **_: BarcodeLookup(None, info))
    r = checker(FakeReader((LABEL, 0.9))).check(CROP, ["계란"])
    assert r.message == "해당 제품에는 계란이 들어 있어요."


# --- 성분표 OCR ----------------------------------------------------------------


def test_judges_after_single_ocr_read(monkeypatch):
    """OCR 한 번에 읽히면 바로 판정한다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = FakeReader((LABEL, 0.9))
    r = checker(reader).check(CROP, ["땅콩"])
    assert reader.calls == 1 and not r.reprocessed
    assert r.judgement.verdict is V.MAY_CONTAIN and r.judgement.source is IngredientSource.OCR
    assert "같은 시설에서 쓰는 땅콩이" in r.message and r.retake is None
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
    assert r.message.startswith("해당 제품에는")


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
        return [(box(i), t, conf)
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


# --- 다른 모듈이 부르는 입구 (answer) --------------------------------------------


def test_answer_skips_when_ingredient_not_needed(monkeypatch):
    """성분 분석이 필요 없으면 바코드도 OCR 도 돌리지 않고 None 이다."""
    called = []
    monkeypatch.setattr(service, "lookup_product", lambda image, **_: called.append(1))
    reader = FakeReader((LABEL, 0.9))
    assert checker(reader).answer(CROP, False, ["땅콩"]) is None
    assert called == [] and reader.calls == 0


@pytest.mark.parametrize("crop", [None, np.zeros((0, 0, 3), np.uint8)])
def test_answer_without_crop_is_none(crop):
    """크롭이 없으면 None 이다."""
    assert checker(FakeReader((LABEL, 0.9))).answer(crop, True, ["땅콩"]) is None


def test_answer_returns_one_sentence(monkeypatch):
    """성분 분석이 필요하면 읽어 줄 문장 하나를 돌려준다."""
    info = ProductInfo(barcode=CODE, product_name="라면", raw_ingredients="소맥분, 계란분말",
                       allergen_notice="밀, 알류")  # fmt: skip
    monkeypatch.setattr(service, "lookup_product", lambda image, **_: BarcodeLookup(None, info))
    text = checker(FakeReader((LABEL, 0.9))).answer(CROP, True, ["계란", "밀"])
    assert text == "해당 제품에는 계란, 밀이 들어 있어요."


def test_answer_returns_retake_guide_when_unreadable(monkeypatch):
    """사진으로 못 읽으면 재촬영 안내 문장을 돌려준다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    reader = FakeReader((["글자 없음"], 0.2))
    text = checker(reader).answer(CROP, True, ["땅콩"])
    assert text in {RETAKE_CLOSER, RETAKE_WHOLE, RETAKE_STEADY, RETAKE_GLARE}


def test_close_closes_food_qr():
    """close 는 푸드QR 연결을 닫는다."""

    class FakeQr:
        closed = False

        def close(self):
            self.closed = True

    qr = FakeQr()
    checker(FakeReader((LABEL, 0.9)), food_qr=qr).close()
    assert qr.closed


def test_upright_panel_without_header_is_not_rotated(monkeypatch):
    """머리말을 못 찾아도 높은 신뢰도로 많이 읽었으면 돌려 읽지 않는다 (OCR 호출을 줄인다)."""
    monkeypatch.setattr(service, "lookup_product", miss())
    upright = ["밀가루, 설탕, 탈지분유, 정제소금, 코코아분말, 식물성유지", "땅콩, 호두, 대두 함유"]
    reader = FakeReader((upright, 0.95))
    r = checker(reader, try_rotation=True).check(CROP, ["땅콩"])
    assert r.rotation == 0
    assert reader.calls == 2  # 그대로 1번 + 재처리 1번. 90°·270° 는 읽지 않는다


# --- 성분표가 가려졌을 때 --------------------------------------------------------


class BoxReader:
    """줄마다 (글자, 위쪽 y) 를 정해 돌려준다. 가려짐·잘림을 만들려고 좌표를 직접 준다."""

    def __init__(self, rows):
        self.rows = rows

    def readtext(self, image, **kwargs):
        return [([[20, y], [300, y], [300, y + 20], [20, y + 20]], t, 0.9) for t, y in self.rows]


# 머리말이 '원' 없이 읽힘 → 성분표 왼쪽이 잘린 자리 (label_01 실측 모양)
COVERED = [("재료명:", 20), ("밀가루, 설탕,", 50), ("대두 함유", 80)]
# 원재료명이 사진 아래 가장자리에서 잘림 (CROP 높이 120 × 배율 2 = 240)
CUT = [("원재료명:", 20), ("밀가루, 설탕,", 50), ("정제소금, 탈지", 220)]


def test_occluded_label_with_allergen_warns_and_says_covered(monkeypatch):
    """가려졌어도 찾은 경고는 바로 전하고, 가려진 데가 있다고 덧붙인다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(BoxReader(COVERED)).check(CROP, ["밀"])
    assert r.occlusion == "cut_header" and r.judgement.verdict is V.CONTAINS
    assert r.message == f"해당 제품에는 밀이 들어 있어요. {OCCLUDED_NOTE}"
    assert r.retake is None


def test_occluded_label_without_allergen_asks_to_uncover(monkeypatch):
    """가려진 사진에서 못 찾았으면 '못 찾았어요'가 아니라 가려졌다고 말한다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(BoxReader(CUT)).check(CROP, ["땅콩"])
    assert r.occlusion == "cut_edge"
    assert r.judgement.verdict is V.UNDETERMINED and r.judgement.reason == "occluded"
    assert r.retake == RETAKE_UNCOVER and r.message == RETAKE_UNCOVER


def test_occluded_after_max_retakes_says_could_not_recognize(monkeypatch):
    """재촬영을 다 써도 가려져 있으면 잘 인식하지 못했다고 말하고 끝낸다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(BoxReader(CUT), max_retake=2).check(CROP, ["땅콩"], retake_count=2)
    assert r.retake is None
    assert r.message.startswith("성분표 일부가 가려져서 잘 인식하지 못했어요.")


def test_occluded_ocr_is_not_cached(monkeypatch):
    """가려진 사진에서 읽은 성분은 캐시에 남기지 않는다."""
    monkeypatch.setattr(service, "lookup_product", miss(CODE, "not_found"))
    store = MemoryStore(database_url="sqlite://")
    store.init_schema()
    checker(BoxReader(COVERED), store=store).check(CROP, ["밀"])
    assert store.get_product(CODE) is None


def test_visible_label_has_no_occlusion(monkeypatch):
    """성분표가 다 보이면 평소대로 말한다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    rows = [("원재료명:", 20), ("밀가루, 설탕,", 50), ("대두 함유", 80)]
    r = checker(BoxReader(rows)).check(CROP, ["밀"])
    assert r.occlusion == "" and r.message == "해당 제품에는 밀이 들어 있어요."


# --- 글씨가 작게 찍혔을 때 --------------------------------------------------------

# 가짜 상자 높이 20px ÷ 배율 2 = 원본 글줄 10px
SMALL = [("원재료명: 밀가루, 설탕,", 20), ("대두 함유, 정제소금", 50)]


def test_small_text_with_allergen_warns_and_says_may_miss(monkeypatch):
    """글씨가 작아도 찾은 경고는 바로 전하고, 빠진 성분이 있을 수 있다고 덧붙인다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(BoxReader(SMALL), min_text_px=22).check(CROP, ["밀"])
    assert r.text_px == 10 and r.judgement.verdict is V.CONTAINS
    assert r.message == f"해당 제품에는 밀이 들어 있어요. {SMALL_TEXT_NOTE}"
    assert r.retake is None


def test_small_text_without_allergen_asks_closer(monkeypatch):
    """글씨가 작은데 못 찾았으면 '못 찾았어요' 대신 더 가까이 보여 달라고 한다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(BoxReader(SMALL), min_text_px=22).check(CROP, ["땅콩"])
    assert r.judgement.verdict is V.UNDETERMINED and r.judgement.reason == "small_text"
    assert r.retake == RETAKE_CLOSER and r.message == RETAKE_CLOSER


def test_small_text_after_max_retakes_says_could_not_read(monkeypatch):
    """재촬영을 다 써도 글씨가 작으면 정확히 읽지 못했다고 말하고 끝낸다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(BoxReader(SMALL), min_text_px=22, max_retake=2).check(
        CROP, ["땅콩"], retake_count=2
    )
    assert r.retake is None
    assert r.message.startswith("글씨가 작아서 성분표를 정확히 읽지 못했어요.")


def test_small_text_ocr_is_not_cached(monkeypatch):
    """글씨가 작은 사진에서 읽은 성분은 캐시에 남기지 않는다. 오독이 굳으면 안 된다."""
    monkeypatch.setattr(service, "lookup_product", miss(CODE, "not_found"))
    store = MemoryStore(database_url="sqlite://")
    store.init_schema()
    checker(BoxReader(SMALL), store=store, min_text_px=22).check(CROP, ["밀"])
    assert store.get_product(CODE) is None


def test_large_enough_text_has_no_note(monkeypatch):
    """글씨가 기준보다 크면 평소대로 말한다."""
    monkeypatch.setattr(service, "lookup_product", miss())
    r = checker(BoxReader(SMALL), min_text_px=8).check(CROP, ["밀"])
    assert r.message == "해당 제품에는 밀이 들어 있어요."
