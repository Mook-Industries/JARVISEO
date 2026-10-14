"""성분표 글자 → 구간 분리 → 성분 목록 테스트.

OCR 엔진 없이 돈다. OCR 이 읽어 냈다고 가정한 글자를 넣는다.

실행: pytest tests/test_nutrition_ocr.py
"""

import numpy as np
import pytest

from jarviseo.nutrition.ocr import (
    crop_and_upscale,
    join_lines,
    mean_confidence,
    parse_ingredient_text,
    parse_ingredients,
    read_panel,
    split_sections,
    split_terms,
)
from jarviseo.types import BBox, OCRLine

COOKIE = (
    "제품명 초코쿠키 식품유형 과자 원재료명 : 밀가루(밀:미국산, 호주산), 설탕, "
    "초콜릿 15%(설탕, 코코아매스, 전지분유, 대두레시틴), 쇼트닝(팜유), 탈지 분유, 계란, "
    "정제소금, 산도조절제(구연산), 합성향료 (우유, 대두, 밀, 알류 함유) "
    "이 제품은 땅콩, 호두를 사용한 제품과 같은 제조시설에서 제조하고 있습니다. "
    "내용량 120g 보관방법 직사광선을 피해"
)
CHIPS = (
    "원재료 및 함량: 감자(국산)85%, 해바라기유, 양파분말, 새우분말(새우:베트남산) "
    "알레르기 유발물질: 새우, 대두 함유 ※ 같은 시설에서 우유를 사용합니다. 소비기한 별도표기"
)


def lines(*texts: str) -> list[OCRLine]:
    return [
        OCRLine(text=t, confidence=0.9, bbox=BBox(0, i * 20, 100, i * 20 + 15))
        for i, t in enumerate(texts)
    ]


# --- 구간 분리 -----------------------------------------------------------------


def test_세_구간을_나눈다():
    sec = split_sections(COOKIE)
    assert sec.has_ingredient_header
    assert sec.ingredients.startswith("밀가루")
    assert "내용량" not in sec.ingredients  # 다른 항목에서 멈춘다
    assert split_terms(sec.allergen_notice) == ["우유", "대두", "밀", "알류"]
    assert sec.cross_contamination.startswith("이 제품은 땅콩, 호두")
    # 혼입 문장이 원재료 구간에 섞이면 땅콩이 '함유'로 잘못 잡힌다
    assert "땅콩" not in sec.ingredients


def test_알레르기_유발물질_머리말과_참고표시_구분():
    # "※" 뒤는 다른 문장이다
    sec = split_sections(CHIPS)
    assert split_terms(sec.allergen_notice) == ["새우", "대두"]
    assert sec.cross_contamination == "같은 시설에서 우유를 사용합니다"
    assert "소비기한" not in sec.ingredients


def test_원재료명이_없으면_남은_글자를_보조로_둔다():
    # 머리말이 없으면 판정은 '확인 불가'로 가지만(has_ingredient_header=False),
    # 남은 글자에서 알레르겐을 찾으면 '함유' 경고는 해야 해서 버리지 않는다.
    sec = split_sections("설탕, 물엿, 젤라틴 영양정보")
    assert not sec.has_ingredient_header
    assert sec.ingredients == "설탕, 물엿, 젤라틴 영양정보"


def test_마침표가_없어도_혼입_문장이_원재료를_삼키지_않는다():
    text = "원재료명 밀가루, 설탕, 버터, 소금, 효모 같은 제조시설에서 땅콩 사용"
    sec = split_sections(text)
    assert "밀가루" in sec.ingredients
    assert "땅콩" in sec.cross_contamination


def test_함유량은_알레르기_표시가_아니다():
    sec = split_sections("원재료명 설탕, 정제수 나트륨 함유량 낮음")
    assert sec.allergen_notice == ""


def test_ocr_이_글자_사이에_공백을_넣어도_찾는다():
    sec = split_sections("원 재 료 명 : 밀가루, 설탕 알 레 르 기 유발물질: 밀")
    assert sec.has_ingredient_header
    assert split_terms(sec.allergen_notice) == ["밀"]


# --- 성분 목록 ----------------------------------------------------------------


def test_쿠키_성분_목록():
    got = parse_ingredient_text(split_sections(COOKIE).ingredients)
    assert got == [
        "밀가루", "밀", "설탕", "초콜릿", "코코아매스", "전지분유", "대두레시틴",
        "쇼트닝", "팜유", "탈지 분유", "계란", "정제소금", "산도조절제", "구연산", "합성향료",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 괄호 안 알레르겐을 버리면 안 된다 (재현율 우선)
        ("초콜릿(설탕, 전지분유)", ["초콜릿", "설탕", "전지분유"]),
        # 원산지·함량만 버린다
        ("감자(국산)85%, 대두(외국산:미국, 캐나다)", ["감자", "대두", "미국", "캐나다"]),
        # '산'으로 끝나는 성분은 원산지가 아니다
        ("아황산, 구연산, 젖산", ["아황산", "구연산", "젖산"]),
        # 중첩 괄호
        ("과자(밀가루(밀:미국산), 버터)", ["과자", "밀가루", "밀", "버터"]),
        # OCR 이 닫는 괄호를 놓친 경우
        ("크림(우유, 설탕", ["크림", "우유", "설탕"]),
        # 전각·변형 기호
        ("밀가루（밀），설탕、우유；소금", ["밀가루", "밀", "설탕", "우유", "소금"]),
        # 중복 제거, 순서 유지
        ("설탕, 밀가루, 설탕", ["설탕", "밀가루"]),
    ],
)
def test_성분_파싱(text, expected):
    assert parse_ingredient_text(text) == expected


def test_알레르기_표시_나열():
    assert split_terms("우유, 대두·밀/땅콩 등 함유") == ["우유", "대두", "밀", "땅콩"]


# --- 실제 EasyOCR 출력 (맑은 고딕 14px 합성 성분표, 2배 확대) ---------------------
# 쉼표 누락, "같은"→"끝은", "호두를"→"호두록" 같은 실제 오인식이 들어 있다.

EASYOCR_SCALE2 = (
    "원재료명: 밀가루(밀:미국산) , 설탕 초홀릿(설탕 "
    "코코아매스 전지분유 대두레시터) 소트님(판유) , "
    "탈지분유 계란  정제소금 (우유 대두 밀 알류 함유) "
    "이 제품은 땅콩 호두록 사용한 제품과 끝은 제조시설에서 제조하고 있습니다."
)


def test_실제_ocr_쉼표가_빠진_알레르기_표시도_자른다():
    sec = split_sections(EASYOCR_SCALE2)
    assert split_terms(sec.allergen_notice) == ["우유", "대두", "밀", "알류"]


def test_실제_ocr_같은을_잘못_읽어도_혼입_문장을_찾는다():
    sec = split_sections(EASYOCR_SCALE2)
    assert "땅콩" in sec.cross_contamination
    assert "땅콩" not in sec.ingredients


def test_실제_ocr_알레르기를_잘못_읽어도_머리말을_찾는다():
    # scripts/check_ingredient_ocr.py 로 돌린 실제 출력 ("르"→"로")
    text = (
        "원재료 및 함량: 감자(국산) 85%, 해바라기유 양파분말 새우분말(새우:베트님산) "
        "알레로기 유발물질: 새우 대두 함유"
    )
    sec = split_sections(text)
    assert split_terms(sec.allergen_notice) == ["새우", "대두"]
    assert "유발물질" not in sec.ingredients


def test_실제_ocr_성분이_붙어도_알레르겐_표기는_남는다():
    got = parse_ingredient_text(split_sections(EASYOCR_SCALE2).ingredients)
    # 쉼표가 빠져 붙은 덩어리 안에도 '전지분유', '탈지분유' 가 들어 있어야 판정에서 잡힌다
    joined = " | ".join(got)
    assert "전지분유" in joined and "탈지분유" in joined


# --- 줄 단위 입력 ------------------------------------------------------------


def test_줄을_이어_붙여_파싱한다():
    got = parse_ingredients(lines("원재료명: 밀가루, 탈지분", "유, 설탕", "영양정보 100kcal"))
    # 줄바꿈에 걸린 "탈지분 유"는 판정 단계에서 공백을 지우고 비교한다
    assert got == ["밀가루", "탈지분 유", "설탕"]


def test_join_lines_는_공백_하나로_잇는다():
    assert join_lines(lines("밀가루,  ", "  설탕")) == "밀가루, 설탕"


# --- 이미지 → 글자 (EasyOCR 은 가짜로) ---------------------------------------


class FakeReader:
    """easyocr.Reader.readtext 와 같은 모양으로 돌려준다: [(꼭짓점 4개, 글자, 신뢰도), ...]"""

    def __init__(self, results):
        self.results = results
        self.calls = 0

    def readtext(self, image, **kwargs):
        self.calls += 1
        return self.results


def box(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_crop_and_upscale_은_자르고_키운다():
    image = np.zeros((100, 200, 3), np.uint8)
    out = crop_and_upscale(image, BBox(10, 20, 60, 40), scale=3.0)
    assert out.shape[:2] == (60, 150)


def test_crop_and_upscale_은_이미지_밖_좌표를_안쪽으로_자른다():
    image = np.zeros((100, 200, 3), np.uint8)
    out = crop_and_upscale(image, BBox(-50, -50, 50, 30), scale=1.0)
    assert out.shape[:2] == (30, 50)


def test_crop_and_upscale_은_너무_크게_키우지_않는다():
    image = np.zeros((1080, 1920, 3), np.uint8)
    out = crop_and_upscale(image, BBox(0, 0, 1920, 1080), scale=3.0)
    assert max(out.shape[:2]) <= 2560


def test_crop_and_upscale_빈_영역은_예외():
    with pytest.raises(ValueError):
        crop_and_upscale(np.zeros((100, 100, 3), np.uint8), BBox(150, 150, 200, 200))


def test_read_panel_은_덩어리를_줄로_묶고_위에서_아래로_정렬한다():
    reader = FakeReader([
        (box(120, 52, 200, 70), "설탕", 0.8),       # 둘째 줄 오른쪽
        (box(10, 10, 110, 30), "원재료명:", 0.9),   # 첫째 줄 왼쪽
        (box(10, 50, 110, 68), "밀가루,", 0.6),      # 둘째 줄 왼쪽 (살짝 높이 다름)
        (box(120, 12, 220, 31), "밀가루", 0.9),      # 첫째 줄 오른쪽
        (box(10, 90, 50, 105), "  ", 0.1),           # 빈 글자는 버린다
    ])  # fmt: skip
    lines = read_panel(np.zeros((10, 10, 3), np.uint8), reader=reader)
    assert [ln.text for ln in lines] == ["원재료명: 밀가루", "밀가루, 설탕"]
    # 신뢰도는 글자 수 가중 평균: (0.6*4 + 0.8*2) / 6
    assert lines[1].confidence == pytest.approx((0.6 * 4 + 0.8 * 2) / 6)
    assert lines[0].bbox == BBox(10, 10, 220, 31)


def test_mean_confidence():
    assert mean_confidence([]) == 0.0
    ls = [OCRLine("가나다라", 0.9, BBox(0, 0, 1, 1)), OCRLine("마바", 0.3, BBox(0, 0, 1, 1))]
    assert mean_confidence(ls) == pytest.approx((0.9 * 4 + 0.3 * 2) / 6)


# --- 평가 세트에서 찾은 파서 버그 (#49 정답 글자로 돌린 결과) ------------------------
# 정답 글자를 그대로 넣어도 알레르겐을 69% 만 찾던 원인들이다.

LONG_LIST = (
    "밀, 달걀, 땅콩, 호두, 복숭아, 토마토, 닭고기, 쇠고기, 돼지고기, 오징어, 새우, 게, "
    "조개류(굴, 홍합 포함) 혼입가능"
)


def test_혼입_목록이_40자를_넘어도_경계부터_잡는다():
    sec = split_sections(f"원재료명 밀가루, 설탕 • {LONG_LIST} • 제품의 신선도 유지")
    assert sec.cross_contamination == LONG_LIST
    assert "땅콩" not in sec.ingredients
    assert "신선도" not in sec.cross_contamination  # 다음 • 에서 끝난다


def test_품목보고번호_뒤_혼입_목록():
    sec = split_sections("원재료명 밀가루 품목보고번호 19870415003381 우유, 대두, 땅콩 혼입가능")
    assert sec.cross_contamination == "우유, 대두, 땅콩 혼입가능"


def test_경계_없이_이어진_목록은_원재료를_혼입으로_낮추지_않는다():
    # 원재료도 이름 목록처럼 보인다. 경계가 없으면 목록 전체를 혼입으로 보면 안 된다.
    sec = split_sections("원재료명 밀가루, 설탕, 버터 땅콩, 호두 혼입가능")
    assert "밀가루" in sec.ingredients
    assert "버터" in sec.ingredients


def test_괄호가_끼어도_이_제품은_부터_잡는다():
    sec = split_sections(
        "원재료명 밀가루 •이 제품은 땅콩, 조개류(굴, 전복, 홍합 포함)를 사용한 제품과 "
        "같은 제조 시설에서 제조하고 있습니다."
    )
    assert sec.cross_contamination.startswith("이 제품은 땅콩")


def test_함유_표시가_원재료명_머리말을_삼키지_않는다():
    sec = split_sections("원재료명 밀가루, 설탕, 정제 미강유 우유, 대두 함유")
    assert sec.has_ingredient_header
    assert sec.ingredients.startswith("밀가루")
    assert "정제 미강유" in sec.ingredients
    assert split_terms(sec.allergen_notice) == ["우유", "대두"]


def test_괄호가_닫히지_않아도_원재료는_표시로_가져가지_않는다():
    sec = split_sections("원재료명 코코아분말 1(인도네시아 물엿, 향료 2종 함유")
    assert sec.allergen_notice == ""
    assert "물엿" in sec.ingredients


def test_첫_글자가_잘린_머리말도_찾는다():
    sec = split_sections("재료명 밀가루(밀 미국산), 백설탕 장재질 폴리프로필렌")
    assert sec.has_ingredient_header
    assert sec.ingredients == "밀가루(밀 미국산), 백설탕"


def test_보관_안내에서_원재료가_끝난다():
    # "얼지 않게"의 '게'가 알레르겐(게)으로 잡히던 문제
    sec = split_sections("원재료명 정제수, 구연산. •직사광선을 피해 얼지 않게 보관")
    assert sec.ingredients == "정제수, 구연산"


# --- 실제 PaddleOCR 출력에서 찾은 것 (검수한 정답으로 다시 채점, 2026-10-14) ---------


def test_쉼표가_빠진_함유_표시도_이름만_모은다():
    sec = split_sections("원재료명 간장분말, 건청경채 닭고기 쇠고기 함유")
    assert split_terms(sec.allergen_notice) == ["닭고기", "쇠고기"]
    assert "건청경채" in sec.ingredients


def test_쉼표가_빠진_목록_중간에_쉼표가_섞여도_모은다():
    sec = split_sections("원재료명 볶음참깨, 구운김 밀 대두 계란,우유,닭고기 쇠고기 함유")
    assert split_terms(sec.allergen_notice) == ["밀", "대두", "계란", "우유", "닭고기", "쇠고기"]
    assert "구운김" in sec.ingredients


def test_머리말이_없어도_원재료_속_알레르겐은_남는다():
    # 머리말이 잘린 라면 성분표. 예전엔 원재료 글자를 통째로 버려서 간장(대두)·계란을 놓쳤다
    sec = split_sections("면/소맥분(밀:호주산), 간장분말, 계란스크램블 닭고기 쇠고기 함유")
    assert not sec.has_ingredient_header
    assert "간장분말" in sec.ingredients and "계란스크램블" in sec.ingredients
    assert split_terms(sec.allergen_notice) == ["닭고기", "쇠고기"]


def test_쉼표를_마침표로_읽어도_혼입_목록이_끊기지_않는다():
    sec = split_sections(
        "원재료명 정제수 품목보고번호 199102620031 ·대두,땅콩.밀.복숭아 혼입 가능."
    )
    assert sec.cross_contamination.endswith("대두,땅콩,밀,복숭아 혼입 가능")


def test_문장_끝_마침표는_그대로_둔다():
    sec = split_sections(
        "원재료명 밀가루. 이 제품은 땅콩을 사용한 제품과 같은 제조시설에서 제조합니다."
    )
    assert sec.ingredients == "밀가루"
    assert sec.cross_contamination.startswith("이 제품은 땅콩")
