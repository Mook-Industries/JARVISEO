"""성분표 글자 → 구간 분리 → 성분 목록 테스트.

OCR 엔진 없이 돈다. OCR 이 읽어 냈다고 가정한 글자를 넣는다.

실행: pytest tests/test_nutrition_ocr.py
"""

import numpy as np
import pytest

from jarviseo.nutrition.ocr import (
    _group_lines,
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


def test_splits_three_sections():
    """세 구간을 나눈다."""
    sec = split_sections(COOKIE)
    assert sec.has_ingredient_header
    assert sec.ingredients.startswith("밀가루")
    assert "내용량" not in sec.ingredients  # 다른 항목에서 멈춘다
    assert split_terms(sec.allergen_notice) == ["우유", "대두", "밀", "알류"]
    assert sec.cross_contamination.startswith("이 제품은 땅콩, 호두")
    # 혼입 문장이 원재료 구간에 섞이면 땅콩이 '함유'로 잘못 잡힌다
    assert "땅콩" not in sec.ingredients


def test_distinguishes_allergen_header_from_note():
    """알레르기 유발물질 머리말과 참고 표시를 구분한다."""
    # "※" 뒤는 다른 문장이다
    sec = split_sections(CHIPS)
    assert split_terms(sec.allergen_notice) == ["새우", "대두"]
    assert sec.cross_contamination == "같은 시설에서 우유를 사용합니다"
    assert "소비기한" not in sec.ingredients


def test_keeps_rest_as_fallback_without_ingredient_header():
    """원재료명이 없으면 남은 글자를 보조로 둔다."""
    # 머리말이 없으면 판정은 '확인 불가'로 가지만(has_ingredient_header=False),
    # 남은 글자에서 알레르겐을 찾으면 '함유' 경고는 해야 해서 버리지 않는다.
    sec = split_sections("설탕, 물엿, 젤라틴 영양정보")
    assert not sec.has_ingredient_header
    assert sec.ingredients == "설탕, 물엿, 젤라틴 영양정보"


def test_may_contain_without_period_keeps_ingredients():
    """마침표가 없어도 혼입 문장이 원재료를 삼키지 않는다."""
    text = "원재료명 밀가루, 설탕, 버터, 소금, 효모 같은 제조시설에서 땅콩 사용"
    sec = split_sections(text)
    assert "밀가루" in sec.ingredients
    assert "땅콩" in sec.cross_contamination


def test_content_amount_is_not_allergen_label():
    """함유량은 알레르기 표시가 아니다."""
    sec = split_sections("원재료명 설탕, 정제수 나트륨 함유량 낮음")
    assert sec.allergen_notice == ""


def test_finds_when_ocr_inserts_spaces():
    """OCR 이 글자 사이에 공백을 넣어도 찾는다."""
    sec = split_sections("원 재 료 명 : 밀가루, 설탕 알 레 르 기 유발물질: 밀")
    assert sec.has_ingredient_header
    assert split_terms(sec.allergen_notice) == ["밀"]


# --- 성분 목록 ----------------------------------------------------------------


def test_cookie_ingredient_list():
    """쿠키 성분 목록."""
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
def test_parses_ingredients(text, expected):
    """성분 파싱."""
    assert parse_ingredient_text(text) == expected


def test_lists_allergen_label():
    """알레르기 표시 나열."""
    assert split_terms("우유, 대두·밀/땅콩 등 함유") == ["우유", "대두", "밀", "땅콩"]


# --- 실제 EasyOCR 출력 (맑은 고딕 14px 합성 성분표, 2배 확대) ---------------------
# 쉼표 누락, "같은"→"끝은", "호두를"→"호두록" 같은 실제 오인식이 들어 있다.

EASYOCR_SCALE2 = (
    "원재료명: 밀가루(밀:미국산) , 설탕 초홀릿(설탕 "
    "코코아매스 전지분유 대두레시터) 소트님(판유) , "
    "탈지분유 계란  정제소금 (우유 대두 밀 알류 함유) "
    "이 제품은 땅콩 호두록 사용한 제품과 끝은 제조시설에서 제조하고 있습니다."
)


def test_real_ocr_splits_allergen_label_missing_commas():
    """실제 OCR 결과: 쉼표가 빠진 알레르기 표시도 자른다."""
    sec = split_sections(EASYOCR_SCALE2)
    assert split_terms(sec.allergen_notice) == ["우유", "대두", "밀", "알류"]


def test_real_ocr_finds_may_contain_with_misread_word():
    """실제 OCR 결과: "같은"을 잘못 읽어도 혼입 문장을 찾는다."""
    sec = split_sections(EASYOCR_SCALE2)
    assert "땅콩" in sec.cross_contamination
    assert "땅콩" not in sec.ingredients


def test_real_ocr_finds_header_with_misread_word():
    """실제 OCR 결과: "알레르기"를 잘못 읽어도 머리말을 찾는다."""
    # scripts/check_ingredient_ocr.py 로 돌린 실제 출력 ("르"→"로")
    text = (
        "원재료 및 함량: 감자(국산) 85%, 해바라기유 양파분말 새우분말(새우:베트님산) "
        "알레로기 유발물질: 새우 대두 함유"
    )
    sec = split_sections(text)
    assert split_terms(sec.allergen_notice) == ["새우", "대두"]
    assert "유발물질" not in sec.ingredients


def test_real_ocr_keeps_allergen_terms_in_merged_text():
    """실제 OCR 결과: 성분이 붙어도 알레르겐 표기는 남는다."""
    got = parse_ingredient_text(split_sections(EASYOCR_SCALE2).ingredients)
    # 쉼표가 빠져 붙은 덩어리 안에도 '전지분유', '탈지분유' 가 들어 있어야 판정에서 잡힌다
    joined = " | ".join(got)
    assert "전지분유" in joined and "탈지분유" in joined


# --- 줄 단위 입력 ------------------------------------------------------------


def test_parses_joined_lines():
    """줄을 이어 붙여 파싱한다."""
    got = parse_ingredients(lines("원재료명: 밀가루, 탈지분", "유, 설탕", "영양정보 100kcal"))
    # 줄바꿈에 걸린 "탈지분 유"는 판정 단계에서 공백을 지우고 비교한다
    assert got == ["밀가루", "탈지분 유", "설탕"]


def test_join_lines_uses_single_space():
    """`join_lines` 는 공백 하나로 잇는다."""
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


def test_crop_and_upscale_crops_and_enlarges():
    """`crop_and_upscale` 은 자르고 키운다."""
    image = np.zeros((100, 200, 3), np.uint8)
    out = crop_and_upscale(image, BBox(10, 20, 60, 40), scale=3.0)
    assert out.shape[:2] == (60, 150)


def test_crop_and_upscale_clips_outside_coords():
    """`crop_and_upscale` 은 이미지 밖 좌표를 안쪽으로 자른다."""
    image = np.zeros((100, 200, 3), np.uint8)
    out = crop_and_upscale(image, BBox(-50, -50, 50, 30), scale=1.0)
    assert out.shape[:2] == (30, 50)


def test_crop_and_upscale_caps_scale():
    """`crop_and_upscale` 은 너무 크게 키우지 않는다."""
    image = np.zeros((1080, 1920, 3), np.uint8)
    out = crop_and_upscale(image, BBox(0, 0, 1920, 1080), scale=3.0)
    assert max(out.shape[:2]) <= 2560


def test_crop_and_upscale_empty_region_raises():
    """`crop_and_upscale` 에 빈 영역을 주면 예외다."""
    with pytest.raises(ValueError):
        crop_and_upscale(np.zeros((100, 100, 3), np.uint8), BBox(150, 150, 200, 200))


def test_read_panel_groups_blocks_into_sorted_lines():
    """`read_panel` 은 덩어리를 줄로 묶고 위에서 아래로 정렬한다."""
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


def test_long_may_contain_list_starts_at_boundary():
    """혼입 목록이 40자를 넘어도 경계부터 잡는다."""
    sec = split_sections(f"원재료명 밀가루, 설탕 • {LONG_LIST} • 제품의 신선도 유지")
    assert sec.cross_contamination == LONG_LIST
    assert "땅콩" not in sec.ingredients
    assert "신선도" not in sec.cross_contamination  # 다음 • 에서 끝난다


def test_may_contain_list_after_report_number():
    """품목보고번호 뒤의 혼입 목록."""
    sec = split_sections("원재료명 밀가루 품목보고번호 19870415003381 우유, 대두, 땅콩 혼입가능")
    assert sec.cross_contamination == "우유, 대두, 땅콩 혼입가능"


def test_list_without_boundary_keeps_ingredients():
    """경계 없이 이어진 목록은 원재료를 혼입으로 낮추지 않는다."""
    # 원재료도 이름 목록처럼 보인다. 경계가 없으면 목록 전체를 혼입으로 보면 안 된다.
    sec = split_sections("원재료명 밀가루, 설탕, 버터 땅콩, 호두 혼입가능")
    assert "밀가루" in sec.ingredients
    assert "버터" in sec.ingredients


def test_starts_at_this_product_despite_parentheses():
    """괄호가 끼어도 "이 제품은"부터 잡는다."""
    sec = split_sections(
        "원재료명 밀가루 •이 제품은 땅콩, 조개류(굴, 전복, 홍합 포함)를 사용한 제품과 "
        "같은 제조 시설에서 제조하고 있습니다."
    )
    assert sec.cross_contamination.startswith("이 제품은 땅콩")


def test_contains_label_keeps_ingredient_header():
    """함유 표시가 원재료명 머리말을 삼키지 않는다."""
    sec = split_sections("원재료명 밀가루, 설탕, 정제 미강유 우유, 대두 함유")
    assert sec.has_ingredient_header
    assert sec.ingredients.startswith("밀가루")
    assert "정제 미강유" in sec.ingredients
    assert split_terms(sec.allergen_notice) == ["우유", "대두"]


def test_unclosed_parenthesis_keeps_ingredients_out_of_label():
    """괄호가 닫히지 않아도 원재료는 표시로 가져가지 않는다."""
    sec = split_sections("원재료명 코코아분말 1(인도네시아 물엿, 향료 2종 함유")
    assert sec.allergen_notice == ""
    assert "물엿" in sec.ingredients


def test_finds_header_with_cut_first_letter():
    """첫 글자가 잘린 머리말도 찾는다."""
    sec = split_sections("재료명 밀가루(밀 미국산), 백설탕 장재질 폴리프로필렌")
    assert sec.has_ingredient_header
    assert sec.ingredients == "밀가루(밀 미국산), 백설탕"


def test_finds_cut_header_misread_as_dae():
    """첫 글자가 잘리고 '재'를 '대'로 읽은 머리말도 찾는다 (label_01 실측)."""
    sec = split_sections("대료명 밀가루(밀 미국산) 백설탕 장재질 폴리프로필렌")
    assert sec.has_ingredient_header
    assert sec.ingredients == "밀가루(밀 미국산) 백설탕"


def test_ingredients_end_at_storage_notice():
    """보관 안내에서 원재료가 끝난다."""
    # "얼지 않게"의 '게'가 알레르겐(게)으로 잡히던 문제
    sec = split_sections("원재료명 정제수, 구연산. •직사광선을 피해 얼지 않게 보관")
    assert sec.ingredients == "정제수, 구연산"


# --- 실제 PaddleOCR 출력에서 찾은 것 (검수한 정답으로 다시 채점, 2026-10-14) ---------


def test_collects_names_from_contains_label_missing_commas():
    """쉼표가 빠진 함유 표시도 이름만 모은다."""
    sec = split_sections("원재료명 간장분말, 건청경채 닭고기 쇠고기 함유")
    assert split_terms(sec.allergen_notice) == ["닭고기", "쇠고기"]
    assert "건청경채" in sec.ingredients


def test_collects_list_with_some_commas_missing():
    """쉼표가 빠진 목록 중간에 쉼표가 섞여도 모은다."""
    sec = split_sections("원재료명 볶음참깨, 구운김 밀 대두 계란,우유,닭고기 쇠고기 함유")
    assert split_terms(sec.allergen_notice) == ["밀", "대두", "계란", "우유", "닭고기", "쇠고기"]
    assert "구운김" in sec.ingredients


def test_keeps_allergens_in_ingredients_without_header():
    """머리말이 없어도 원재료 속 알레르겐은 남는다."""
    # 머리말이 잘린 라면 성분표. 예전엔 원재료 글자를 통째로 버려서 간장(대두)·계란을 놓쳤다
    sec = split_sections("면/소맥분(밀:호주산), 간장분말, 계란스크램블 닭고기 쇠고기 함유")
    assert not sec.has_ingredient_header
    assert "간장분말" in sec.ingredients and "계란스크램블" in sec.ingredients
    assert split_terms(sec.allergen_notice) == ["닭고기", "쇠고기"]


def test_comma_read_as_period_keeps_may_contain_list():
    """쉼표를 마침표로 읽어도 혼입 목록이 끊기지 않는다."""
    sec = split_sections(
        "원재료명 정제수 품목보고번호 199102620031 ·대두,땅콩.밀.복숭아 혼입 가능."
    )
    assert sec.cross_contamination.endswith("대두,땅콩,밀,복숭아 혼입 가능")


def test_keeps_sentence_end_period():
    """문장 끝 마침표는 그대로 둔다."""
    sec = split_sections(
        "원재료명 밀가루. 이 제품은 땅콩을 사용한 제품과 같은 제조시설에서 제조합니다."
    )
    assert sec.ingredients == "밀가루"
    assert sec.cross_contamination.startswith("이 제품은 땅콩")


# --- '혼입' 글자 없는 알레르겐 목록 (#76) -----------------------------------

# label_03 OCR 원문을 줄인 것. '혼입 가능' 글자를 OCR 이 못 읽었다.
LABEL_03 = (
    "원재료명: 밀가루, 설탕, 버터(우유), 대두레시틴 우유,대두 함유 포장재질 플리에틸렌 "
    "품목보고번호 19870415003245 기,쇠고기,최고기,오징어 ·밀,달길, 땅몸, 호두복승아, 새우,거조 "
    "을위해질소충전"
)


def test_allergen_list_after_other_item_is_may_contain():
    """다른 항목 뒤의 알레르겐 목록은 혼입으로 잡는다."""
    sec = split_sections(LABEL_03)
    assert "포장재질" not in sec.ingredients and "쇠고기" not in sec.ingredients
    for name in ("쇠고기", "오징어", "밀", "새우"):
        assert name in sec.cross_contamination
    assert "질소충전" not in sec.cross_contamination  # 마지막 이름에서 끝낸다
    assert sec.allergen_notice == "우유,대두"


def test_single_name_is_not_a_list():
    """이름이 하나뿐이면 목록으로 보지 않는다."""
    sec = split_sections("원재료명: 밀가루, 설탕 보관방법: 우유와 함께 드세요, 직사광선을 피하세요")
    assert sec.cross_contamination == ""


def test_keeps_may_contain_phrase_with_list():
    """혼입 문구가 있으면 목록과 함께 남긴다."""
    text = (
        "원재료명: 밀가루 이 제품은 땅콩을 사용한 제품과 같은 제조시설에서 제조합니다. "
        "포장재질 폴리에틸렌 대두, 호두"
    )
    sec = split_sections(text)
    assert "땅콩" in sec.cross_contamination and sec.cross_contamination.endswith("대두, 호두")


# --- 표의 "원재료명" 칸이 내용 가운데에 있을 때 (label_05 새우깡) -------------------


def _box(x1, y1, x2, y2, text):
    return (BBox(x1, y1, x2, y2), text, 0.9)


def test_centered_header_moves_before_first_content_line():
    """제목 칸이 내용 여러 줄의 세로 가운데에 있으면, 제목 위 내용 줄도 원재료로 본다."""
    boxes = [
        _box(700, 210, 760, 240, "식품유형"), _box(780, 210, 900, 240, "과자"),
        _box(780, 245, 1200, 275, "쉬림프크래커(소맥분, 새우,"),
        _box(780, 275, 1200, 305, "혼합제제(말토덱스트린)),"),
        _box(700, 287, 760, 320, "원재료명"),
        _box(780, 302, 1200, 335, "미강유(태국산), 매운맛시즈닝,"),
        _box(780, 338, 1150, 365, "밀, 새우, 대두 함유"),
        _box(700, 367, 760, 397, "유통기한"), _box(780, 367, 900, 397, "후면 표기일까지"),
    ]  # fmt: skip
    text = join_lines(_group_lines(boxes))
    assert "과자 원재료명 쉬림프크래커" in text
    assert "소맥분" in split_sections(text).ingredients


def test_header_aligned_with_first_line_stays():
    """보통 표(제목이 내용 첫 줄과 나란함)는 그대로 둔다. 위 칸 내용을 끌어오지 않는다."""
    boxes = [
        _box(700, 210, 760, 240, "제품명"), _box(780, 210, 1100, 240, "새우깡 매운맛"),
        _box(780, 245, 1100, 275, "(스낵과자)"),
        _box(700, 280, 760, 310, "원재료명"), _box(780, 280, 1200, 310, "소맥분, 새우,"),
        _box(780, 315, 1200, 345, "미강유, 대두 함유"),
        _box(700, 350, 760, 380, "유통기한"), _box(780, 350, 900, 380, "후면 표기일까지"),
    ]  # fmt: skip
    text = join_lines(_group_lines(boxes))
    assert "(스낵과자) 원재료명 소맥분" in text
    assert "스낵과자" not in split_sections(text).ingredients
