"""성분표 글자 → 구간 분리 → 성분 목록 테스트.

OCR 엔진 없이 돈다. OCR 이 읽어 냈다고 가정한 글자를 넣는다.

실행: pytest tests/test_nutrition_ocr.py
"""

import pytest

from jarviseo.nutrition.ocr import (
    join_lines,
    parse_ingredient_text,
    parse_ingredients,
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


def test_원재료명이_없으면_표시만_남긴다():
    sec = split_sections("설탕, 물엿, 젤라틴 영양정보")
    assert not sec.has_ingredient_header
    assert sec.ingredients == ""


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


# --- 줄 단위 입력 ------------------------------------------------------------


def test_줄을_이어_붙여_파싱한다():
    got = parse_ingredients(lines("원재료명: 밀가루, 탈지분", "유, 설탕", "영양정보 100kcal"))
    # 줄바꿈에 걸린 "탈지분 유"는 판정 단계에서 공백을 지우고 비교한다
    assert got == ["밀가루", "탈지분 유", "설탕"]


def test_join_lines_는_공백_하나로_잇는다():
    assert join_lines(lines("밀가루,  ", "  설탕")) == "밀가루, 설탕"
