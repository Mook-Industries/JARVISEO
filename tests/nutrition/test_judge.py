"""알레르기 판정 테스트. 실제 동의어 사전(data/datasets/allergen_synonyms.json)을 쓴다.

실행: pytest tests/test_nutrition_judge.py
"""

import pytest

from jarviseo import config
from jarviseo.nutrition.allergen import (
    find_allergens,
    judge_allergens,
    judge_product,
    load_synonyms,
)
from jarviseo.types import AllergenVerdict as V
from jarviseo.types import IngredientSource, ProductInfo


@pytest.fixture(scope="module")
def syn() -> dict[str, list[str]]:
    return load_synonyms(config.ALLERGEN_SYNONYMS_PATH)


# --- 표기 찾기 -----------------------------------------------------------------


def test_finds_by_synonym(syn):
    """동의어로 찾는다."""
    assert find_allergens("밀가루, 탈지분유, 설탕", syn) == {"밀": ["밀가루"], "우유": ["탈지분유"]}


def test_finds_without_spaces_or_commas(syn):
    """공백과 쉼표가 없어도 찾는다."""
    # OCR 이 쉼표를 빼먹고 붙여 낸 실제 모양
    assert "새우" in find_allergens("해바라기유 양파분말새우분말", syn)
    assert find_allergens("탈지 분유", syn) == {"우유": ["탈지분유"]}


@pytest.mark.parametrize(
    ("text", "expected", "not_expected"),
    [
        ("밀크초콜릿", "우유", "밀"),  # 밀크 ⊃ 밀
        ("버터밀크", "우유", "밀"),
        ("땅콩버터", "땅콩", "우유"),  # 땅콩버터 ⊃ 버터
        ("땅콩분태", "땅콩", "대두"),  # 땅콩 ⊃ 콩
        ("메밀가루", "메밀", "밀"),  # 메밀가루 ⊃ 밀가루
    ],
)
def test_longer_term_first_avoids_short_false_positive(syn, text, expected, not_expected):
    """긴 표기가 먼저라 짧은 표기 오탐이 없다."""
    found = find_allergens(text, syn)
    assert expected in found
    assert not_expected not in found


def test_catches_short_term_elsewhere(syn):
    """다른 자리에 있으면 짧은 표기도 잡는다."""
    # 밀크 자리는 우유가 가져가도, 따로 적힌 밀가루는 밀이다
    assert find_allergens("밀크초콜릿, 밀가루", syn) == {"우유": ["밀크"], "밀": ["밀가루"]}


# --- 판정 ---------------------------------------------------------------------


def test_in_ingredients_is_contains(syn):
    """원재료에 있으면 함유다."""
    j = judge_allergens(["우유", "땅콩"], syn, ingredients="밀가루, 탈지분유, 유청")
    assert j.verdict is V.CONTAINS
    assert j.matched_allergens == ["우유"]
    assert j.matched_terms == ["탈지분유", "유청"]


def test_only_in_allergen_label_is_contains(syn):
    """알레르기 표시에만 있어도 함유다."""
    j = judge_allergens(["대두"], syn, ingredients="설탕, 소금", notice="우유, 대두 함유")
    assert j.verdict is V.CONTAINS and j.matched_allergens == ["대두"]


def test_only_same_facility_is_may_contain(syn):
    """같은 제조시설 문구에만 있으면 혼입 가능이다."""
    j = judge_allergens(
        ["땅콩"],
        syn,
        ingredients="밀가루, 설탕",
        cross="이 제품은 땅콩을 사용한 제품과 같은 제조시설",
    )
    assert j.verdict is V.MAY_CONTAIN
    assert j.may_contain_allergens == ["땅콩"] and j.matched_allergens == []


def test_contains_wins_over_may_contain_for_same_allergen(syn):
    """함유와 혼입이 같은 알레르겐이면 함유로만 본다."""
    j = judge_allergens(["우유", "땅콩"], syn, ingredients="탈지분유", cross="우유, 땅콩 사용 시설")
    assert j.verdict is V.CONTAINS
    assert j.matched_allergens == ["우유"]
    assert j.may_contain_allergens == ["땅콩"]  # 우유는 함유에만


def test_not_found_is_not_detected(syn):
    """못 찾으면 미검출이다."""
    j = judge_allergens(["땅콩"], syn, ingredients="밀가루, 설탕, 탈지분유")
    assert j.verdict is V.NOT_DETECTED and j.reason == ""


def test_no_registered_allergen_is_undetermined(syn):
    """등록 알레르기가 없으면 확인 불가다."""
    j = judge_allergens([], syn, ingredients="탈지분유")
    assert j.verdict is V.UNDETERMINED and j.reason == "no_user_allergens"


def test_unreadable_is_undetermined(syn):
    """판독 불가면 확인 불가다."""
    j = judge_allergens(["땅콩"], syn, ingredients="밀가루", readable=False, reason="low_ocr_conf")
    assert j.verdict is V.UNDETERMINED and j.reason == "low_ocr_conf"


def test_unreadable_with_may_contain_is_undetermined(syn):
    """판독 불가면 혼입 문장을 찾았어도 확인 불가다."""
    # '혼입 가능'으로 끝내면 "원재료에는 없어요"라고 말하게 되는데 원재료를 못 읽었다
    j = judge_allergens(
        ["땅콩"], syn, ingredients="밀가루", cross="같은 제조시설에서 땅콩 사용", readable=False
    )
    assert j.verdict is V.UNDETERMINED
    assert j.may_contain_allergens == ["땅콩"]  # 정보는 남긴다


def test_unreadable_but_found_contains_is_contains(syn):
    """판독 불가여도 함유를 찾았으면 함유다."""
    # 경고는 확실하므로 판독 품질과 상관없이 낸다
    j = judge_allergens(["우유"], syn, ingredients="탈지분유", readable=False)
    assert j.verdict is V.CONTAINS


def test_no_ingredient_text_is_undetermined(syn):
    """원재료 글자가 없으면 확인 불가다."""
    j = judge_allergens(["땅콩"], syn, ingredients="  ", cross="같은 시설에서 우유 사용")
    assert j.verdict is V.UNDETERMINED and j.reason == "no_ingredients"


def test_unknown_compound_ingredient_is_undetermined(syn):
    """내용 모를 원료가 있으면 확인 불가다."""
    j = judge_allergens(["땅콩"], syn, ingredients="밀가루, 기타가공품, 설탕")
    assert j.verdict is V.UNDETERMINED and j.reason == "opaque_ingredient"
    # 괄호로 내용이 적혀 있으면 읽을 수 있다
    j = judge_allergens(["땅콩"], syn, ingredients="밀가루, 혼합제제(설탕, 소금)")
    assert j.verdict is V.NOT_DETECTED


def test_registered_synonym_maps_to_standard_name(syn):
    """사전의 다른 표기로 등록해도 표준명으로 본다."""
    j = judge_allergens(["계란"], syn, ingredients="밀가루, 전란액")
    assert j.verdict is V.CONTAINS and j.matched_allergens == ["알류"]


def test_finds_unlisted_allergen_by_name(syn):
    """사전에 없는 알레르기도 이름으로 찾는다."""
    j = judge_allergens(["키위"], syn, ingredients="설탕, 키위농축액")
    assert j.verdict is V.CONTAINS and j.matched_allergens == ["키위"]


def test_judges_barcode_lookup_result(syn):
    """바코드 조회 결과로 판정한다."""
    info = ProductInfo(
        barcode="8801234567893",
        product_name="테스트 과자",
        raw_ingredients="밀가루, 설탕, 버터",
        source=IngredientSource.CACHE,
    )
    j = judge_product(info, ["우유"], syn)
    assert j.verdict is V.CONTAINS and j.matched_terms == ["버터"]
    assert j.product is info and j.source is IngredientSource.CACHE
