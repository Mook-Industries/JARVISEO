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


def test_동의어로_찾는다(syn):
    assert find_allergens("밀가루, 탈지분유, 설탕", syn) == {"밀": ["밀가루"], "우유": ["탈지분유"]}


def test_공백과_쉼표가_없어도_찾는다(syn):
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
def test_긴_표기가_먼저라_짧은_표기_오탐이_없다(syn, text, expected, not_expected):
    found = find_allergens(text, syn)
    assert expected in found
    assert not_expected not in found


def test_다른_자리에_있으면_짧은_표기도_잡는다(syn):
    # 밀크 자리는 우유가 가져가도, 따로 적힌 밀가루는 밀이다
    assert find_allergens("밀크초콜릿, 밀가루", syn) == {"우유": ["밀크"], "밀": ["밀가루"]}


# --- 판정 ---------------------------------------------------------------------


def test_원재료에_있으면_함유(syn):
    j = judge_allergens(["우유", "땅콩"], syn, ingredients="밀가루, 탈지분유, 유청")
    assert j.verdict is V.CONTAINS
    assert j.matched_allergens == ["우유"]
    assert j.matched_terms == ["탈지분유", "유청"]


def test_알레르기_표시에만_있어도_함유(syn):
    j = judge_allergens(["대두"], syn, ingredients="설탕, 소금", notice="우유, 대두 함유")
    assert j.verdict is V.CONTAINS and j.matched_allergens == ["대두"]


def test_같은_제조시설에만_있으면_혼입_가능(syn):
    j = judge_allergens(
        ["땅콩"],
        syn,
        ingredients="밀가루, 설탕",
        cross="이 제품은 땅콩을 사용한 제품과 같은 제조시설",
    )
    assert j.verdict is V.MAY_CONTAIN
    assert j.may_contain_allergens == ["땅콩"] and j.matched_allergens == []


def test_함유와_혼입이_같은_알레르겐이면_함유로만(syn):
    j = judge_allergens(["우유", "땅콩"], syn, ingredients="탈지분유", cross="우유, 땅콩 사용 시설")
    assert j.verdict is V.CONTAINS
    assert j.matched_allergens == ["우유"]
    assert j.may_contain_allergens == ["땅콩"]  # 우유는 함유에만


def test_못_찾으면_미검출(syn):
    j = judge_allergens(["땅콩"], syn, ingredients="밀가루, 설탕, 탈지분유")
    assert j.verdict is V.NOT_DETECTED and j.reason == ""


def test_등록_알레르기가_없으면_확인_불가(syn):
    j = judge_allergens([], syn, ingredients="탈지분유")
    assert j.verdict is V.UNDETERMINED and j.reason == "no_user_allergens"


def test_판독_불가면_확인_불가(syn):
    j = judge_allergens(["땅콩"], syn, ingredients="밀가루", readable=False, reason="low_ocr_conf")
    assert j.verdict is V.UNDETERMINED and j.reason == "low_ocr_conf"


def test_판독_불가면_혼입_문장을_찾았어도_확인_불가(syn):
    # '혼입 가능'으로 끝내면 "원재료에는 없어요"라고 말하게 되는데 원재료를 못 읽었다
    j = judge_allergens(
        ["땅콩"], syn, ingredients="밀가루", cross="같은 제조시설에서 땅콩 사용", readable=False
    )
    assert j.verdict is V.UNDETERMINED
    assert j.may_contain_allergens == ["땅콩"]  # 정보는 남긴다


def test_판독_불가여도_함유를_찾았으면_함유(syn):
    # 경고는 확실하므로 판독 품질과 상관없이 낸다
    j = judge_allergens(["우유"], syn, ingredients="탈지분유", readable=False)
    assert j.verdict is V.CONTAINS


def test_원재료_글자가_없으면_확인_불가(syn):
    j = judge_allergens(["땅콩"], syn, ingredients="  ", cross="같은 시설에서 우유 사용")
    assert j.verdict is V.UNDETERMINED and j.reason == "no_ingredients"


def test_내용_모를_원료가_있으면_확인_불가(syn):
    j = judge_allergens(["땅콩"], syn, ingredients="밀가루, 기타가공품, 설탕")
    assert j.verdict is V.UNDETERMINED and j.reason == "opaque_ingredient"
    # 괄호로 내용이 적혀 있으면 읽을 수 있다
    j = judge_allergens(["땅콩"], syn, ingredients="밀가루, 혼합제제(설탕, 소금)")
    assert j.verdict is V.NOT_DETECTED


def test_사전의_다른_표기로_등록해도_표준명으로_본다(syn):
    j = judge_allergens(["계란"], syn, ingredients="밀가루, 전란액")
    assert j.verdict is V.CONTAINS and j.matched_allergens == ["알류"]


def test_사전에_없는_알레르기도_이름으로_찾는다(syn):
    j = judge_allergens(["키위"], syn, ingredients="설탕, 키위농축액")
    assert j.verdict is V.CONTAINS and j.matched_allergens == ["키위"]


def test_바코드_조회_결과_판정(syn):
    info = ProductInfo(
        barcode="8801234567893",
        product_name="테스트 과자",
        raw_ingredients="밀가루, 설탕, 버터",
        source=IngredientSource.CACHE,
    )
    j = judge_product(info, ["우유"], syn)
    assert j.verdict is V.CONTAINS and j.matched_terms == ["버터"]
    assert j.product is info and j.source is IngredientSource.CACHE
