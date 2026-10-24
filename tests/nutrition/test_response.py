"""판정 결과 → 안내 문장 테스트.

실행: pytest tests/test_nutrition_response.py
"""

import pytest

from jarviseo.nutrition.allergen import user_labels
from jarviseo.nutrition.response import build_message, josa, warning_sentence
from jarviseo.types import AllergenJudgement
from jarviseo.types import AllergenVerdict as V


@pytest.mark.parametrize(
    ("word", "pair", "expected"),
    [
        ("우유", "이/가", "우유가"),
        ("밀", "이/가", "밀이"),
        ("땅콩", "을/를", "땅콩을"),
        ("새우", "을/를", "새우를"),
        ("대두", "은/는", "대두는"),
        ("호두", "과/와", "호두와"),
        ("탈지분유", "으로/로", "탈지분유로"),
        ("밀", "으로/로", "밀로"),  # ㄹ 받침은 '로'
        ("전란액", "으로/로", "전란액으로"),
        ("우유, 대두", "이/가", "우유, 대두가"),  # 마지막 낱말 기준
        ("ABC", "이/가", "ABC가"),  # 한글로 안 끝나면 받침 없는 쪽
    ],
)
def test_particles(word, pair, expected):
    """조사."""
    assert josa(word, pair) == expected


def J(verdict, **kw) -> AllergenJudgement:
    return AllergenJudgement(verdict=verdict, **kw)


def test_contains_is_one_sentence():
    """함유는 "해당 제품에는"으로 시작하는 한 문장으로 말하고 제품명·성분표 표기는 뺀다."""
    msg = build_message(
        J(V.CONTAINS, matched_allergens=["우유"], matched_terms=["탈지분유", "유청"]),
        product_name="초코쿠키",
    )
    assert msg == "해당 제품에는 우유가 들어 있어요."


def test_contains_and_may_contain_in_one_sentence():
    """함유와 혼입 가능을 한 문장으로 말한다."""
    msg = build_message(J(V.CONTAINS, matched_allergens=["밀"], may_contain_allergens=["새우"]))
    assert msg == "해당 제품에는 밀이 들어 있고, 새우는 같은 시설에서 써서 섞여 들어갔을 수 있어요."


def test_may_contain_does_not_say_contains():
    """혼입 가능만 있으면 '들어 있다'고 하지 않는다."""
    msg = build_message(J(V.MAY_CONTAIN, may_contain_allergens=["땅콩", "호두"]))
    assert msg == "해당 제품에는 같은 시설에서 쓰는 땅콩, 호두가 섞여 들어갔을 수 있어요."
    assert "들어 있" not in msg and "없어요" not in msg


def test_uses_names_user_registered():
    """사용자가 '계란'으로 등록했으면 법정 표시명 '알류' 대신 '계란'으로 말한다."""
    j = J(V.CONTAINS, matched_allergens=["밀", "알류"])
    assert build_message(j, labels={"알류": "계란"}) == "해당 제품에는 밀, 계란이 들어 있어요."


def test_user_labels_maps_canonical_to_registered_name():
    """판정용 표준 이름 → 사용자가 등록한 이름."""
    syn = {"알류": ["알류", "계란", "달걀"], "우유": ["우유", "탈지분유"]}
    assert user_labels(["계란", " 우유", "키위", "달걀"], syn) == {
        "알류": "계란",  # 같은 알레르겐을 두 번 적으면 처음 이름
        "우유": "우유",
        "키위": "키위",  # 사전에 없는 이름은 그대로
    }


def test_long_warning_shortens_wording_not_names():
    """50자를 넘으면 '같은 시설에서 써서 섞여 들어갔을'을 '섞였을'로 줄인다. 이름은 그대로다."""
    msg = warning_sentence(["밀", "대두"], ["새우", "땅콩"])
    assert msg == "해당 제품에는 밀, 대두가 들어 있고, 새우, 땅콩은 섞였을 수 있어요."


def test_never_drops_allergen_names():
    """이름이 많아 길어져도 알레르기는 하나도 빼지 않고 다 말한다."""
    contains = ["밀", "대두", "우유", "계란", "쇠고기"]
    may = ["새우", "땅콩", "게", "오징어", "잣", "메밀", "고등어"]
    msg = warning_sentence(contains, may)
    assert msg == (
        "해당 제품에는 밀, 대두, 우유, 계란, 쇠고기가 들어 있고, "
        "새우, 땅콩, 게, 오징어, 잣, 메밀, 고등어는 섞였을 수 있어요."
    )
    assert all(name in msg for name in contains + may) and "가지" not in msg


def test_short_warning_keeps_full_sentence():
    """50자 안이면 줄이지 않는다."""
    msg = warning_sentence(["밀"], ["새우"])
    assert msg == "해당 제품에는 밀이 들어 있고, 새우는 같은 시설에서 써서 섞여 들어갔을 수 있어요."


@pytest.mark.parametrize("verdict", [V.CONTAINS, V.MAY_CONTAIN])
def test_warning_is_single_sentence(verdict):
    """경고는 "해당 제품에는"으로 시작하는 한 문장이다."""
    j = J(verdict, matched_allergens=["우유"], may_contain_allergens=["땅콩"])
    msg = build_message(j, product_name="쿠키")
    assert msg.startswith("해당 제품에는 ") and msg.count(".") == 1 and msg.endswith(".")


def test_not_detected_is_not_called_safe():
    """미검출은 안전하다고 하지 않는다."""
    msg = build_message(J(V.NOT_DETECTED))
    assert "찾지 못했어요" in msg
    assert "포장지" in msg


@pytest.mark.parametrize(
    ("reason", "phrase"),
    [
        ("no_user_allergens", "알레르기를 등록해"),
        ("low_ocr_conf", "정확히 읽지 못했어요"),
        ("no_ingredients", "원재료명 부분을 보여 주세요"),
        ("opaque_ingredient", "알 수 없는 원료"),
        ("모르는사유", "확인하지 못했어요"),
    ],
)
def test_undetermined_explains_by_reason(reason, phrase):
    """확인 불가는 사유별로 안내한다."""
    assert phrase in build_message(J(V.UNDETERMINED, reason=reason))


@pytest.mark.parametrize("verdict", list(V))
def test_no_verdict_says_safe(verdict):
    """어떤 판정에도 안전하다는 말은 없다."""
    msg = build_message(J(verdict, matched_allergens=["우유"], may_contain_allergens=["땅콩"]))
    for banned in ("안전", "먹어도 돼", "드셔도"):
        assert banned not in msg
