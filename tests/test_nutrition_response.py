"""판정 결과 → 안내 문장 테스트.

실행: pytest tests/test_nutrition_response.py
"""

import pytest

from jarviseo.nutrition.response import build_message, josa
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
def test_조사(word, pair, expected):
    assert josa(word, pair) == expected


def J(verdict, **kw) -> AllergenJudgement:
    return AllergenJudgement(verdict=verdict, **kw)


def test_함유는_알레르겐과_근거_표기를_말한다():
    msg = build_message(
        J(V.CONTAINS, matched_allergens=["우유"], matched_terms=["탈지분유", "유청"]),
        product_name="초코쿠키",
    )
    assert (
        msg == "주의하세요. 초코쿠키에 우유가 들어 있어요. 성분표에 탈지분유, 유청으로 적혀 있어요."
    )


def test_표기가_알레르겐_이름과_같으면_근거는_생략한다():
    msg = build_message(J(V.CONTAINS, matched_allergens=["새우"], matched_terms=["새우"]))
    assert msg == "주의하세요. 새우가 들어 있어요."


def test_근거_표기는_세_개까지만():
    terms = ["탈지분유", "유청", "카제인", "버터", "치즈"]
    msg = build_message(J(V.CONTAINS, matched_allergens=["우유"], matched_terms=terms))
    assert "카제인" in msg and "버터" not in msg


def test_함유와_혼입_가능을_같이_말한다():
    msg = build_message(
        J(
            V.CONTAINS,
            matched_allergens=["밀"],
            matched_terms=["밀가루"],
            may_contain_allergens=["땅콩"],
        )
    )
    assert msg.endswith("땅콩은 같은 시설에서 써서 섞여 들어갔을 수 있어요.")


def test_혼입_가능():
    msg = build_message(J(V.MAY_CONTAIN, may_contain_allergens=["땅콩", "호두"]))
    assert "땅콩, 호두를 쓰는 시설" in msg
    assert msg.startswith("원재료에는 등록하신 알레르기 성분이 없어요.")


def test_미검출은_안전하다고_하지_않는다():
    msg = build_message(J(V.NOT_DETECTED))
    assert "찾지 못했어요" in msg
    assert "포장지" in msg


@pytest.mark.parametrize(
    ("reason", "phrase"),
    [
        ("no_user_allergens", "알레르기를 등록해"),
        ("low_ocr_conf", "정확히 읽지 못했어요"),
        ("no_ingredients", "원재료명을 찾지 못했어요"),
        ("opaque_ingredient", "알 수 없는 원료"),
        ("모르는사유", "확인하지 못했어요"),
    ],
)
def test_확인_불가는_사유별로_안내한다(reason, phrase):
    assert phrase in build_message(J(V.UNDETERMINED, reason=reason))


@pytest.mark.parametrize("verdict", list(V))
def test_어떤_판정에도_안전하다는_말은_없다(verdict):
    msg = build_message(J(verdict, matched_allergens=["우유"], may_contain_allergens=["땅콩"]))
    for banned in ("안전", "먹어도 돼", "드셔도"):
        assert banned not in msg
