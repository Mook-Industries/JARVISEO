"""알레르기 동의어 사전 테스트.

사전은 판정의 유일한 근거다(LLM 미사용). 잘못된 사전이 조용히 읽히면
알레르겐을 놓치므로, 형식 오류는 전부 예외로 드러나야 한다.

실행: pytest tests/test_nutrition_allergen.py
"""

import json
import unicodedata
from pathlib import Path

import pytest

from jarviseo.nutrition.allergen import KNOWN_ALLERGENS, load_synonyms, normalize_term

SYNONYMS_PATH = Path(__file__).resolve().parents[2] / "data" / "datasets" / "allergen_synonyms.json"


@pytest.fixture(scope="module")
def synonyms() -> dict[str, list[str]]:
    return load_synonyms(SYNONYMS_PATH)


def write_json(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "syn.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


# --- 실제 사전 ---------------------------------------------------------------


def test_dictionary_keys_match_labeling_list(synonyms):
    """사전의 키는 표시대상 목록과 같다."""
    assert set(synonyms) == set(KNOWN_ALLERGENS)


def test_each_list_starts_with_standard_name(synonyms):
    """각 목록은 표준명으로 시작한다."""
    for name, terms in synonyms.items():
        assert terms[0] == name


@pytest.mark.parametrize(
    ("term", "allergen"),
    [
        ("탈지분유", "우유"),
        ("유청단백", "우유"),
        ("카제인나트륨", "우유"),
        ("난백", "알류"),
        ("소맥분", "밀"),
        ("분리대두단백", "대두"),
        ("굴", "조개류"),
        ("전복", "조개류"),
        ("홍합", "조개류"),
        ("메타중아황산나트륨", "아황산류"),
        ("돈육", "돼지고기"),
    ],
)
def test_common_terms_map_to_right_allergen(synonyms, term, allergen):
    """대표 표기가 맞는 알레르겐에 있다."""
    assert term in synonyms[allergen]


def test_short_term_inside_longer_term_of_other_allergen(synonyms):
    """단순 '포함' 검사로 찾으면 오탐이 나는 조합. 판정은 긴 표기부터 맞춰야 한다.

    예: "밀크"를 '밀'로, "땅콩버터"를 '우유(버터)'로 잘못 잡는다.
    사전에 표기를 추가하다 이 목록이 늘어나면 테스트가 깨진다.
    늘어난 조합이 판정에서 문제없는지 확인한 뒤 여기에 추가한다.
    """
    owner = {t: a for a, ts in synonyms.items() for t in ts}
    found = {(x, y) for x in owner for y in owner if x != y and x in y and owner[x] != owner[y]}
    assert found == {
        ("콩", "땅콩"),
        ("콩", "땅콩버터"),
        ("콩", "땅콩분태"),
        ("밀", "메밀"),
        ("밀", "메밀가루"),
        ("밀", "모밀"),
        ("밀", "밀크"),
        ("밀", "버터밀크"),
        ("밀가루", "메밀가루"),
        ("버터", "땅콩버터"),
    }


# --- load_synonyms 동작 --------------------------------------------------------


def test_skips_note_keys(tmp_path):
    """메모용 키는 건너뛴다."""
    path = write_json(tmp_path, {"_meta": {"설명": "x"}, "우유": ["탈지분유"]})
    assert load_synonyms(path) == {"우유": ["우유", "탈지분유"]}


def test_strips_spaces_and_duplicates(tmp_path):
    """공백과 중복을 정리한다."""
    path = write_json(tmp_path, {"우유": ["탈지 분유", "탈지분유", " ", "우유"]})
    assert load_synonyms(path) == {"우유": ["우유", "탈지분유"]}


def test_treats_mac_nfd_hangul_as_same(tmp_path):
    """맥에서 만든 자모 분리(NFD) 한글도 같게 본다."""
    nfd = unicodedata.normalize("NFD", "탈지분유")
    assert nfd != "탈지분유"
    path = write_json(tmp_path, {unicodedata.normalize("NFD", "우유"): [nfd]})
    assert load_synonyms(path) == {"우유": ["우유", "탈지분유"]}
    assert normalize_term(nfd) == "탈지분유"


def test_unknown_allergen_key_raises(tmp_path):
    """모르는 알레르겐 키는 예외다."""
    path = write_json(tmp_path, {"우우": ["탈지분유"]})  # 오타
    with pytest.raises(ValueError, match="알 수 없는 알레르겐"):
        load_synonyms(path)


def test_term_in_two_allergens_raises(tmp_path):
    """한 표기가 두 알레르겐에 있으면 예외다."""
    path = write_json(tmp_path, {"우유": ["버터"], "땅콩": ["버터"]})
    with pytest.raises(ValueError, match="중복"):
        load_synonyms(path)


@pytest.mark.parametrize("bad", ["탈지분유", ["탈지분유", 3], {"a": "b"}])
def test_non_string_list_value_raises(tmp_path, bad):
    """값이 문자열 배열이 아니면 예외다."""
    path = write_json(tmp_path, {"우유": bad})
    with pytest.raises(ValueError, match="문자열 배열"):
        load_synonyms(path)


# --- _제외: 알레르겐 표기를 품었지만 알레르겐이 아닌 성분 (#49 평가에서 발견) ---------


def test_locust_bean_gum_is_not_soybean(synonyms):
    """로커스트콩검은 대두가 아니다."""
    from jarviseo.nutrition.allergen import find_allergens

    assert "대두" not in find_allergens("카라기난, 로커스트콩검, 구아검", synonyms)
    # 같은 글자에 진짜 '콩'이 따로 있으면 잡는다
    assert "대두" in find_allergens("로커스트콩검, 콩", synonyms)


def test_exclusion_list_is_not_an_allergen_key(tmp_path):
    """제외 목록은 알레르겐 키로 섞이지 않는다."""
    path = write_json(tmp_path, {"_제외": ["로커스트콩검"], "대두": ["콩"]})
    syn = load_synonyms(path)
    assert syn == {"대두": ["대두", "콩"]}
    assert syn.not_allergen == ["로커스트콩검"]


def test_exclusion_equal_to_allergen_term_raises(tmp_path):
    """제외 표기가 알레르겐 표기와 같으면 예외다."""
    path = write_json(tmp_path, {"_제외": ["콩"], "대두": ["콩"]})
    with pytest.raises(ValueError, match="중복"):
        load_synonyms(path)
