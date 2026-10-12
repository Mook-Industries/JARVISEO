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

SYNONYMS_PATH = Path(__file__).resolve().parents[1] / "data" / "datasets" / "allergen_synonyms.json"


@pytest.fixture(scope="module")
def synonyms() -> dict[str, list[str]]:
    return load_synonyms(SYNONYMS_PATH)


def write_json(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "syn.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


# --- 실제 사전 ---------------------------------------------------------------


def test_사전의_키는_표시대상_목록과_같다(synonyms):
    assert set(synonyms) == set(KNOWN_ALLERGENS)


def test_각_목록은_표준명으로_시작한다(synonyms):
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
def test_대표_표기가_맞는_알레르겐에_있다(synonyms, term, allergen):
    assert term in synonyms[allergen]


def test_짧은_표기가_다른_알레르겐의_긴_표기에_들어있는_경우(synonyms):
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


def test_메모용_키는_건너뛴다(tmp_path):
    path = write_json(tmp_path, {"_meta": {"설명": "x"}, "우유": ["탈지분유"]})
    assert load_synonyms(path) == {"우유": ["우유", "탈지분유"]}


def test_공백과_중복을_정리한다(tmp_path):
    path = write_json(tmp_path, {"우유": ["탈지 분유", "탈지분유", " ", "우유"]})
    assert load_synonyms(path) == {"우유": ["우유", "탈지분유"]}


def test_맥에서_만든_자모분리_한글도_같게_본다(tmp_path):
    nfd = unicodedata.normalize("NFD", "탈지분유")
    assert nfd != "탈지분유"
    path = write_json(tmp_path, {unicodedata.normalize("NFD", "우유"): [nfd]})
    assert load_synonyms(path) == {"우유": ["우유", "탈지분유"]}
    assert normalize_term(nfd) == "탈지분유"


def test_모르는_알레르겐_키는_예외(tmp_path):
    path = write_json(tmp_path, {"우우": ["탈지분유"]})  # 오타
    with pytest.raises(ValueError, match="알 수 없는 알레르겐"):
        load_synonyms(path)


def test_한_표기가_두_알레르겐에_있으면_예외(tmp_path):
    path = write_json(tmp_path, {"우유": ["버터"], "땅콩": ["버터"]})
    with pytest.raises(ValueError, match="중복"):
        load_synonyms(path)


@pytest.mark.parametrize("bad", ["탈지분유", ["탈지분유", 3], {"a": "b"}])
def test_값이_문자열_배열이_아니면_예외(tmp_path, bad):
    path = write_json(tmp_path, {"우유": bad})
    with pytest.raises(ValueError, match="문자열 배열"):
        load_synonyms(path)
