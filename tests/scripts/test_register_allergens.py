"""알레르기 등록 스크립트 — 목록 바꾸기 · 더하기 · 빼기 · 지우기 · 사전 대조.

SQLite 인메모리 DB 로 돈다. 등록한 것을 판정 쪽(get_user_allergens)이 그대로 읽는지 본다.
"""

import importlib.util
from pathlib import Path

import pytest

from jarviseo import config
from jarviseo.memory import MemoryStore
from jarviseo.nutrition.allergen import load_synonyms

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "register_allergens.py"
_spec = importlib.util.spec_from_file_location("register_allergens", _SCRIPT)
reg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reg)

SYN = load_synonyms(config.ALLERGEN_SYNONYMS_PATH)


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(database_url="sqlite://")
    s.init_schema()
    return s


def test_apply_replaces_list():
    """이름만 주면 목록을 통째로 바꾼다."""
    assert reg.apply(["우유"], ["계란", "땅콩"], add=False, remove=False, clear=False) == [
        "계란",
        "땅콩",
    ]


def test_apply_add_keeps_order_without_duplicates():
    """--add 는 지금 목록 뒤에 붙이고 같은 이름은 한 번만 둔다."""
    out = reg.apply(["계란"], ["땅콩", "계란"], add=True, remove=False, clear=False)
    assert out == ["계란", "땅콩"]


def test_apply_remove_and_clear():
    """--remove 는 그 이름만 빼고, --clear 는 모두 지운다."""
    assert reg.apply(["계란", "땅콩"], ["땅콩"], add=False, remove=True, clear=False) == ["계란"]
    assert reg.apply(["계란"], [], add=False, remove=False, clear=True) == []


def test_registered_allergens_are_read_by_checker(store):
    """등록한 이름 그대로 판정 쪽이 읽는다 (assistant.py 와 같은 owner 사용자)."""
    user_id = store.ensure_user()
    store.set_user_allergens(
        user_id, reg.apply([], ["계란", "땅콩"], add=False, remove=False, clear=False)
    )
    assert sorted(store.get_user_allergens(user_id)) == ["계란", "땅콩"]


def test_describe_shows_canonical_name():
    """'계란'처럼 다른 표기로 등록하면 표준명(알류)도 보여 준다."""
    assert reg.describe("계란", SYN) == "계란 (= 알류)"
    assert reg.describe("땅콩", SYN) == "땅콩"


def test_describe_unknown_name_is_searched_as_is():
    """사전에 없는 이름은 그 글자 그대로 찾는다고 알린다."""
    assert "사전에 없음" in reg.describe("키위", SYN)
