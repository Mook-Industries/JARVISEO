"""알레르기 목록의 "모두" — 표시 대상 알레르기 전부를 확인한다(판정기에서 펼친다)."""

from types import SimpleNamespace

import numpy as np

from jarviseo.nutrition import service
from jarviseo.nutrition.allergen import KNOWN_ALLERGENS, expand_allergens


def test_all_keyword_expands_to_every_known_allergen():
    assert expand_allergens(["모두"]) == KNOWN_ALLERGENS


def test_other_names_stay_after_known_ones():
    names = expand_allergens(["키위", " 모두 ", "땅콩"])
    assert names[: len(KNOWN_ALLERGENS)] == KNOWN_ALLERGENS
    assert names[len(KNOWN_ALLERGENS) :] == ["키위"]  # 땅콩은 이미 목록에 있다


def test_without_keyword_list_is_unchanged():
    assert expand_allergens(["계란", "땅콩"]) == ["계란", "땅콩"]


def test_checker_judges_every_allergen_when_all_registered(monkeypatch):
    """음성 · 대시보드 질문은 그래프가 등록 목록을 그대로 넘긴다. 판정기가 "모두"를 펼쳐야 한다."""
    seen = {}

    def fake_ocr(self, crop, lookup, allergens, *args):
        seen["allergens"] = allergens
        return "checked"

    no_barcode = SimpleNamespace(barcode=None, hit=False)  # 바코드를 못 찾아 성분표를 읽는 길
    monkeypatch.setattr(service, "lookup_product", lambda *a, **k: no_barcode)
    monkeypatch.setattr(service.IngredientChecker, "_from_ocr", fake_ocr)
    checker = service.IngredientChecker.__new__(service.IngredientChecker)
    checker.store = checker.food_qr = None
    assert checker.check(np.zeros((8, 8, 3), np.uint8), ["모두"]) == "checked"
    assert seen["allergens"] == KNOWN_ALLERGENS
