"""환경설정 '미량 혼입 가능성 알림'(api/trace.py) — 끄면 함유만 말한다."""

import pytest

from jarviseo.api.trace import DEFAULT_TRACE, TraceChecker, get_trace, set_trace
from jarviseo.memory import MemoryStore
from jarviseo.nutrition.response import OCCLUDED_NOTE, build_message
from jarviseo.nutrition.service import IngredientCheck, without_may_contain
from jarviseo.types import AllergenJudgement, AllergenVerdict


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(database_url=f"sqlite:///{tmp_path / 't.db'}")
    s.init_schema()
    return s


def result(contains, may, note=""):
    if contains:
        verdict = AllergenVerdict.CONTAINS
    else:
        verdict = AllergenVerdict.MAY_CONTAIN
    j = AllergenJudgement(verdict=verdict, matched_allergens=contains, may_contain_allergens=may)
    msg = build_message(j) + (" " + note if note else "")
    j.response_text = msg
    return IngredientCheck(judgement=j, message=msg)


def test_contains_and_may_keeps_only_contains():
    r = without_may_contain(result(["밀"], ["땅콩"]), ["밀", "땅콩"], {})
    assert r.message == "해당 제품에는 밀이 들어 있어요."
    assert (
        r.judgement.verdict is AllergenVerdict.CONTAINS and r.judgement.may_contain_allergens == []
    )


def test_may_only_becomes_not_detected_and_keeps_note():
    r = without_may_contain(result([], ["땅콩"], OCCLUDED_NOTE), ["땅콩"], {})
    assert r.judgement.verdict is AllergenVerdict.NOT_DETECTED
    assert "섞" not in r.message and r.message.endswith(OCCLUDED_NOTE)


def test_retake_and_plain_contains_untouched():
    plain = result(["밀"], [])
    assert without_may_contain(plain, ["밀"], {}) is plain
    retake = result([], ["땅콩"])
    retake.retake = "다시 보여 주세요."
    assert without_may_contain(retake, ["땅콩"], {}) is retake


def test_trace_is_stored_on_registered_allergens(store):
    user = store.ensure_user()
    assert get_trace(store, user) is DEFAULT_TRACE  # 등록한 알레르기가 없으면 받기
    store.set_user_allergens(user, ["밀", "땅콩"])
    set_trace(store, user, True)
    assert get_trace(store, user) is True
    set_trace(store, user, False)
    assert get_trace(store, user) is False


class Fixed:
    synonyms: dict = {}

    def __init__(self, r):
        self.r, self.kwargs = r, None

    def check(self, image, allergens, barcode_image=None):
        self.kwargs = {"barcode_image": barcode_image}
        return self.r


def test_trace_checker_follows_setting(store):
    user = store.ensure_user()
    store.set_user_allergens(user, ["밀", "땅콩"])
    inner = Fixed(result(["밀"], ["땅콩"]))
    checker = TraceChecker(inner, store, user)
    set_trace(store, user, True)
    assert "땅콩" in checker.check(None, ["밀", "땅콩"]).message
    set_trace(store, user, False)
    assert checker.check(None, ["밀", "땅콩"], barcode_image="whole").message == (
        "해당 제품에는 밀이 들어 있어요."
    )
    assert inner.kwargs == {"barcode_image": "whole"}  # 바코드용 사진은 그대로 넘긴다


def test_api_trace_and_allergen_change_keep_setting(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("langgraph")
    from fastapi.testclient import TestClient

    from jarviseo.api.assistant_web import WebAssistant
    from jarviseo.api.main import create_app
    from jarviseo.vlm.client import FakeVLMClient
    from tests.api.test_assistant_web import NoObjects, _NoChecker

    s = MemoryStore(database_url=f"sqlite:///{tmp_path / 'a.db'}")
    s.init_schema()
    user = s.ensure_user()
    web = WebAssistant(
        s, None, user, vlm=FakeVLMClient(["-"]), detector=NoObjects(), photo_dir=None
    )
    with TestClient(create_app(store=s, checker=_NoChecker(), assistant=web)) as c:
        c.put("/api/allergens", json={"allergens": ["밀"]})
        assert c.put("/api/trace", json={"on": False}).json() == {"on": False}
        c.put("/api/allergens", json={"allergens": ["밀", "땅콩"]})  # 새로 더해도
        assert c.get("/api/trace").json() == {"on": False}  # 설정은 그대로
        assert c.put("/api/trace", json={"on": True}).json() == {"on": True}
