"""성분 판정 결과가 DB 에 남는지 확인한다 (``MemoryStore.log_ingredient``).

SQLite 인메모리로 돌린다. 실행: pytest tests/test_ingredient_log.py
"""

import pytest

from jarviseo import config
from jarviseo.memory.models import SessionTurn, TurnIngredient
from jarviseo.memory.store import MemoryStore
from jarviseo.nutrition.allergen import judge_allergens, load_synonyms
from jarviseo.nutrition.barcode import BarcodeLookup
from jarviseo.nutrition.service import IngredientCheck
from jarviseo.types import (
    AllergenVerdict,
    AssistantResponse,
    BarcodeRead,
    IngredientSource,
    Intent,
    ProductInfo,
    Utterance,
)

CODE = "8801234567893"


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(database_url="sqlite://")
    s.init_schema()
    return s


def _turn(store: MemoryStore) -> int:
    session_id = store.start_session(store.ensure_user())
    utterance = Utterance(text="이거 먹어도 돼?", started_at=0.0, ended_at=1.0)
    response = AssistantResponse(text="", intent=Intent.INGREDIENT)
    return store.log_turn(session_id, utterance, response)


def _check(**kw) -> IngredientCheck:
    synonyms = load_synonyms(config.ALLERGEN_SYNONYMS_PATH)
    j = judge_allergens(["우유"], synonyms, ingredients="밀가루, 탈지분유, 설탕")
    j.source = IngredientSource.OCR
    base = dict(
        judgement=j,
        message="주의하세요. 우유가 들어 있어요.",
        ingredients=["밀가루", "탈지분유", "설탕"],
        ocr_text="원재료명: 밀가루, 탈지분유, 설탕",
        ocr_conf=0.83456,
        retake_count=1,
        timings_ms={"barcode": 40, "ocr": 900, "match": 2},
    )
    base.update(kw)
    return IngredientCheck(**base)


def _lookup(code: str) -> BarcodeLookup:
    read = BarcodeRead(code=code, symbology="EAN13", bbox=None)
    return BarcodeLookup(barcode=read, product=None, elapsed_ms=40, reason="not_found")


def test_judgement_is_attached_to_turn(store: MemoryStore):
    """판정 결과가 턴에 붙는다."""
    turn_id = _turn(store)
    store.log_ingredient(turn_id, _check())

    with store.session() as db:
        row = db.get(TurnIngredient, turn_id)
        assert row.verdict == AllergenVerdict.CONTAINS.value
        assert row.source == IngredientSource.OCR.value
        assert row.matched == ["탈지분유"]
        assert row.normalized == ["밀가루", "탈지분유", "설탕"]
        assert float(row.ocr_conf) == pytest.approx(0.835)
        assert row.retry_count == 1
        assert row.response_text.startswith("주의하세요")
        assert (row.ocr_ms, row.match_ms) == (900, 2)
        assert row.barcode is None
        assert db.get(SessionTurn, turn_id).ocr_text.startswith("원재료명")


def test_barcode_is_empty_when_product_not_in_db(store: MemoryStore):
    """외래키라서, 캐시에 없는 바코드를 적으면 Postgres 에서 저장이 깨진다."""
    turn_id = _turn(store)
    store.log_ingredient(turn_id, _check(barcode=_lookup(CODE)))
    with store.session() as db:
        assert db.get(TurnIngredient, turn_id).barcode is None


def test_barcode_is_kept_when_product_in_db(store: MemoryStore):
    """제품이 DB 에 있으면 바코드를 남긴다."""
    store.save_product(
        ProductInfo(
            barcode=CODE,
            product_name="테스트 과자",
            report_no="",
            raw_ingredients="밀가루, 탈지분유",
            source=IngredientSource.API,
        )
    )
    turn_id = _turn(store)
    store.log_ingredient(turn_id, _check(barcode=_lookup(CODE)))
    with store.session() as db:
        assert db.get(TurnIngredient, turn_id).barcode == CODE


def test_retake_overwrites_same_turn_record(store: MemoryStore):
    """재촬영하면 같은 턴의 기록을 덮어쓴다."""
    turn_id = _turn(store)
    store.log_ingredient(turn_id, _check(retake_count=0, ocr_conf=0.4))
    store.log_ingredient(turn_id, _check(retake_count=1, ocr_conf=0.9))
    with store.session() as db:
        rows = db.query(TurnIngredient).filter(TurnIngredient.turn_id == turn_id).all()
        assert len(rows) == 1
        assert rows[0].retry_count == 1


def test_saved_without_ocr_confidence(store: MemoryStore):
    """바코드로 찾은 경우엔 OCR 을 안 돌려서 신뢰도·원문이 없다."""
    turn_id = _turn(store)
    store.log_ingredient(turn_id, _check(ocr_conf=None, ocr_text=""))
    with store.session() as db:
        assert db.get(TurnIngredient, turn_id).ocr_conf is None
        assert db.get(SessionTurn, turn_id).ocr_text is None
