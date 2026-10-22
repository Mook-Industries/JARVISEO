"""성분 확인 API — 알레르기 조회·변경, 사진 올리기 → 판정 결과.

판정기는 가짜를 넣는다(OCR·네트워크 없이 돈다). DB 는 임시 SQLite 파일.
(인메모리 SQLite 는 스레드마다 다른 DB 가 되는데, API 는 요청을 다른 스레드에서 돌린다)
CI 에는 fastapi 를 설치하지 않으므로(requirements-dev.txt) 로컬에서만 돈다.
"""

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from jarviseo.api.main import barcode_steps, create_app, evidence, expand_allergens  # noqa: E402
from jarviseo.memory import MemoryStore  # noqa: E402
from jarviseo.nutrition.allergen import KNOWN_ALLERGENS  # noqa: E402
from jarviseo.nutrition.barcode import BarcodeLookup  # noqa: E402
from jarviseo.nutrition.service import IngredientCheck  # noqa: E402
from jarviseo.types import (  # noqa: E402
    AllergenJudgement,
    AllergenVerdict,
    BarcodeRead,
    IngredientSource,
    ProductInfo,
)


class FakeChecker:
    """check 에 들어온 알레르기를 남기고, 정해 둔 결과를 돌려준다."""

    def __init__(self, result: IngredientCheck):
        self.result = result
        self.calls: list[list[str]] = []
        self.closed = False

    def check(self, image, allergens):
        self.calls.append(list(allergens))
        return self.result

    def close(self):
        self.closed = True


def barcode_result() -> IngredientCheck:
    info = ProductInfo(barcode="8801234567893", product_name="라면", raw_ingredients="소맥분")
    read = BarcodeRead(code="8801234567893", symbology="EAN_13")
    return IngredientCheck(
        judgement=AllergenJudgement(verdict=AllergenVerdict.CONTAINS, product=info),
        message="해당 제품에는 밀이 들어 있어요.",
        barcode=BarcodeLookup(read, info),
        timings_ms={"barcode": 80, "match": 1},
    )


def jpeg() -> bytes:
    return cv2.imencode(".jpg", np.full((60, 80, 3), 255, np.uint8))[1].tobytes()


@pytest.fixture
def client_and_checker(tmp_path):
    store = MemoryStore(database_url=f"sqlite:///{tmp_path / 'api.db'}")
    checker = FakeChecker(barcode_result())
    with TestClient(create_app(store=store, checker=checker)) as client:
        yield client, checker
    assert checker.closed  # 서버가 꺼지면 푸드QR 연결을 닫는다


def test_allergens_put_then_get(client_and_checker):
    """알레르기를 저장하면 그대로 읽힌다. 빈 이름 · 같은 이름은 정리한다."""
    client, _ = client_and_checker
    r = client.put("/api/allergens", json={"allergens": ["계란", " 땅콩 ", "", "계란"]})
    assert sorted(r.json()["allergens"]) == ["계란", "땅콩"]
    assert sorted(client.get("/api/allergens").json()["allergens"]) == ["계란", "땅콩"]


def test_ingredient_returns_sentence_and_source(client_and_checker):
    """사진을 올리면 읽어 줄 문장과 판정 출처 · 제품 · 바코드를 돌려준다."""
    client, checker = client_and_checker
    r = client.post("/api/ingredient?allergens=밀,계란", content=jpeg())
    d = r.json()
    assert r.status_code == 200
    assert d["sentence"] == "해당 제품에는 밀이 들어 있어요."
    assert d["verdict"] == "contains" and d["source"] == "barcode"
    assert d["product"] == "라면" and d["barcode"] == "8801234567893"
    assert checker.calls == [["밀", "계란"]]


def test_ingredient_uses_registered_allergens_when_none_given(client_and_checker):
    """알레르기를 안 주면 등록된 알레르기로 판정한다."""
    client, checker = client_and_checker
    client.put("/api/allergens", json={"allergens": ["새우"]})
    client.post("/api/ingredient", content=jpeg())
    assert checker.calls == [["새우"]]


@pytest.mark.parametrize("body", [b"", b"not an image"])
def test_ingredient_rejects_bad_photo(client_and_checker, body):
    """사진이 없거나 사진이 아니면 400 이다."""
    client, checker = client_and_checker
    assert client.post("/api/ingredient", content=body).status_code == 400
    assert checker.calls == []


def test_ingredient_page_is_served(client_and_checker):
    """/ingredient 에서 사진 올리기 페이지를 준다."""
    client, _ = client_and_checker
    r = client.get("/ingredient")
    assert r.status_code == 200 and "성분 확인" in r.text


def test_ocr_status_without_remote_server(client_and_checker, monkeypatch):
    """Colab 주소를 안 적었으면 원래부터 이 PC 로 읽는 설정이라고 알려 준다."""
    from jarviseo.api import main

    monkeypatch.delenv(
        "GOOGLE_VISION_API_KEY", raising=False
    )  # .env 에 키가 있어도 Colab 경로를 본다
    monkeypatch.setattr(main.config, "OCR_REMOTE_URL", None)
    client, _ = client_and_checker
    assert client.get("/api/ocr-status").json()["remote"] is False


@pytest.mark.parametrize(
    ("health", "up"),
    [({"ok": True, "device": "gpu:0"}, True), (None, False)],  # None = 꺼짐(ngrok 안내 페이지)
)
def test_ocr_status_probes_remote_server(client_and_checker, monkeypatch, health, up):
    """Colab 서버 /health 를 물어 켜짐 · 꺼짐을 돌려준다. 꺼져 있으면 노트북 주소도 준다."""
    import httpx

    from jarviseo.api import main

    monkeypatch.delenv("GOOGLE_VISION_API_KEY", raising=False)

    def fake_get(url, timeout):
        if health is None:
            return httpx.Response(
                200, text="<html>offline</html>", headers={"content-type": "text/html"}
            )
        return httpx.Response(200, json=health)

    monkeypatch.setattr(main.config, "OCR_REMOTE_URL", "https://ocr.example")
    monkeypatch.setattr(main, "OCR_NOTEBOOK_URL", "https://colab.example/nb")
    monkeypatch.setattr(main.httpx, "get", fake_get)
    client, _ = client_and_checker
    d = client.get("/api/ocr-status").json()
    assert d["remote"] is True and d["up"] is up
    assert d["notebook_url"] == "https://colab.example/nb"


# --- 시험용 '모두' · 근거 · 바코드 단계 ----------------------------------------------


@pytest.mark.parametrize("word", ["모두", "테스트"])
def test_all_keyword_expands_to_every_known_allergen(word):
    names = expand_allergens([word, "키위"])
    assert names[: len(KNOWN_ALLERGENS)] == KNOWN_ALLERGENS and names[-1] == "키위"


def test_without_keyword_names_stay():
    assert expand_allergens(["계란", "땅콩"]) == ["계란", "땅콩"]


def test_ingredient_with_all_keyword_checks_every_allergen(client_and_checker):
    client, checker = client_and_checker
    client.post("/api/ingredient?allergens=모두", content=jpeg())
    assert checker.calls[-1] == KNOWN_ALLERGENS


def ocr_result(ingredients="", notice="", cross="") -> IngredientCheck:
    info = ProductInfo(barcode=None, product_name="", raw_ingredients=ingredients,
                       allergen_notice=notice, cross_contamination=cross,
                       source=IngredientSource.OCR)  # fmt: skip
    return IngredientCheck(
        judgement=AllergenJudgement(verdict=AllergenVerdict.CONTAINS, product=info),
        message="",
        barcode=BarcodeLookup(None, None, reason="no_barcode"),
        ocr_text="원재료명 ...",
    )


def test_evidence_shows_where_each_allergen_was_found():
    """알레르기마다 어느 구간에서 어떤 글자로 찾았는지, 못 찾은 것도 함께 준다."""
    synonyms = {"밀": ["밀", "소맥분"], "우유": ["우유", "탈지분유"], "알류": ["알류", "계란"]}
    r = ocr_result("소맥분, 설탕", "밀 함유", "이 제품은 계란을 사용한 제품과 같은 제조시설")
    rows = {e["allergen"]: e for e in evidence(r, ["밀", "계란", "우유"], synonyms)}
    assert rows["밀"]["status"] == "contains"
    assert rows["밀"]["where"] == [
        {"section": "원재료", "terms": ["소맥분"]},
        {"section": "알레르기 표시", "terms": ["밀"]},
    ]
    assert rows["계란"]["status"] == "may"  # 사용자가 쓴 이름으로 돌려준다
    assert rows["우유"] == {
        "allergen": "우유",
        "status": "not_found",
        "where": [],
        "ingredient_only": False,
    }
    assert not rows["밀"]["ingredient_only"]  # 함유 표시에도 있다


def test_barcode_steps_when_not_found_then_ocr():
    r = ocr_result("소맥분")
    r.barcode = BarcodeLookup(BarcodeRead("8801234567893", "EAN_13"), None, reason="not_found")
    assert barcode_steps(r) == [
        "바코드 8801234567893 읽음", "푸드QR에 제품 정보가 없음", "성분표를 읽음",
    ]  # fmt: skip


def test_barcode_steps_when_no_barcode():
    assert barcode_steps(ocr_result("소맥분")) == ["바코드를 찾지 못함", "성분표를 읽음"]


def test_barcode_steps_when_found():
    r = barcode_result()
    assert barcode_steps(r) == ["바코드 8801234567893 읽음", "푸드QR에서 제품을 찾음"]


def test_ingredient_response_has_evidence_and_steps(client_and_checker):
    client, _ = client_and_checker
    d = client.post("/api/ingredient?allergens=밀", content=jpeg()).json()
    # 가짜 판정기에는 동의어 사전이 없어서 '소맥분'을 밀로 보지 못한다. 칸이 채워지는지만 본다.
    assert d["evidence"] == [
        {"allergen": "밀", "status": "not_found", "where": [], "ingredient_only": False}
    ]
    assert d["barcode_steps"][0].startswith("바코드 8801234567893")


def test_evidence_flags_ingredient_only_hit_when_notice_was_read():
    """함유 표시가 읽혔는데 원재료에서만 찾은 이름은 판정은 함유 그대로, 표시만 따로 단다."""
    synonyms = {"밀": ["밀"], "우유": ["우유", "유당"]}
    r = ocr_result("코코아분밀, 유당", "우유 함유")  # '코코아분말'을 잘못 읽음
    rows = {e["allergen"]: e for e in evidence(r, ["밀", "우유"], synonyms)}
    assert rows["밀"]["status"] == "contains" and rows["밀"]["ingredient_only"]
    assert not rows["우유"]["ingredient_only"]


def test_evidence_without_notice_does_not_flag():
    """함유 표시를 못 읽었으면 원재료가 곧 근거라 따로 달지 않는다."""
    r = ocr_result("소맥분", "")
    rows = evidence(r, ["밀"], {"밀": ["밀", "소맥분"]})
    assert rows[0]["status"] == "contains" and not rows[0]["ingredient_only"]


def test_ocr_status_reports_google_vision(client_and_checker, monkeypatch):
    """Google Vision 키가 있으면 Colab 을 묻지 않고 켜짐(google-vision)으로 알린다."""
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "k")
    client, _ = client_and_checker
    body = client.get("/api/ocr-status").json()
    assert body["up"] is True and body["device"] == "google-vision"
