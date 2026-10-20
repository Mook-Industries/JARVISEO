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

from jarviseo.api.main import create_app  # noqa: E402
from jarviseo.memory import MemoryStore  # noqa: E402
from jarviseo.nutrition.barcode import BarcodeLookup  # noqa: E402
from jarviseo.nutrition.service import IngredientCheck  # noqa: E402
from jarviseo.types import (  # noqa: E402
    AllergenJudgement,
    AllergenVerdict,
    BarcodeRead,
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
