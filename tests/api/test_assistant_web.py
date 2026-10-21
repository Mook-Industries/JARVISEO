"""웹에서 자비서 전체 흐름 — 팀 그래프를 가짜 VLM · 가짜 검출기로 돌린다.

CI 에는 fastapi 를 설치하지 않으므로(requirements-dev.txt) 로컬에서만 돈다.
"""

import base64

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")
from fastapi.testclient import TestClient  # noqa: E402

from jarviseo.api.assistant_web import WebAssistant  # noqa: E402
from jarviseo.api.main import create_app  # noqa: E402
from jarviseo.memory import MemoryStore  # noqa: E402
from jarviseo.vlm.client import FakeVLMClient  # noqa: E402


class NoObjects:
    """아무것도 못 찾는 검출기. 가리킴 질문이면 그래프가 되묻거나 사진 전체로 답한다."""

    def detect(self, image):
        return []


def photo() -> str:
    ok, buf = cv2.imencode(".jpg", np.full((120, 160, 3), 200, np.uint8))
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@pytest.fixture
def client(tmp_path):
    store = MemoryStore(database_url=f"sqlite:///{tmp_path / 'web.db'}")
    store.init_schema()
    user = store.ensure_user()
    vlm = FakeVLMClient(["컵이에요."])
    web = WebAssistant(
        store, checker=None, user_id=user, vlm=vlm, detector=NoObjects(), photo_dir=None
    )
    with TestClient(create_app(store=store, checker=_NoChecker(), assistant=web)) as c:
        yield c, vlm


class _NoChecker:
    synonyms: dict = {}

    def close(self):
        pass


def test_general_question_goes_through_graph_to_vlm(client):
    c, vlm = client
    d = c.post("/api/ask", json={"question": "오늘 기분 어때?"}).json()
    assert d["answer"] == "컵이에요." and d["spoken"] == ["컵이에요."]
    assert d["intent"] and d["turn_id"]  # 의도를 정하고 턴을 기록했다
    assert vlm.asked[-1][1] == "오늘 기분 어때?" and vlm.asked[-1][0] is None  # 사진 없이 물었다


def test_photo_reaches_vlm(client):
    """가리킴이 필요 없는 질문이면 사진 한 장이 그대로 VLM 에 간다."""
    c, vlm = client
    d = c.post("/api/ask", json={"question": "오늘 기분 어때?", "image": photo()}).json()
    assert d["answer"] == "컵이에요."
    assert vlm.asked[-1][0] is not None


def test_pointing_without_objects_does_not_guess(client):
    """'저거'라고 했는데 물체가 없으면 VLM 에 아무 사진이나 보내 지어내지 않는다."""
    c, vlm = client
    d = c.post("/api/ask", json={"question": "저거 뭐야?", "image": photo()}).json()
    assert d["spoken"] and d["target"] is None
    assert vlm.asked == []


def test_speech_timings_are_hidden(client):
    c, _ = client
    d = c.post("/api/ask", json={"question": "오늘 기분 어때?"}).json()
    assert not {"stt", "tts", "total"} & set(d["latency_ms"])


@pytest.mark.parametrize(
    "body", [{"question": "  "}, {"question": "뭐야?", "image": "bm90IGltYWdl"}]
)
def test_ask_rejects_bad_input(client, body):
    c, _ = client
    assert c.post("/api/ask", json=body).status_code == 400


def test_assistant_page_is_served(client):
    c, _ = client
    r = c.get("/assistant")
    assert r.status_code == 200 and "자비서에게 물어보기" in r.text


# --- 손에 든 제품: 성분 질문의 두 보완 (웹에서만) -----------------------------------


class CupOnTheLeft:
    """왼쪽 위 작은 컵 하나를 찾는 검출기. 크롭이 사진 전체보다 작아진다."""

    def detect(self, image):
        from jarviseo.types import BBox, Detection

        return [Detection(label="cup", confidence=0.9, bbox=BBox(10, 10, 60, 60))]


class PanelOnlyInWholePhoto:
    """사진 전체를 받아야 성분표를 읽는 판정기. 크롭이면 재촬영을 청한다."""

    def __init__(self, whole_shape):
        self.whole_shape, self.seen = whole_shape, []

    def check(self, image, allergens):
        from jarviseo.nutrition.service import IngredientCheck
        from jarviseo.types import AllergenJudgement, AllergenVerdict

        self.seen.append(image.shape)
        if image.shape != self.whole_shape:
            j = AllergenJudgement(verdict=AllergenVerdict.UNDETERMINED, reason="no_ingredients")
            msg = "성분표 전체가 보이도록 가리켜 주세요."
            return IngredientCheck(judgement=j, message=msg, retake=msg)
        j = AllergenJudgement(verdict=AllergenVerdict.CONTAINS)
        return IngredientCheck(judgement=j, message="해당 제품에는 밀이 들어 있어요.")


def web_with(tmp_path, detector, checker):
    store = MemoryStore(database_url=f"sqlite:///{tmp_path / 'web.db'}")
    store.init_schema()
    user = store.ensure_user()
    vlm = FakeVLMClient(["-"])
    return WebAssistant(store, checker, user, vlm=vlm, detector=detector, photo_dir=tmp_path)


IMAGE = np.full((120, 160, 3), 200, np.uint8)


def test_ingredient_reads_whole_photo_when_crop_has_no_panel(tmp_path):
    """가리킴 크롭에서 성분표를 못 읽으면 다시 찍어 달라기 전에 사진 전체로 한 번 더 읽는다."""
    checker = PanelOnlyInWholePhoto(IMAGE.shape)
    web = web_with(tmp_path, CupOnTheLeft(), checker)
    state, spoken = web.ask("이거 먹어도 돼?", IMAGE)
    assert state.get("crop") is not None  # 가리킴 → 크롭은 그대로 했다
    assert checker.seen[-1] == IMAGE.shape and len(checker.seen) == 2
    assert spoken == ["해당 제품에는 밀이 들어 있어요."] and web.retried_whole
    assert (tmp_path / f"turn{state['turn_id']}_crop.jpg").exists()  # 사진을 남겼다


def test_ingredient_uses_whole_photo_when_nothing_detected(tmp_path):
    """손에 든 과자처럼 물체를 못 찾으면 '모르겠어요' 대신 사진 전체를 대상으로 판정한다."""
    checker = PanelOnlyInWholePhoto(IMAGE.shape)
    web = web_with(tmp_path, NoObjects(), checker)
    state, spoken = web.ask("이거 먹어도 돼?", IMAGE)
    assert state["target"].chosen.detection.label == "사진 전체"
    assert spoken == ["해당 제품에는 밀이 들어 있어요."]


def test_other_questions_keep_plain_pointing(tmp_path):
    """성분 질문이 아니면 보완을 끼우지 않는다. 물체가 없으면 그래프대로 되묻는다."""
    vlm_web = web_with(tmp_path, NoObjects(), PanelOnlyInWholePhoto(IMAGE.shape))
    state, _ = vlm_web.ask("저거 뭐야?", IMAGE)
    assert state.get("crop") is None and not vlm_web.retried_whole
