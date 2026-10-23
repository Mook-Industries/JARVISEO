"""Google Vision OCR 클라이언트 테스트. 네트워크 없이 httpx.MockTransport 로 돈다."""

import json

import httpx
import numpy as np
import pytest

from jarviseo.nutrition import ocr, service
from jarviseo.nutrition.ocr import read_panel
from jarviseo.nutrition.ocr_google import GoogleVisionError, GoogleVisionReader

IMAGE = np.full((60, 200, 3), 255, np.uint8)


def sym(text, x, brk=None, conf=0.9):
    s = {
        "text": text,
        "confidence": conf,
        "boundingBox": {"vertices": [{"x": x, "y": 10}, {"x": x + 9, "y": 10},
                                     {"x": x + 9, "y": 20}, {"x": x, "y": 20}]},
    }  # fmt: skip
    if brk:
        s["property"] = {"detectedBreak": {"type": brk}}
    return s


# "원재료명: 밀" / "우유 함유" 두 줄
ANNOTATION = {
    "pages": [
        {
            "blocks": [
                {
                    "paragraphs": [
                        {
                            "words": [
                                {"symbols": [sym("원", 0), sym("재", 10), sym("료", 20),
                                             sym("명", 30), sym(":", 40, "SPACE")]},
                                {"symbols": [sym("밀", 50, "EOL_SURE_SPACE")]},
                                {"symbols": [sym("우", 0), sym("유", 10, "SPACE")]},
                                {"symbols": [sym("함", 20), sym("유", 30, conf=0.5)]},
                            ]
                        }
                    ]
                }
            ]
        }
    ]
}  # fmt: skip


def vision(handler, **kw) -> GoogleVisionReader:
    return GoogleVisionReader("test-key", transport=httpx.MockTransport(handler), **kw)


def ok(request):
    return httpx.Response(200, json={"responses": [{"fullTextAnnotation": ANNOTATION}]})


def test_groups_symbols_into_lines_in_readtext_shape():
    """글자 단위 응답을 줄로 묶어 easyocr.readtext 모양으로 돌려준다."""
    lines = vision(ok).readtext(IMAGE)
    assert [t for _, t, _ in lines] == ["원재료명: 밀", "우유 함유"]
    box, _, conf = lines[1]
    assert box == [[0, 10], [39, 10], [39, 20], [0, 20]]
    assert conf == pytest.approx((0.9 * 3 + 0.5) / 4)


def test_sends_jpeg_document_text_request_with_key():
    """키는 쿼리로, 사진은 JPEG base64 로 보내고 문서용 OCR 을 요청한다."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)["requests"][0]
        seen["key"] = request.url.params["key"]
        seen["feature"] = body["features"][0]["type"]
        seen["hint"] = body["imageContext"]["languageHints"]
        seen["head"] = body["image"]["content"][:4]
        return ok(request)

    vision(handler).readtext(IMAGE)
    assert seen == {"key": "test-key", "feature": "DOCUMENT_TEXT_DETECTION",
                    "hint": ["ko"], "head": "/9j/"}  # fmt: skip


def test_plugs_into_read_panel():
    lines = read_panel(IMAGE, reader=vision(ok))
    # 가짜 응답은 두 줄의 높이가 같아서 read_panel 이 한 줄로 합친다
    assert lines[0].text.startswith("원재료명: 밀")


class FakeLocal:
    def __init__(self):
        self.calls = 0

    def readtext(self, image, **kwargs):
        self.calls += 1
        return [([[0, 0], [1, 0], [1, 1], [0, 1]], "대체", 0.5)]


@pytest.mark.parametrize(
    "response",
    [
        httpx.ConnectError("down"),
        httpx.Response(400, json={"error": {"message": "API key not valid"}}),
        httpx.Response(200, json={"responses": [{"error": {"message": "quota"}}]}),
    ],
)
def test_falls_back_when_vision_fails(response):
    """키 오류 · 네트워크 · 할당량이면 대체 OCR 로 읽는다(한 번만 만든다)."""

    def handler(request):
        if isinstance(response, Exception):
            raise response
        return response

    local = FakeLocal()
    made = []
    reader = vision(handler, fallback=lambda: made.append(1) or local)
    assert reader.readtext(IMAGE)[0][1] == "대체"
    reader.readtext(IMAGE)
    assert local.calls == 2 and made == [1]


def test_raises_without_fallback():
    def handler(request):
        return httpx.Response(403, json={"error": {"message": "denied"}})

    with pytest.raises(GoogleVisionError):
        vision(handler).readtext(IMAGE)


def test_empty_image_gives_no_lines():
    def handler(request):
        return httpx.Response(200, json={"responses": [{}]})

    assert vision(handler).readtext(IMAGE) == []


def test_get_reader_prefers_google_when_key_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "k")
    monkeypatch.setattr(ocr, "_reader", None)
    try:
        assert isinstance(ocr._get_reader(), GoogleVisionReader)
    finally:
        ocr._reader = None


def test_checker_keeps_preprocessing_with_google(monkeypatch):
    """Vision 이어도 품질 검사 · 영역 잘라 읽기(여백) · 돌려 읽기 · 대비 보정 재처리를 켠다."""
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "k")
    monkeypatch.setattr(service, "make_food_qr_client", lambda: None)
    c = service.make_checker()
    assert c.quality_check is not None and c.auto_region and c.try_rotation
    assert c.max_reprocess >= 1 and c.region_pad == service.REGION_PAD
    assert service.make_checker(auto_region=False).auto_region is False


def test_checker_unchanged_without_google(monkeypatch):
    monkeypatch.delenv("GOOGLE_VISION_API_KEY", raising=False)
    monkeypatch.setattr(service, "make_food_qr_client", lambda: None)
    assert service.make_checker().quality_check is not None


def test_keeps_vision_reading_order_instead_of_regrouping():
    """나란한 두 칸이 같은 높이여도 Vision 이 준 순서(칸별)대로 둔다.

    대체 OCR 로 읽었으면 예전처럼 높이로 묶는다.
    """

    def handler(request):
        return httpx.Response(200, json={"responses": [{"fullTextAnnotation": ANNOTATION}]})

    reader = vision(handler)
    assert [line.text for line in read_panel(IMAGE, reader=reader)] == ["원재료명: 밀", "우유 함유"]

    local = FakeLocal()
    down = vision(lambda r: httpx.Response(500, json={"error": {"message": "x"}}),
                  fallback=lambda: local)  # fmt: skip
    read_panel(IMAGE, reader=down)
    assert down.last_ordered is False


def test_fixes_one_letter_typo_inside_allergen_list():
    """'대두,일,우유' 처럼 이름 목록 사이에 낀 한 글자 오타를 바로잡는다(일 → 밀)."""
    from jarviseo.nutrition.ocr import split_sections

    s = split_sections("원재료명: 밀가루, 설탕 대두,일,우유,계란,닭고기 함유")
    assert s.allergen_notice.replace(" ", "").startswith("대두,밀,우유")


def test_typo_fix_leaves_normal_ingredients_alone():
    """앞뒤가 알레르기 이름이 아니면 고치지 않는다(원재료 '일반'·'설탕' 등은 그대로)."""
    from jarviseo.nutrition.ocr import _fix_list_typos

    assert _fix_list_typos("설탕,일,소금") == "설탕,일,소금"
    assert _fix_list_typos("우유,일,대두") == "우유,밀,대두"
    assert _fix_list_typos("우유,잣,대두") == "우유,잣,대두"
