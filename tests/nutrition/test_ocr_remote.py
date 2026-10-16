"""Colab OCR 서버 클라이언트 테스트. 네트워크·GPU 없이 돈다.

- 클라이언트는 httpx.MockTransport 로 가짜 응답을 준다.
- 서버 스크립트(scripts/colab_ocr_server.py)는 가짜 모델을 끼워 이 PC 의 127.0.0.1 에 띄운다.
"""

import importlib.util
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import numpy as np
import pytest

from jarviseo.nutrition.ocr import read_panel
from jarviseo.nutrition.ocr_remote import RemoteOcrError, RemoteOcrReader

IMAGE = np.full((60, 200, 3), 255, np.uint8)
BOX = [[0, 0], [100, 0], [100, 20], [0, 20]]


class FakeLocal:
    def __init__(self):
        self.calls = 0

    def readtext(self, image, **kwargs):
        self.calls += 1
        return [(BOX, "이 PC", 0.5)]


def remote(handler, **kw) -> RemoteOcrReader:
    return RemoteOcrReader("https://ocr.test", transport=httpx.MockTransport(handler), **kw)


def test_jpeg로_보내고_readtext_모양으로_돌려준다():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(path=request.url.path, type=request.headers["content-type"],
                    token=request.headers.get("x-token"), head=request.content[:2])  # fmt: skip
        return httpx.Response(200, json={"lines": [[BOX, "원재료명: 밀가루", 0.97]], "ms": 300})

    lines = remote(handler, token="abc").readtext(IMAGE)
    assert lines == [(BOX, "원재료명: 밀가루", 0.97)]
    assert seen == {"path": "/ocr", "type": "image/jpeg", "token": "abc", "head": b"\xff\xd8"}


def test_read_panel_에_그대로_끼울_수_있다():
    def handler(request):
        return httpx.Response(200, json={"lines": [[BOX, "원재료명: 밀가루", 0.97]]})

    lines = read_panel(IMAGE, reader=remote(handler))
    assert lines[0].text == "원재료명: 밀가루"


@pytest.mark.parametrize(
    "response",
    [
        httpx.ConnectError("down"),
        httpx.Response(502, text="bad gateway"),  # Colab 세션이 끊긴 터널
        httpx.Response(200, json={"error": "?"}),
    ],
)
def test_서버가_안_되면_이_pc에서_읽는다(response):
    def handler(request):
        if isinstance(response, Exception):
            raise response
        return response

    local = FakeLocal()
    made = []
    reader = remote(handler, fallback=lambda: made.append(1) or local)
    assert reader.readtext(IMAGE) == [(BOX, "이 PC", 0.5)]
    reader.readtext(IMAGE)
    assert local.calls == 2 and made == [1]  # 대체 OCR 은 한 번만 만든다


def test_대체_ocr이_없으면_예외():
    def handler(request):
        return httpx.Response(200, json={})

    with pytest.raises(RemoteOcrError):
        remote(handler).readtext(IMAGE)


# --- 서버 스크립트 (가짜 모델) ---------------------------------------------------


def load_server():
    path = Path(__file__).resolve().parents[2] / "scripts" / "colab_ocr_server.py"
    spec = importlib.util.spec_from_file_location("colab_ocr_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakePaddle:
    """PaddleOCR 3.x predict() 결과 모양."""

    def predict(self, image):
        h, w = image.shape[:2]
        poly = np.array([[0, 0], [w, 0], [w, 10], [0, 10]])
        return [{"rec_polys": [poly], "rec_texts": [f"{w}x{h}"], "rec_scores": [0.9]}]


@pytest.fixture
def server():
    mod = load_server()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), mod.make_handler(FakePaddle(), "cpu"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield mod, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_서버와_클라이언트가_맞물린다(server):
    mod, url = server
    reader = RemoteOcrReader(url, token=mod.TOKEN)
    try:
        assert reader.readtext(IMAGE) == [([[0, 0], [200, 0], [200, 10], [0, 10]], "200x60", 0.9)]
    finally:
        reader.close()
    assert httpx.get(f"{url}/health").json() == {"ok": True, "device": "cpu"}


def test_토큰이_틀리면_서버가_거절한다(server):
    _, url = server
    reader = RemoteOcrReader(url, token="wrong")
    with pytest.raises(httpx.HTTPStatusError) as e:
        reader.readtext(IMAGE)
    assert e.value.response.status_code == 401
