"""VLM 이 요청을 맞게 보내고, 실패해도 멈추지 않고, Fake 가 실제와 같은 모양인지 확인한다.

OpenAI API 는 부르지 않는다.
"""

import base64
import inspect
from types import SimpleNamespace

import cv2
import numpy as np
from openai import OpenAIError

from jarviseo import config
from jarviseo.types import Frame, MemoryHit
from jarviseo.vlm import FakeVLMClient, VLMClient
from jarviseo.vlm.client import SCENE_INSTRUCTIONS, encode_image


class FakeClient:
    """openai 클라이언트 대신 요청을 받아 두고 정해 둔 답을 준다. error 를 주면 그걸 던진다."""

    def __init__(self, text="빨간 머그컵이에요.", error=None):
        self.requests = []
        self.text, self.error = text, error
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_text=self.text)


def frame(height=1080, width=1920):
    return Frame(image=np.zeros((height, width, 3), np.uint8), timestamp=0.0, source_id="test")


def hit(text, observed_at=None):
    return MemoryHit(
        memory_id="1", kind="observation", text=text, score=0.4, observed_at=observed_at
    )


def sent(client):
    """마지막 요청의 (글자, 사진 목록)."""
    (message,) = client.requests[-1]["input"]
    texts = [c["text"] for c in message["content"] if c["type"] == "input_text"]
    images = [c for c in message["content"] if c["type"] == "input_image"]
    return texts[0], images


def test_ask_는_사진_한_장과_질문을_보내고_답을_돌려준다():
    client = FakeClient(text="  빨간 머그컵이에요.\n")
    vlm = VLMClient(client=client)

    answer = vlm.ask(frame(), "이거 뭐야?")

    assert answer == "빨간 머그컵이에요."  # 앞뒤 공백·줄바꿈은 지운다
    req = client.requests[0]
    assert req["model"] == config.VLM_MODEL
    assert req["reasoning"] == {"effort": "none"}
    text, images = sent(client)
    assert text == "이거 뭐야?"
    assert len(images) == 1
    assert images[0]["image_url"].startswith("data:image/jpeg;base64,")
    assert vlm.last_latency_ms >= 0


def test_사진이_없으면_글자만_보낸다():
    client = FakeClient()

    VLMClient(client=client).ask(None, "오늘 며칠이야?")

    assert sent(client)[1] == []


def test_기억_검색_결과가_있으면_기록과_지금_시각을_붙인다():
    client = FakeClient()
    vlm = VLMClient(client=client)

    vlm.ask(frame(), "아까 본 가방 어디 있었지?", memories=[hit("책상 위에 검은 백팩", 0.0)])
    text = sent(client)[0]
    assert "책상 위에 검은 백팩" in text
    assert "지금 시각" in text
    assert text.endswith("질문: 아까 본 가방 어디 있었지?")

    # 찾았는데 없었으면 없다는 것을 알린다. 안 알리면 사진을 보고 지어낸다.
    vlm.ask(frame(), "아까 본 가방 어디 있었지?", memories=[])
    assert "찾은 기록 없음" in sent(client)[0]


def test_api_가_실패하면_none_을_주고_멈추지_않는다():
    vlm = VLMClient(client=FakeClient(error=OpenAIError("연결 끊김")))

    assert vlm.ask(frame(), "이거 뭐야?") is None
    assert vlm.last_latency_ms is None


def test_답이_비었으면_none():
    assert VLMClient(client=FakeClient(text="  ")).ask(frame(), "이거 뭐야?") is None


def test_describe_는_장면_설명을_받고_지연_값은_건드리지_않는다():
    client = FakeClient(text="책상 위에 검은 백팩이 있다.")
    vlm = VLMClient(client=client)
    vlm.ask(frame(), "이거 뭐야?")
    asked_ms = vlm.last_latency_ms

    assert vlm.describe(frame()) == "책상 위에 검은 백팩이 있다."
    assert client.requests[-1]["instructions"] == SCENE_INSTRUCTIONS
    assert len(sent(client)[1]) == 1  # 사진도 같이 보냈다
    # 뒤에서 도는 describe 가 질문 지연을 덮으면 그래프가 엉뚱한 값을 기록한다
    assert vlm.last_latency_ms == asked_ms


def test_사진은_긴_변을_줄이고_비율은_그대로_둔다():
    def decode(url):
        jpeg = base64.b64decode(url.split(",", 1)[1])
        return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

    assert decode(encode_image(np.zeros((1080, 1920, 3), np.uint8), 960)).shape == (540, 960, 3)
    # 이미 작으면 키우지 않는다
    assert decode(encode_image(np.zeros((100, 200, 3), np.uint8), 960)).shape == (100, 200, 3)


def test_fake_의_공개_메서드와_결과_속성이_실제와_같다():
    for name, method in vars(VLMClient).items():
        if callable(method) and not name.startswith("_"):
            assert inspect.signature(getattr(FakeVLMClient, name)) == inspect.signature(method)

    def last_attrs(obj):
        return {name for name in vars(obj) if name.startswith("last_")}

    assert last_attrs(FakeVLMClient()) == last_attrs(VLMClient(client=object()))


def test_fake_는_답을_차례로_주고_받은_질문을_모은다():
    vlm = FakeVLMClient(["컵이에요.", None, "책이에요."])

    assert vlm.ask(None, "하나") == "컵이에요."
    assert vlm.ask(None, "둘") is None  # 실패한 호출
    assert vlm.last_latency_ms is None
    assert vlm.ask(None, "셋") == "책이에요."
    assert vlm.ask(None, "넷") == "책이에요."  # 마지막 답을 계속 준다
    assert [question for _, question, _ in vlm.asked] == ["하나", "둘", "셋", "넷"]
    assert vlm.describe(frame()) == "책상 위에 흰 컵이 있다."
