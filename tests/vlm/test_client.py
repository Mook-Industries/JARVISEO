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
from jarviseo.vlm.client import encode_image


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


def test_ask_sends_photo_and_question_and_returns_answer():
    """`ask()` 는 사진 한 장과 질문을 보내고 답을 돌려준다."""
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


def test_sends_text_only_without_photo():
    """사진이 없으면 글자만 보낸다."""
    client = FakeClient()

    VLMClient(client=client).ask(None, "오늘 며칠이야?")

    assert sent(client)[1] == []


def test_memories_add_records_and_current_time():
    """기억 검색 결과가 있으면 기록과 지금 시각을 붙인다."""
    client = FakeClient()
    vlm = VLMClient(client=client)

    record = "질문: 이거 뭐야?\n대답: 책상 위에 있는 검은 백팩이에요."
    vlm.ask(frame(), "아까 본 가방 어디 있었지?", memories=[hit(record, 0.0)])
    text = sent(client)[0]
    assert "대화 기록" in text
    # 기록 하나는 한 줄로 편다. 지금 질문과 헷갈리지 않게 지금 질문에는 "지금"을 붙인다.
    assert " 질문: 이거 뭐야? / 대답: 책상 위에 있는 검은 백팩이에요.\n" in text
    assert "지금 시각" in text
    assert text.endswith("\n지금 질문: 아까 본 가방 어디 있었지?")

    # 찾았는데 없었으면 없다는 것을 알린다. 안 알리면 사진을 보고 지어낸다.
    vlm.ask(frame(), "아까 본 가방 어디 있었지?", memories=[])
    assert "찾은 기록 없음" in sent(client)[0]


def test_api_failure_returns_none_without_crash():
    """API 가 실패하면 None 을 주고 멈추지 않는다."""
    vlm = VLMClient(client=FakeClient(error=OpenAIError("연결 끊김")))

    assert vlm.ask(frame(), "이거 뭐야?") is None
    assert vlm.last_latency_ms is None


def test_empty_answer_returns_none():
    """답이 비었으면 None 이다."""
    assert VLMClient(client=FakeClient(text="  ")).ask(frame(), "이거 뭐야?") is None


def test_photo_long_side_shrinks_keeping_ratio():
    """사진은 긴 변을 줄이고 비율은 그대로 둔다."""

    def decode(url):
        jpeg = base64.b64decode(url.split(",", 1)[1])
        return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

    assert decode(encode_image(np.zeros((1080, 1920, 3), np.uint8), 960)).shape == (540, 960, 3)
    # 이미 작으면 키우지 않는다
    assert decode(encode_image(np.zeros((100, 200, 3), np.uint8), 960)).shape == (100, 200, 3)


def test_fake_matches_real_methods_and_attributes():
    """Fake 의 공개 메서드와 결과 속성이 실제와 같다."""
    for name, method in vars(VLMClient).items():
        if callable(method) and not name.startswith("_"):
            assert inspect.signature(getattr(FakeVLMClient, name)) == inspect.signature(method)

    def last_attrs(obj):
        return {name for name in vars(obj) if name.startswith("last_")}

    assert last_attrs(FakeVLMClient()) == last_attrs(VLMClient(client=object()))


def test_fake_returns_answers_in_order_and_records_questions():
    """Fake 는 답을 차례로 주고 받은 질문을 모은다."""
    vlm = FakeVLMClient(["컵이에요.", None, "책이에요."])

    assert vlm.ask(None, "하나") == "컵이에요."
    assert vlm.ask(None, "둘") is None  # 실패한 호출
    assert vlm.last_latency_ms is None
    assert vlm.ask(None, "셋") == "책이에요."
    assert vlm.ask(None, "넷") == "책이에요."  # 마지막 답을 계속 준다
    assert [question for _, question, _ in vlm.asked] == ["하나", "둘", "셋", "넷"]
