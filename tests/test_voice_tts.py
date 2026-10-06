"""TTS 가 요청을 맞게 보내고, 읽을 글자를 맞게 다듬는지 확인한다. OpenAI API 는 부르지 않는다."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from jarviseo import config
from jarviseo.voice.tts import CHUNK_BYTES, TextToSpeech
from jarviseo.voice.tts_text import normalize, split_sentences


class FakeClient:
    """openai 클라이언트 대신 요청을 받아 두고, PCM 조각 두 개를 흘려준다."""

    def __init__(self):
        self.requests = []
        speech = SimpleNamespace(with_streaming_response=SimpleNamespace(create=self._create))
        self.audio = SimpleNamespace(speech=speech)

    @contextmanager
    def _create(self, **kwargs):
        self.requests.append(kwargs)
        yield SimpleNamespace(iter_bytes=lambda size: iter([b"\x01" * size, b"\x02" * size]))


def test_synthesize_는_다듬은_글자를_pcm_으로_요청하고_조각을_그대로_낸다():
    client = FakeClient()
    tts = TextToSpeech(voice="alloy", client=client)

    stream = tts.synthesize("나트륨 1,200mg 이에요.")
    assert tts.last_latency_ms is None  # 첫 조각을 받기 전
    chunks = list(stream)

    (req,) = client.requests
    assert (req["model"], req["voice"], req["response_format"]) == (
        config.TTS_MODEL,
        "alloy",
        "pcm",
    )
    assert req["input"] == "나트륨 1200밀리그램 이에요."
    assert chunks == [b"\x01" * CHUNK_BYTES, b"\x02" * CHUNK_BYTES]
    assert tts.last_latency_ms >= 0


@pytest.mark.parametrize(
    ("speed", "value"),
    [("SLOW", 0.85), ("FAST", 1.2), ("fast", 1.2), ("NORMAL", None), ("모르는값", None)],
)
def test_tts_speed_를_speed_값으로_바꿔_보낸다(speed, value):
    client = FakeClient()

    list(TextToSpeech(speed=speed, client=client).synthesize("안녕하세요."))

    assert client.requests[0].get("speed") == value  # NORMAL 은 아예 안 보낸다


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("열량 250kcal, 하루 권장량의 60%", "열량 250킬로칼로리, 하루 권장량의 60퍼센트"),
        ("탄수화물 3.5g이 들어 있어요.", "탄수화물 3.5그램이 들어 있어요."),
        ("3~5개, 1,234,567원", "3에서 5개, 1234567원"),
        ("**우유**와 `대두`\n- 땅콩", "우유와 대두\n땅콩"),
        ("5m 앞, 5min 뒤, 3,4번째", "5미터 앞, 5min 뒤, 3,4번째"),
    ],
)
def test_단위·쉼표·물결표를_읽는_말로_바꾸고_마크다운을_뺀다(raw, want):
    assert normalize(raw) == want


def test_문장_끝과_줄바꿈에서_자르고_소수점은_안_자른다():
    text = "우유가 들어 있어요. 나트륨은 3.5그램이에요! 괜찮을까요?\n네"

    assert split_sentences(text) == [
        "우유가 들어 있어요.",
        "나트륨은 3.5그램이에요!",
        "괜찮을까요?",
        "네",
    ]
    assert split_sentences("") == []
