"""문장 임베딩 — 실제 클래스는 가짜 API 로, Fake 는 그대로 확인한다."""

import inspect
import math
from types import SimpleNamespace

from jarviseo.memory import FakeEmbedder, TextEmbedder
from jarviseo.memory.embed import DIM


class _FakeEmbeddingsAPI:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5] * DIM)])


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_fake_public_methods_match_real():
    """Fake 의 공개 메서드가 실제와 같다."""
    for name, method in vars(TextEmbedder).items():
        if callable(method) and not name.startswith("_"):
            assert inspect.signature(getattr(FakeEmbedder, name)) == inspect.signature(method), name


def test_fake_has_same_result_attributes():
    """Fake 도 실제와 같은 결과 속성을 가진다."""

    def last_attrs(obj):
        return {name for name in vars(obj) if name.startswith("last_")}

    assert last_attrs(FakeEmbedder()) == last_attrs(TextEmbedder(client=object()))


def test_real_embedder_sends_one_sentence_with_configured_model():
    """실제 임베더는 설정한 모델로 한 문장을 보낸다."""
    api = _FakeEmbeddingsAPI()
    embedder = TextEmbedder(model="text-embedding-3-small", client=SimpleNamespace(embeddings=api))

    vector = embedder.embed("책상 위 검은 백팩")

    assert api.calls == [{"model": "text-embedding-3-small", "input": "책상 위 검은 백팩"}]
    assert len(vector) == DIM
    assert embedder.last_latency_ms is not None


def test_fake_gives_same_unit_vector_for_same_text():
    """Fake 는 같은 문장에 같은 길이 1 벡터를 준다."""
    a, b = FakeEmbedder().embed("책상 위 검은 백팩"), FakeEmbedder().embed("책상 위 검은 백팩")

    assert a == b
    assert len(a) == DIM
    assert math.isclose(_cos(a, a), 1.0)


def test_fake_is_closer_with_more_shared_letters():
    """Fake 는 글자가 겹칠수록 가깝다."""
    embed = FakeEmbedder().embed
    scene = embed("책상 위 검은 백팩")

    assert _cos(scene, embed("검은 백팩 어디 있었지")) > _cos(scene, embed("냉장고 옆 물병"))
