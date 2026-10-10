"""문장 임베딩.  담당: 문태현

장면 설명과 질문을 같은 공간의 벡터로 바꾼다. 둘이 가까우면 같은 것을 말하는 문장이다.
모델 이름은 ``config.EMBED_MODEL`` 하나만 본다.

``FakeEmbedder`` 는 키 없이 쓰는 가짜다. 메서드 모양이 실제와 같아서
테스트·그래프 개발에서 바꿔 끼우면 된다(``tests/test_memory_embed.py`` 가 확인한다).
"""

from __future__ import annotations

import math
import time
import zlib

from openai import OpenAI

from jarviseo import config

__all__ = ["DIM", "FakeEmbedder", "TextEmbedder"]

# observation.embedding 의 VECTOR(1536). text-embedding-3-small 의 기본 차원이다.
DIM = 1536
# 한 번 기다리는 최대 시간(초)과 다시 보내는 횟수. 질문 한 번에 한 번 부르므로 짧게 끊는다.
TIMEOUT_SEC, RETRIES = 5.0, 1


class TextEmbedder:
    """문장을 OpenAI 임베딩으로 바꾼다."""

    def __init__(self, model: str = config.EMBED_MODEL, client: OpenAI | None = None) -> None:
        """client 는 테스트에서 가짜 API 를 끼울 때만 넘긴다."""
        self.model = model
        self.client = client or OpenAI(
            api_key=config.OPENAI_API_KEY or None, timeout=TIMEOUT_SEC, max_retries=RETRIES
        )
        # 마지막 호출의 왕복 시간(ms). 실패했으면 None.
        self.last_latency_ms: float | None = None

    def embed(self, text: str) -> list[float]:
        """text 의 임베딩(길이 1, DIM 차원)을 돌려준다. 실패하면 OpenAIError 가 그대로 올라간다."""
        self.last_latency_ms = None
        asked_at = time.monotonic()
        response = self.client.embeddings.create(model=self.model, input=text)
        self.last_latency_ms = (time.monotonic() - asked_at) * 1000
        return response.data[0].embedding


class FakeEmbedder:
    """글자 두 개씩(바이그램)을 해시해 DIM 칸에 세고 길이를 1로 맞춘다.

    뜻은 모르지만 글자가 많이 겹칠수록 가까워진다.
    "검은 백팩"과 "백팩 어디 뒀지"는 가깝고 "물병"과는 멀다. 같은 문장은 늘 같은 벡터다.
    """

    def __init__(self) -> None:
        self.model = "fake-bigram"  # observation.embed_model 에 남는 이름
        self.last_latency_ms: float | None = None

    def embed(self, text: str) -> list[float]:
        self.last_latency_ms = 0.0
        counts = [0.0] * DIM
        for word in text.split():
            for gram in (word[i : i + 2] for i in range(max(len(word) - 1, 1))):
                counts[zlib.crc32(gram.encode()) % DIM] += 1.0
        norm = math.sqrt(sum(c * c for c in counts)) or 1.0
        return [c / norm for c in counts]
