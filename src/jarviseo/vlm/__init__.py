"""기본 계층 — 범용 VLM 호출.

이 계층이 항상 먼저 동작하고, 해당되는 특화 모듈이 있으면 맥락을 덧붙인다.
특화 모듈이 해당 없다고 판단하면 기본 응답이 그대로 나간다.
그래서 질문 주제에 제한이 없다.
"""

from jarviseo.vlm.client import VLMClient

__all__ = ["VLMClient"]
