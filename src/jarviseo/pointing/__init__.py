"""① 지시 대상 특정 — 담당: 최홍묵.

이 프로젝트의 핵심 기여다.

시야에 물체가 여러 개일 때 범용 모델은 사용자가 무엇을 묻는지 모른다.
착용형 기기에는 화면을 터치해 대상을 지정할 방법도 없다.
이 문제를 손짓·시선·언어 여러 단서를 합쳐서 푼다.

단서를 하나씩 추가하며 기여도를 따로 재는 것(ablation)이 발표의 중심이다.
"""

from jarviseo.pointing.resolver import resolve_target

__all__ = ["resolve_target"]
