"""웨이크워드 "자비서" 감지.  담당: 문태현

책임 지표: FAR / FRR

- FAR (False Acceptance Rate) : 안 불렀는데 깨어난 비율
- FRR (False Rejection Rate)  : 불렀는데 안 깨어난 비율

둘은 맞바꿈 관계다. 민감하게 만들면 잘 깨어나지만 아무 말에나 반응하고,
둔감하게 만들면 조용하지만 불러도 안 깨어난다.
발표 데모에서는 FRR 이 낮은 쪽이 중요하다. 불렀는데 반응이 없으면
시연이 그 자리에서 멈추기 때문이다. 임계값을 데모용으로 따로 둘 것.

측정 방법: 조용한 환경/시끄러운 환경에서 각각 "자비서" 100회,
그리고 비슷한 발음("자비스", "아비서")과 일반 대화 100회를 흘려보내
깨어난 횟수를 센다.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

__all__ = ["WakeWordDetector"]


class WakeWordDetector:
    """마이크를 계속 듣고 있다가 "자비서"가 들리면 콜백을 부른다.

    이 부분은 로컬에서만 돈다. 상시 동작하는 것을 전부 로컬에 두는 것이
    이 프로젝트의 프라이버시 근거다. 클라우드로 올리면 그 근거가 사라진다.
    """

    def __init__(
        self,
        model_path: Path,
        threshold: float = 0.5,
        on_detect: Callable[[float], None] | None = None,
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.on_detect = on_detect
        raise NotImplementedError

    def start(self) -> None:
        """감지를 시작한다. 별도 스레드에서 돈다."""
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def mute(self, seconds: float) -> None:
        """이만큼 감지를 쉰다. TTS 재생 중 되먹임을 막는 데 쓴다."""
        raise NotImplementedError
