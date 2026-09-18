"""프레임 선명도 측정과 프레임 버퍼.

알려진 함정 두 개를 여기서 같이 처리한다.

1. **모션 블러** — 안경을 쓴 채 고개를 돌리면 프레임이 흐려져서 OCR 도
   검출도 다 실패한다. 그래서 한 장만 찍는 게 아니라 최근 프레임을 버퍼에
   모아두고, 그중 가장 선명한 것을 고른다.

2. **질문 시점 프레임 오차** — 발화가 '끝난' 뒤에 찍으면 이미 고개가
   돌아가서 엉뚱한 장면이 잡힌다. 그래서 발화가 '시작된' 시각 근처의
   프레임 중에서 고른다.

두 조건이 충돌할 수 있다(발화 시작 시점 프레임이 흐릴 수 있다).
그래서 ``pick_best`` 는 시각 근처로 후보를 좁힌 뒤, 그 안에서 가장 선명한
것을 고르는 순서로 동작한다.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from jarviseo.types import Frame

__all__ = ["laplacian_variance", "FrameBuffer"]

# 이 값보다 선명도가 낮으면 흐린 프레임으로 본다.
# 카메라·조명이 바뀌면 절대값이 통째로 달라지므로, 실물 카메라가 도착하면
# 실제로 찍어보고 다시 정해야 하는 값이다. 지금 값은 출발점일 뿐이다.
DEFAULT_SHARPNESS_THRESHOLD = 100.0


def laplacian_variance(image: np.ndarray) -> float:
    """이미지가 얼마나 선명한지를 숫자 하나로 반환한다. 클수록 선명하다.

    원리는 단순하다. 초점이 맞은 사진은 물체의 경계에서 밝기가 급격히 바뀐다.
    흐린 사진은 그 경계가 뭉개져서 밝기 변화가 완만하다.
    라플라시안은 그 '밝기 변화의 급격함'을 뽑아내는 필터이고,
    그 결과의 분산(variance)을 재면 변화가 얼마나 심한지가 숫자로 나온다.

    주의: 절대값끼리는 비교하면 안 된다. 같은 장면을 찍은 프레임들 사이에서
    상대 비교할 때만 의미가 있다. 어두운 방에서 찍으면 선명해도 값이 낮게 나온다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class FrameBuffer:
    """최근 프레임 N장을 들고 있다가, 조건에 맞는 최적 프레임을 골라주는 버퍼.

    사용 흐름::

        buffer = FrameBuffer(maxlen=30)          # 30fps 기준 약 1초치
        ...  # 캡처 스레드가 계속 buffer.push(frame) 을 호출
        # 웨이크워드 감지 후, VAD 가 알려준 발화 시작 시각으로 프레임을 고른다
        frame = buffer.pick_best(around=utterance.started_at, window=0.5)
    """

    def __init__(self, maxlen: int = 30) -> None:
        self._frames: deque[Frame] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._frames)

    def push(self, frame: Frame) -> None:
        """프레임을 버퍼에 넣는다. 선명도는 이때 한 번만 계산해서 캐시한다."""
        if frame.sharpness is None:
            frame.sharpness = laplacian_variance(frame.image)
        self._frames.append(frame)

    def clear(self) -> None:
        self._frames.clear()

    def pick_best(
        self,
        around: float | None = None,
        window: float = 0.5,
    ) -> Frame | None:
        """가장 쓸 만한 프레임 한 장을 고른다.

        Args:
            around: 이 시각(monotonic 초) 근처의 프레임만 후보로 삼는다.
                보통 발화 시작 시각을 넣는다. None 이면 버퍼 전체가 후보다.
            window: around 기준 앞뒤 몇 초까지 후보로 볼지.

        Returns:
            후보 중 가장 선명한 프레임. 버퍼가 비었으면 None.

        후보 구간에 프레임이 하나도 없으면 구간을 무시하고 버퍼 전체에서
        고른다. 되묻기보다는 흐린 프레임이라도 주는 편이 낫고, 흐린지 여부는
        ``is_too_blurry`` 로 그래프 쪽에서 따로 판단하기 때문이다.
        """
        if not self._frames:
            return None

        candidates = list(self._frames)
        if around is not None:
            in_window = [f for f in candidates if abs(f.timestamp - around) <= window]
            if in_window:
                candidates = in_window

        return max(candidates, key=lambda f: f.sharpness or 0.0)

    @staticmethod
    def is_too_blurry(
        frame: Frame, threshold: float = DEFAULT_SHARPNESS_THRESHOLD
    ) -> bool:
        """이 프레임이 쓰기에 너무 흐린지 판단한다.

        True 가 나오면 LangGraph 에서 recapture 분기로 되돌아간다.
        """
        sharpness = frame.sharpness
        if sharpness is None:
            sharpness = laplacian_variance(frame.image)
            frame.sharpness = sharpness
        return sharpness < threshold
