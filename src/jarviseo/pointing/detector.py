"""손끝·물체 검출 (YOLO).  담당: A (팀장)

책임 지표: fingertip mAP@0.5

여기가 이 프로젝트에서 '딥러닝'에 해당하는 부분이다.
COCO 로 사전학습된 YOLO 는 "사람"은 잡아도 "검지 끝이 어디냐"는 모른다.
그래서 손끝 데이터를 직접 모아 파인튜닝한다.

반드시 지킬 것
--------------
1. **baseline 을 학습 전에 잰다.** 학습을 시작한 뒤에는 사전학습 상태의
   성능을 되돌려 잴 수 없다. 사전학습 YOLO 를 우리 평가셋에 그대로 돌린
   숫자를 먼저 기록하고 나서 학습에 들어간다.
2. **test 셋은 최종 측정 전까지 열지 않는다.** 보면서 튜닝하면 나온 숫자가
   실제 성능이 아니게 된다.
3. 배포 포맷은 ONNX 로 통일한다. CoreML 은 맥 전용이라 팀원이 못 쓴다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jarviseo.types import Detection

__all__ = ["FingertipDetector", "ObjectDetector"]


class FingertipDetector:
    """파인튜닝한 YOLO 로 손끝 위치를 찾는다."""

    def __init__(self, weights: Path, conf_threshold: float = 0.25) -> None:
        self.weights = weights
        self.conf_threshold = conf_threshold
        raise NotImplementedError("W1 스파이크에서 사전학습 YOLO 로드부터 확인할 것")

    def detect(self, image: np.ndarray) -> list[Detection]:
        """손끝을 찾아 반환한다. 손이 안 보이면 빈 리스트.

        label 은 "fingertip" 으로 통일한다. types.py 의 약속이다.
        """
        raise NotImplementedError


class ObjectDetector:
    """장면 안의 후보 물체들을 찾는다.

    이쪽은 파인튜닝하지 않고 사전학습 YOLO 를 그대로 쓴다.
    "저거"의 후보가 될 일반적인 물건(컵, 가방, 병 등)은 COCO 에 이미 있다.
    학습 자원은 손끝 쪽에 몰아주는 편이 낫다.
    """

    def __init__(self, weights: Path | str = "yolov8n.pt", conf_threshold: float = 0.25) -> None:
        self.weights = weights
        self.conf_threshold = conf_threshold
        raise NotImplementedError

    def detect(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError
