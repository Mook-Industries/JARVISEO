"""성분표 영역 검출 (YOLO).  담당: 권용현

책임 지표: 성분표 검출 mAP

여기가 성분 기능에서 유일하게 '학습'이 들어가는 부분이다.
사전학습 YOLO 에는 "성분표"라는 클래스가 없으므로 직접 만든다.

왜 그냥 전체 사진을 OCR 에 넣지 않는가
--------------------------------------
과자 봉지 전체를 OCR 에 넣으면 제품명·광고 문구·유통기한이 전부 섞여 나온다.
성분표만 잘라내야 뒤 단계가 깨끗해진다. 그리고 잘라낸 뒤 확대해야
작은 글자가 읽힌다. 검출 정확도가 OCR 인식률을 직접 끌어올리는 구조다.

baseline: 학습 전에 반드시 측정한다. 사전학습 YOLO 로는 성분표 클래스가
없으니, "텍스트 영역 검출기를 그대로 썼을 때"를 baseline 으로 잡는 것이 현실적이다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jarviseo.types import Detection

__all__ = ["PanelDetector"]


class PanelDetector:
    """식품 포장에서 성분표 영역을 찾는다."""

    def __init__(self, weights: Path, conf_threshold: float = 0.25) -> None:
        self.weights = weights
        self.conf_threshold = conf_threshold
        raise NotImplementedError

    def detect(self, image: np.ndarray) -> list[Detection]:
        """성분표 영역을 반환한다. label 은 "ingredient_panel" 로 통일한다.

        여러 개가 나올 수 있다(앞면 영양정보 + 뒷면 원재료명).
        confidence 내림차순으로 정렬해서 반환한다.
        """
        raise NotImplementedError
