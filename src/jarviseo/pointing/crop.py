"""고른 대상만 잘라내기.  담당: 최홍묵

VLM 과 성분 판정에는 전체 프레임이 아니라 이 크롭이 들어간다(MEMORY.md "결정과 근거").
여백·최소 크기는 대상 크롭 작업에서 이 함수에 더한다.
"""

from __future__ import annotations

import math

import numpy as np

from jarviseo.types import BBox

__all__ = ["crop_target"]


def crop_target(image: np.ndarray, bbox: BBox) -> np.ndarray:
    """박스 영역만 잘라낸다. 사진 밖으로 나간 부분은 버리고, 아예 비면 원본을 준다."""
    height, width = image.shape[:2]
    # 시작은 내림·끝은 올림이라야 물체 가장자리가 잘리지 않는다.
    # 음수 인덱스는 numpy 에서 '뒤에서부터'라서, 0 아래로 내려가지 않게 막는다.
    x1, y1 = max(0, math.floor(bbox.x1)), max(0, math.floor(bbox.y1))
    x2, y2 = min(width, math.ceil(bbox.x2)), min(height, math.ceil(bbox.y2))
    if x2 <= x1 or y2 <= y1:
        # 빈 사진을 VLM 에 보내면 API 오류로 턴 전체가 실패한다.
        return image
    # 슬라이스는 원본과 메모리를 공유한다. 버퍼의 프레임이 바뀌어도 크롭이 그대로 있게 복사한다.
    return image[y1:y2, x1:x2].copy()
