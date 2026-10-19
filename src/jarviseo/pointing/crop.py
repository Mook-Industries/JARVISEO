"""고른 대상만 잘라내기.  담당: 최홍묵

VLM 과 성분 판정에는 전체 프레임이 아니라 이 크롭이 들어간다(MEMORY.md "결정과 근거").
여백(박스의 15%)은 둘 다 받는다. YOLO 박스는 물체에 딱 붙어 손잡이·성분표 끝 글자가 잘린다.
크기 조정(짧은 변 64px 이상, 긴 변 768px 이하)은 VLM 으로 보낼 때만 한다. 성분 판정은 OCR 이라
해상도를 줄이면 손해다. JPEG base64 는 vlm/client.py 의 encode_image 가 한다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from jarviseo.types import BBox

__all__ = ["MARGIN_RATIO", "VLM_MAX_SIDE", "VLM_MIN_SIDE", "crop_target", "fit_for_vlm"]

# 셋 다 크롭 효과 실험(여백 0/15/30%, 크롭 vs 전체 사진) 전의 출발점이다.
MARGIN_RATIO = 0.15
VLM_MIN_SIDE = 64
VLM_MAX_SIDE = 768


def crop_target(image: np.ndarray, bbox: BBox, margin: float = MARGIN_RATIO) -> np.ndarray:
    """박스를 사방으로 margin(박스 크기 기준)만큼 넓혀 잘라낸다.

    사진 밖으로 나간 부분은 버리고, 아예 비면 원본을 준다.
    """
    height, width = image.shape[:2]
    dx, dy = bbox.width * margin, bbox.height * margin
    # 넓힌 다음에 사진 안으로 자른다. 순서를 바꾸면 음수 좌표가 남는다.
    # 시작은 내림·끝은 올림이라야 물체 가장자리가 잘리지 않는다.
    # 음수 인덱스는 numpy 에서 '뒤에서부터'라서, 0 아래로 내려가지 않게 막는다.
    x1, y1 = max(0, math.floor(bbox.x1 - dx)), max(0, math.floor(bbox.y1 - dy))
    x2, y2 = min(width, math.ceil(bbox.x2 + dx)), min(height, math.ceil(bbox.y2 + dy))
    if x2 <= x1 or y2 <= y1:
        # 빈 사진을 VLM 에 보내면 API 오류로 턴 전체가 실패한다.
        return image
    # 슬라이스는 원본과 메모리를 공유한다. 버퍼의 프레임이 바뀌어도 크롭이 그대로 있게 복사한다.
    return image[y1:y2, x1:x2].copy()


def fit_for_vlm(
    image: np.ndarray, min_side: int = VLM_MIN_SIDE, max_side: int = VLM_MAX_SIDE
) -> np.ndarray:
    """VLM 으로 보낼 크기로 맞춘다. 짧은 변이 min_side 보다 작으면 키우고, 긴 변은 max_side 이하.

    둘이 부딪히면(아주 가는 물건) 긴 변 상한이 이긴다. 넘으면 업로드·이미지 토큰이 늘어
    지연 목표가 깨진다. 이미 범위 안이면 받은 사진을 그대로 돌려준다.
    """
    height, width = image.shape[:2]
    scale = 1.0
    if min(height, width) < min_side:
        scale = min_side / min(height, width)
    scale = min(scale, max_side / max(height, width))
    if scale == 1.0:
        return image
    # cv2.resize 는 (너비, 높이) 순서다. numpy shape 의 (높이, 너비)와 반대다.
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # 줄일 때는 픽셀을 평균 내는 INTER_AREA, 키울 때는 계단 무늬가 덜한 INTER_CUBIC.
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, size, interpolation=interpolation)

