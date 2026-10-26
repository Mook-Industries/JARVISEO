"""고른 대상만 잘라내기.  담당: 최홍묵

VLM 과 성분 판정에는 전체 프레임이 아니라 이 크롭이 들어간다(MEMORY.md "결정과 근거").
여백(박스의 15%)은 둘 다 받는다. YOLO 박스는 물체에 딱 붙어 손잡이·성분표 끝 글자가 잘린다.
크기 조정(짧은 변 64px 이상, 긴 변 768px 이하)은 VLM 으로 보낼 때만 한다. 성분 판정은 OCR 이라
해상도를 줄이면 손해다. JPEG base64 는 vlm/client.py 의 encode_image 가 한다.

가리키는 손이 있는데 손가락 선 근처에 후보가 하나도 없으면(정답이 YOLO 후보에 없는 경우)
박스 대신 손끝이 향한 곳을 자른다(손 우선 크롭, #205). 후보 중에서 고르면 엉뚱한 물건을
확신하거나 되묻기만 한다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from jarviseo.pointing.cues import score_point
from jarviseo.pointing.hands import HandPose, Ray, pointing_ray
from jarviseo.types import BBox, Detection

__all__ = [
    "HAND_REACH",
    "HAND_SIDE",
    "MARGIN_RATIO",
    "ON_LINE_SCORE",
    "VLM_MAX_SIDE",
    "VLM_MIN_SIDE",
    "crop_region",
    "crop_target",
    "fit_for_vlm",
    "hand_crop_region",
    "hand_region",
]

# 셋 다 크롭 효과 실험(여백 0/15/30%, 크롭 vs 전체 사진) 전의 출발점이다.
MARGIN_RATIO = 0.15
VLM_MIN_SIDE = 64
VLM_MAX_SIDE = 768
# 손 우선 크롭: 손끝에서 손가락 방향으로 사진 높이의 0.2배 나간 곳이 중심, 한 변은 높이의 0.6배.
# valid 에서 정답이 후보에 없던 11장 중 9장, 가리키는 손을 고른 18장 중 16장의 정답을
# 60% 이상 담았다.
# 손끝이 크롭 안에 들어가서(0.2 < 0.3) VLM 이 손가락이 무엇을 가리키는지 같이 본다.
HAND_REACH = 0.2
HAND_SIDE = 0.6
# 후보의 손 점수(v2)가 모두 이보다 작으면 "선 위에 후보가 없다"고 본다. 0.3 은 선에서 약 23°.
# valid 에서 정답이 후보에 없던 11장 중 8장, 정답이 후보에 있던 7장 중 1장이 여기 걸렸다.
ON_LINE_SCORE = 0.3


def crop_region(
    bbox: BBox, frame_size: tuple[int, int], margin: float = MARGIN_RATIO
) -> BBox | None:
    """crop_target 이 잘라낼 사각형(픽셀 정수 좌표). 사진 안에 남는 부분이 없으면 None.

    frame_size 는 (너비, 높이)다. 실제 크롭과 ablation 채점(크롭 담김)이 이 계산 하나를 같이 써서,
    한쪽만 고쳐져 둘이 어긋나는 일을 막는다.
    """
    width, height = frame_size
    dx, dy = bbox.width * margin, bbox.height * margin
    # 넓힌 다음에 사진 안으로 자른다. 순서를 바꾸면 음수 좌표가 남는다.
    # 시작은 내림·끝은 올림이라야 물체 가장자리가 잘리지 않는다.
    # 음수 인덱스는 numpy 에서 '뒤에서부터'라서, 0 아래로 내려가지 않게 막는다.
    x1, y1 = max(0, math.floor(bbox.x1 - dx)), max(0, math.floor(bbox.y1 - dy))
    x2, y2 = min(width, math.ceil(bbox.x2 + dx)), min(height, math.ceil(bbox.y2 + dy))
    if x2 <= x1 or y2 <= y1:
        return None
    return BBox(x1, y1, x2, y2)


def crop_target(image: np.ndarray, bbox: BBox, margin: float = MARGIN_RATIO) -> np.ndarray:
    """박스를 사방으로 margin(박스 크기 기준)만큼 넓혀 잘라낸다.

    사진 밖으로 나간 부분은 버리고, 아예 비면 원본을 준다.
    """
    height, width = image.shape[:2]
    region = crop_region(bbox, (width, height), margin)
    if region is None:
        # 빈 사진을 VLM 에 보내면 API 오류로 턴 전체가 실패한다.
        return image
    x1, y1, x2, y2 = (int(v) for v in (region.x1, region.y1, region.x2, region.y2))
    # 슬라이스는 원본과 메모리를 공유한다. 버퍼의 프레임이 바뀌어도 크롭이 그대로 있게 복사한다.
    return image[y1:y2, x1:x2].copy()


def hand_region(
    ray: Ray,
    frame_size: tuple[int, int],
    reach: float = HAND_REACH,
    side: float = HAND_SIDE,
) -> BBox | None:
    """손끝에서 손가락 방향으로 reach × 높이 나간 곳을 중심으로 한 변 side × 높이 정사각형.

    frame_size 는 (너비, 높이)다. 높이 기준이라 1920×1080 · 1280×720 에서 같은 비율로 자른다.
    사진 밖으로 나간 부분은 안으로 밀어 넣지 않고 버린다. 밀어 넣으면 손가락이 향하지 않은 곳이
    들어온다. 사진 안에 남는 부분이 없으면 None.
    """
    (tx, ty), (dx, dy) = ray
    height = frame_size[1]
    cx, cy = tx + dx * reach * height, ty + dy * reach * height
    half = side * height / 2
    # 사진 안으로 자르고 정수로 맞추는 규칙은 박스 크롭과 같다(여백만 0).
    return crop_region(BBox(cx - half, cy - half, cx + half, cy + half), frame_size, margin=0.0)


def hand_crop_region(
    candidates: list[Detection], hand: HandPose | None, frame_size: tuple[int, int]
) -> BBox | None:
    """가리키는 손이 있는데 선 위에 후보가 없으면 손 우선 크롭 사각형. 아니면 None(박스로 고른다).

    "선 위에 후보가 없다" = 후보가 0개이거나, 모든 후보의 손 점수가 ON_LINE_SCORE 미만.
    점수는 resolver 결과를 읽지 않고 여기서 다시 매긴다. resolver 는 켠 단서만 기록해서
    손 단서를 끈 실행(v1)에는 point 점수가 없다. 후보마다 각도 하나라 비용은 없다.
    """
    ray = pointing_ray(hand) if hand is not None else None
    if ray is None:
        return None
    if candidates and max(s.score for s in score_point(candidates, ray)) >= ON_LINE_SCORE:
        return None
    return hand_region(ray, frame_size)


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
