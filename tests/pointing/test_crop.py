"""대상 크롭 — 박스 경계를 넘거나 비어 있어도 VLM 에 보낼 사진이 나오는지."""

import numpy as np

from jarviseo.pointing import crop_target
from jarviseo.types import BBox


def test_crops_only_box_area():
    """박스 영역만 잘라낸다."""
    image = np.zeros((1080, 1920, 3), np.uint8)
    assert crop_target(image, BBox(400, 500, 700, 900)).shape == (400, 300, 3)


def test_fractional_coords_round_outward():
    """소수 좌표는 물체가 잘리지 않게 넓힌다."""
    image = np.zeros((100, 100, 3), np.uint8)
    assert crop_target(image, BBox(10.6, 10.6, 20.2, 20.2)).shape == (11, 11, 3)


def test_clips_outside_part():
    """사진 밖으로 나간 부분은 버린다."""
    image = np.zeros((100, 100, 3), np.uint8)
    # 음수 좌표를 그대로 쓰면 numpy 가 '뒤에서부터'로 읽어 엉뚱한 곳을 자른다.
    assert crop_target(image, BBox(-20, -20, 30, 30)).shape == (30, 30, 3)


def test_box_outside_returns_original():
    """박스가 사진 밖에 있으면 원본을 준다."""
    image = np.zeros((100, 100, 3), np.uint8)
    assert crop_target(image, BBox(200, 200, 300, 300)) is image


def test_crop_does_not_share_memory():
    """크롭은 원본과 메모리를 나누지 않는다."""
    image = np.zeros((100, 100, 3), np.uint8)
    crop = crop_target(image, BBox(0, 0, 10, 10))
    image[:] = 255
    assert crop.max() == 0
