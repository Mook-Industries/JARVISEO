"""대상 크롭 — 여백을 넣어 자르고, 경계를 넘거나 비어도 보낼 사진이 나오는지, VLM 용 크기 조정."""

import numpy as np
import pytest

from jarviseo.pointing import crop_target
from jarviseo.pointing.crop import crop_region, fit_for_vlm
from jarviseo.types import BBox


def _image(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), np.uint8)


# --- crop_target: 자르기 규칙 (여백 없이) ----------------------------------------


def test_crops_only_box_area():
    """박스 영역만 잘라낸다."""
    image = _image(1080, 1920)
    assert crop_target(image, BBox(400, 500, 700, 900), margin=0.0).shape == (400, 300, 3)


def test_fractional_coords_round_outward():
    """소수 좌표는 물체가 잘리지 않게 넓힌다."""
    image = _image(100, 100)
    assert crop_target(image, BBox(10.6, 10.6, 20.2, 20.2), margin=0.0).shape == (11, 11, 3)


def test_clips_outside_part():
    """사진 밖으로 나간 부분은 버린다."""
    image = _image(100, 100)
    # 음수 좌표를 그대로 쓰면 numpy 가 '뒤에서부터'로 읽어 엉뚱한 곳을 자른다.
    assert crop_target(image, BBox(-20, -20, 30, 30), margin=0.0).shape == (30, 30, 3)


def test_box_outside_returns_original():
    """박스가 사진 밖에 있으면 원본을 준다."""
    image = _image(100, 100)
    assert crop_target(image, BBox(200, 200, 300, 300)) is image


def test_crop_does_not_share_memory():
    """크롭은 원본과 메모리를 나누지 않는다."""
    image = _image(100, 100)
    crop = crop_target(image, BBox(0, 0, 10, 10))
    image[:] = 255
    assert crop.max() == 0


# --- crop_target: 여백 ------------------------------------------------------------


def test_default_margin_adds_15_percent_each_side():
    """기본 여백은 박스 가로·세로의 15% 를 사방에 더한다. 300×400 박스 → 390×520."""
    image = _image(1080, 1920)
    assert crop_target(image, BBox(400, 500, 700, 900)).shape == (520, 390, 3)


def test_margin_keeps_box_content_centered():
    """여백은 사방에 같게 붙어서, 박스 안의 물체가 크롭 가운데에 온다."""
    image = _image(1080, 1920)
    image[500:900, 400:700] = 255  # 박스 자리만 흰색

    crop = crop_target(image, BBox(400, 500, 700, 900))

    assert crop[60:460, 45:345].min() == 255  # 위·왼쪽 여백 60·45 px 뒤에 물체
    assert crop[:60].max() == 0 and crop[:, :45].max() == 0


def test_margin_is_clipped_at_photo_edge():
    """사진 가장자리에 붙은 박스는 바깥쪽 여백이 잘린다."""
    image = _image(1080, 1920)
    # 왼쪽 위 모서리 박스 200×100: 오른쪽·아래로만 30·15 px 넓어진다
    assert crop_target(image, BBox(0, 0, 200, 100)).shape == (115, 230, 3)


# --- crop_region: 크롭 사각형 (채점과 같이 쓴다) -------------------------------------


@pytest.mark.parametrize(
    "bbox",
    [BBox(400, 500, 700, 900), BBox(0, 0, 200, 100), BBox(10.6, 10.6, 20.2, 20.2)],
)
def test_crop_region_is_the_area_crop_target_cuts(bbox):
    """crop_region 의 사각형이 crop_target 이 실제로 자른 영역과 같다.

    여백 · 사진 가장자리 · 소수 좌표 세 경우.
    """
    image = _image(1080, 1920)
    image[:] = np.arange(1920, dtype=np.uint16)[None, :, None] % 251  # 열마다 다른 값

    region = crop_region(bbox, (1920, 1080))
    x1, y1, x2, y2 = (int(v) for v in (region.x1, region.y1, region.x2, region.y2))

    assert np.array_equal(crop_target(image, bbox), image[y1:y2, x1:x2])


def test_crop_region_uses_width_then_height():
    """frame_size 는 (너비, 높이)다. 세로 사진(600×1000)에서 아래쪽 여백이 1000 까지 남는다."""
    region = crop_region(BBox(100, 800, 300, 950), (600, 1000))
    assert (region.x1, region.y1, region.x2, region.y2) == (70, 777, 330, 973)


def test_crop_region_outside_photo_is_none():
    """사진 안에 남는 부분이 없으면 None — crop_target 은 이때 원본을 준다."""
    assert crop_region(BBox(200, 200, 300, 300), (100, 100)) is None


# --- fit_for_vlm --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((520, 390), (520, 390)),  # 64~768 사이면 그대로
        ((1000, 2000), (384, 768)),  # 긴 변을 768 로 줄인다
        ((30, 40), (64, 85)),  # 짧은 변을 64 로 키운다
        ((1000, 30), (768, 23)),  # 둘이 부딪히면 긴 변 768 이 이긴다
    ],
)
def test_fit_for_vlm_sizes(shape, expected):
    """짧은 변 64px 미만은 키우고 긴 변은 768px 이하로 줄이며, 가로세로 비율은 지킨다."""
    assert fit_for_vlm(_image(*shape)).shape[:2] == expected


def test_fit_for_vlm_returns_same_image_when_in_range():
    """이미 범위 안이면 다시 계산하지 않고 받은 사진을 그대로 돌려준다."""
    image = _image(520, 390)
    assert fit_for_vlm(image) is image


def test_fit_for_vlm_keeps_portrait_orientation():
    """세로 사진은 세로로 남는다(cv2.resize 의 (너비, 높이) 순서를 뒤집지 않았다)."""
    height, width = fit_for_vlm(_image(1600, 800)).shape[:2]
    assert (height, width) == (768, 384)
