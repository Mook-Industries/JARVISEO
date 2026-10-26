"""대상 크롭 — 여백을 넣어 자르고, 경계를 넘거나 비어도 보낼 사진이 나오는지, VLM 용 크기 조정,
손 우선 크롭(가리키는 손이 있는데 선 위에 후보가 없을 때 손끝 방향 사각형)."""

import math

import numpy as np
import pytest

from jarviseo.pointing import crop_target
from jarviseo.pointing.crop import (
    crop_region,
    fit_for_vlm,
    hand_crop_region,
    hand_region,
)
from jarviseo.pointing.hands import HandPose, pointing_ray
from jarviseo.types import BBox, Detection


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


# --- 손 우선 크롭 -------------------------------------------------------------------

# 손끝 (960, 1000) 에서 똑바로 위를 가리키는 선
UP = ((960.0, 1000.0), (0.0, -1.0))


def _hand(base: tuple[float, float], tip: tuple[float, float]) -> HandPose:
    """검지 뿌리(5) → 끝(8)만 의미 있는 손. 손 우선 크롭은 방향만 쓴다."""
    landmarks = [(0.0, 0.0)] * 21
    landmarks[5], landmarks[8] = base, tip
    return HandPose(tuple(landmarks), ((0.0, 0.0, 0.0),) * 21, "Right", 0.9)


POINT_UP = _hand(base=(960.0, 1040.0), tip=(960.0, 1000.0))


def _at_angle(degrees: float, distance: float = 300.0) -> Detection:
    """손끝 (960, 1000) 에서 위쪽 선과 degrees 만큼 벌어진 곳에 중심이 있는 40×40 컵."""
    cx = 960 + distance * math.sin(math.radians(degrees))
    cy = 1000 - distance * math.cos(math.radians(degrees))
    return Detection("cup", 0.9, BBox(cx - 20, cy - 20, cx + 20, cy + 20))


def _corners(region: BBox) -> tuple[float, float, float, float]:
    return (region.x1, region.y1, region.x2, region.y2)


def test_hand_region_is_square_ahead_of_fingertip():
    """중심은 손끝에서 위로 0.2 × 1080 = 216px, 한 변은 0.6 × 1080 = 648px.

    아래쪽은 사진 끝에서 잘린다.
    """
    assert _corners(hand_region(UP, (1920, 1080))) == (636, 460, 1284, 1080)


def test_hand_region_scales_with_photo_height():
    """1280×720 이면 같은 비율로 작아진다(중심 위로 144px, 한 변 432px)."""
    ray = ((640.0, 600.0), (0.0, -1.0))
    assert _corners(hand_region(ray, (1280, 720))) == (424, 240, 856, 672)


def test_hand_region_clips_instead_of_shifting():
    """사진 밖으로 나간 부분은 안으로 밀어 넣지 않고 버린다. 왼쪽 끝을 가리키면 폭이 줄어든다."""
    ray = ((100.0, 540.0), (-1.0, 0.0))
    assert _corners(hand_region(ray, (1920, 1080))) == (0, 216, 208, 864)


def test_hand_region_outside_photo_is_none():
    """사각형이 통째로 사진 밖이면 None 이다(박스 크롭으로 돌아간다)."""
    ray = ((100.0, 540.0), (-1.0, 0.0))
    assert hand_region(ray, (1920, 1080), reach=2.0) is None


def test_hand_crop_without_pointing_hand_is_none():
    """가리키는 손이 없거나 방향이 없으면(뿌리 = 끝) 손 크롭을 하지 않는다."""
    flat = _hand(base=(960.0, 1000.0), tip=(960.0, 1000.0))
    assert hand_crop_region([], None, (1920, 1080)) is None
    assert hand_crop_region([], flat, (1920, 1080)) is None


def test_hand_crop_when_no_candidate():
    """가리키는 손이 있는데 후보가 0개면 손끝 방향을 자른다(전에는 '다시 가리켜 주세요')."""
    region = hand_crop_region([], POINT_UP, (1920, 1080))
    assert region == hand_region(pointing_ray(POINT_UP), (1920, 1080))


@pytest.mark.parametrize(
    ("degrees", "crops"),
    [
        (0.0, False),  # 선 위 — 손 점수 1.0
        (20.0, False),  # 손 점수 0.41 ≥ 0.3 — 아직 선 위
        (30.0, True),  # 손 점수 0.14 < 0.3 — 선 위에 후보 없음
        (90.0, True),  # 옆쪽
    ],
)
def test_hand_crop_only_when_every_candidate_is_off_the_line(degrees, crops):
    """손 점수가 0.3(약 23°) 이상인 후보가 하나라도 있으면 박스로 고른다.

    없으면 손끝 방향을 자른다.
    """
    region = hand_crop_region([_at_angle(degrees), _at_angle(-90.0)], POINT_UP, (1920, 1080))
    assert (region is not None) is crops
