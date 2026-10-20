"""손 관절 → 가리키는 방향 — 굽힘 각도, 가리킴 판정, 방향, 손 고르기, MediaPipe 결과 바꾸기.

MediaPipe 없이 관절 좌표와 결과 모양을 직접 만들어 본다.
실제 모델로 사진 한 장을 돌리는 테스트는 모델 파일 · 사진이 있을 때만 돈다(CI 는 건너뜀).
손가락은 뿌리에서 위로 뻗고, 마디마다 같은 각도로 앞(z)으로 접는다.
"""

import math
from types import SimpleNamespace

import pytest

from jarviseo import config
from jarviseo.pointing.hands import (
    HandDetector,
    HandPose,
    angle_to,
    bend_deg,
    bend_gap_deg,
    finger_bend,
    is_pointing,
    pick_pointing_hand,
    pointing_ray,
    pointing_score,
    poses_from_result,
)

SEG = 0.02  # 마디 길이 2cm
FINGER_BASES = (5, 9, 13, 17)


def _finger(base_x: float, joint_bend: float) -> list[tuple[float, float, float]]:
    """뿌리부터 관절 4개. 두 번째 · 세 번째 관절에서 각각 joint_bend 도 꺾인다."""
    points = [(base_x, 0.0, 0.0)]
    heading = 0.0
    for step in range(3):
        if step > 0:
            heading += math.radians(joint_bend)
        x, y, z = points[-1]
        points.append((x, y + SEG * math.cos(heading), z + SEG * math.sin(heading)))
    return points


def _hand(
    index_bend: float = 0.0,
    other_bend: float = 60.0,
    base: tuple[float, float] = (100.0, 200.0),
    tip: tuple[float, float] = (100.0, 100.0),
) -> HandPose:
    """index_bend · other_bend 는 마디 하나의 꺾임. 손가락 굽힘은 그 두 배다."""
    world = [(0.0, -0.05, 0.0)] * 5  # 손목 · 엄지는 판정에 안 쓴다
    for i, finger_base in enumerate(FINGER_BASES):
        world += _finger(i * 0.02, index_bend if finger_base == 5 else other_bend)
    landmarks = [(0.0, 0.0)] * 21
    landmarks[5], landmarks[8] = base, tip
    return HandPose(landmarks=tuple(landmarks), world=tuple(world), handedness="Right", score=0.9)


# --- bend_deg ------------------------------------------------------------------


def test_bend_deg_straight_is_zero():
    """일직선이면 꺾이지 않았다."""
    assert bend_deg((0, 0, 0), (0, 1, 0), (0, 2, 0)) == pytest.approx(0.0)


def test_bend_deg_right_angle():
    """직각으로 꺾이면 90°."""
    assert bend_deg((0, 0, 0), (0, 1, 0), (1, 1, 0)) == pytest.approx(90.0)


def test_bend_deg_overlapping_points_is_zero():
    """점이 겹치면 각도를 정할 수 없어 0° 로 둔다(acos 가 터지지 않는다)."""
    assert bend_deg((0, 1, 0), (0, 1, 0), (0, 2, 0)) == 0.0


# --- 가리킴 판정 ---------------------------------------------------------------


def test_finger_bend_sums_two_joints():
    """마디마다 30° 꺾인 손가락은 굽힘 60°."""
    assert finger_bend(_hand(index_bend=30.0), 5) == pytest.approx(60.0)


def test_pointing_hand():
    """검지만 펴고 나머지를 접은 손은 가리키는 손이고 점수 1."""
    hand = _hand(index_bend=0.0, other_bend=60.0)
    assert bend_gap_deg(hand) == pytest.approx(120.0)
    assert is_pointing(hand)
    assert pointing_score(hand) == 1.0


def test_open_hand_is_not_pointing():
    """다섯 손가락을 다 편 손(키보드에 얹은 손)은 가리키지 않는다."""
    hand = _hand(index_bend=0.0, other_bend=0.0)
    assert not is_pointing(hand)
    assert pointing_score(hand) == 0.0


def test_fist_is_not_pointing():
    """다 접은 손도 검지와 나머지 차이가 없다."""
    assert not is_pointing(_hand(index_bend=60.0, other_bend=60.0))


def test_score_never_negative():
    """검지만 접은 손은 차이가 음수지만 점수는 0 에서 멈춘다."""
    hand = _hand(index_bend=60.0, other_bend=0.0)
    assert bend_gap_deg(hand) < 0
    assert pointing_score(hand) == 0.0


def test_score_scales_below_full_gap():
    """차이 45° 는 점수 0.5."""
    assert pointing_score(_hand(other_bend=22.5)) == pytest.approx(0.5)


def test_gap_threshold():
    """차이 10° 는 기본 기준(15°)에 못 미치고, 기준을 낮추면 통과한다."""
    hand = _hand(index_bend=0.0, other_bend=5.0)
    assert bend_gap_deg(hand) == pytest.approx(10.0)
    assert not is_pointing(hand)
    assert is_pointing(hand, min_gap_deg=5.0)


@pytest.mark.parametrize("field", ["landmarks", "world"])
def test_wrong_landmark_count_raises(field):
    """관절이 21개가 아니면 잘못된 입력이다."""
    hand = _hand()
    kwargs = {"landmarks": hand.landmarks, "world": hand.world}
    kwargs[field] = kwargs[field][:20]
    with pytest.raises(ValueError, match="21"):
        HandPose(**kwargs, handedness="Right", score=0.9)


# --- 방향 ----------------------------------------------------------------------


def test_ray_starts_at_tip_and_has_unit_length():
    """선은 검지 끝(8)에서 시작하고, 뿌리(5) → 끝 방향이며 길이가 1이다."""
    origin, direction = pointing_ray(_hand(base=(100.0, 200.0), tip=(100.0, 100.0)))
    assert origin == (100.0, 100.0)
    assert direction == pytest.approx((0.0, -1.0))


def test_ray_none_when_base_and_tip_overlap():
    """뿌리와 끝이 1픽셀 안에 겹치면 방향이 없다."""
    assert pointing_ray(_hand(base=(100.0, 100.0), tip=(100.5, 100.0))) is None


def test_angle_to():
    """앞은 0°, 옆은 90°, 뒤는 180°."""
    ray = ((100.0, 100.0), (0.0, -1.0))
    assert angle_to(ray, (100.0, 0.0)) == pytest.approx(0.0)
    assert angle_to(ray, (200.0, 100.0)) == pytest.approx(90.0)
    assert angle_to(ray, (100.0, 300.0)) == pytest.approx(180.0)


def test_angle_to_tip_itself_is_none():
    """점이 손끝과 같으면 방향을 비교할 수 없다."""
    assert angle_to(((100.0, 100.0), (0.0, -1.0)), (100.0, 100.0)) is None


# --- pick_pointing_hand --------------------------------------------------------


def test_pick_skips_open_hand():
    """한 손은 펴서 얹고 다른 손으로 가리키면, 순서와 상관없이 가리키는 손을 고른다."""
    open_hand, pointing = _hand(other_bend=0.0), _hand(other_bend=60.0)
    assert pick_pointing_hand([open_hand, pointing]) is pointing


def test_pick_prefers_higher_score():
    """둘 다 가리키면 더 가리키는 모양인 손."""
    weak, strong = _hand(other_bend=20.0), _hand(other_bend=60.0)
    assert pick_pointing_hand([weak, strong]) is strong


def test_pick_skips_hand_without_direction():
    """모양은 가리켜도 방향을 못 구하는 손은 고르지 않는다."""
    assert pick_pointing_hand([_hand(base=(100.0, 100.0), tip=(100.0, 100.0))]) is None


def test_pick_none_when_no_pointing_hand():
    """손이 없거나 가리키는 손이 없으면 None(손 단서 기권)."""
    assert pick_pointing_hand([]) is None
    assert pick_pointing_hand([_hand(other_bend=0.0)]) is None


def test_pick_uses_given_threshold():
    """기준을 낮추면 약하게 가리킨 손도 고른다(측정 스크립트가 기준별로 잰다)."""
    hand = _hand(other_bend=5.0)
    assert pick_pointing_hand([hand]) is None
    assert pick_pointing_hand([hand], min_gap_deg=5.0) is hand


# --- MediaPipe 결과 → HandPose -------------------------------------------------


def _point(x: float, y: float, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _result(hands: int = 1, sides: int | None = None) -> SimpleNamespace:
    """MediaPipe HandLandmarkerResult 와 같은 모양. 화면 관절은 0~1, 3D 관절은 미터."""
    screen = [[_point(0.5, 0.25)] * 21 for _ in range(hands)]
    world = [[_point(0.01, 0.02, 0.03)] * 21 for _ in range(hands)]
    side = [[SimpleNamespace(category_name="Left", score=0.8)] for _ in range(sides or hands)]
    return SimpleNamespace(hand_landmarks=screen, hand_world_landmarks=world, handedness=side)


def test_poses_from_result_converts_screen_to_pixels():
    """화면 관절은 0~1 에 너비 · 높이를 곱해 픽셀로, 3D 관절과 왼손/오른손은 그대로."""
    (hand,) = poses_from_result(_result(), (1920, 1080))
    assert hand.landmarks[8] == (960.0, 270.0)
    assert hand.world[8] == (0.01, 0.02, 0.03)
    assert (hand.handedness, hand.score) == ("Left", 0.8)


def test_poses_from_result_no_hands():
    """손이 없으면 빈 리스트. 오류가 아니다."""
    assert poses_from_result(_result(hands=0), (1920, 1080)) == []


def test_poses_from_result_mismatched_lists_raise():
    """관절과 왼손/오른손 수가 어긋나면 짝을 잘못 맞추지 않고 멈춘다."""
    with pytest.raises(ValueError):
        poses_from_result(_result(hands=2, sides=1), (1920, 1080))


# --- HandDetector --------------------------------------------------------------


def test_detector_missing_model_explains_how_to_get_it(tmp_path):
    """모델 파일이 없으면 받는 법을 알려준다. mediapipe 가 없어도 이 안내가 먼저 나온다."""
    with pytest.raises(FileNotFoundError, match="README"):
        HandDetector(tmp_path / "hand_landmarker.task")


PHOTO = config.DATASETS_DIR / "pointing" / "raw" / "s01" / "s01_desk_0002.jpg"


@pytest.mark.skipif(
    not (config.HAND_LANDMARKER_PATH.exists() and PHOTO.exists()),
    reason="모델 파일이나 평가셋 사진이 없다(CI)",
)
def test_detector_finds_pointing_hand_in_real_photo():
    """마우스를 가리킨 사진에서 가리키는 손 하나를 찾고, 검지 끝이 실측 위치(863, 522) 근처다."""
    pytest.importorskip("mediapipe")
    import cv2
    import numpy as np

    image = cv2.imdecode(np.fromfile(PHOTO, dtype=np.uint8), cv2.IMREAD_COLOR)
    hands = HandDetector().detect(image)
    hand = pick_pointing_hand(hands)
    assert hand is not None
    assert hand.landmarks[8] == pytest.approx((863, 522), abs=5)
