"""손 관절 → 가리키는 방향 — 굽힘 각도, 가리킴 판정, 방향, 손 고르기, MediaPipe 결과 바꾸기.

MediaPipe 없이 관절 좌표와 결과 모양을 직접 만들어 본다.
실제 모델로 사진 한 장을 돌리는 테스트는 모델 파일 · 사진이 있을 때만 돈다(CI 는 건너뜀).
손가락은 뿌리에서 위로 뻗고, 마디마다 같은 각도로 앞(z)으로 접는다.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from jarviseo import config
from jarviseo.pointing.hands import (
    HAND_PATCH_SIZE,
    HandDetector,
    HandPose,
    angle_to,
    attach_patch_features,
    bend_deg,
    bend_gap_deg,
    finger_bend,
    hand_patch_box,
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


class _FixedProb:
    """손마다 정해 둔 확률을 주는 가짜 분류기(#223)."""

    needs_patch = False

    def __init__(self, probs: dict[int, float], threshold: float = 0.5) -> None:
        self.probs, self.threshold = probs, threshold

    def prob(self, hand: HandPose) -> float:
        return self.probs[id(hand)]


def test_pick_with_classifier_drops_fist_that_rule_passes():
    """15° 를 넘는 손도 분류기 확률이 낮으면 버린다(주먹 · 브이를 가리킴으로 읽던 문제)."""
    hand = _hand(other_bend=60.0)
    assert pick_pointing_hand([hand], classifier=None) is hand
    assert pick_pointing_hand([hand], classifier=_FixedProb({id(hand): 0.1})) is None


def test_pick_with_classifier_takes_hand_rule_rejects():
    """15° 를 못 넘어도 확률이 문턱 이상이면 고른다."""
    hand = _hand(other_bend=0.0)
    assert pick_pointing_hand([hand], classifier=None) is None
    assert pick_pointing_hand([hand], classifier=_FixedProb({id(hand): 0.9})) is hand


def test_pick_with_classifier_prefers_higher_prob_and_needs_direction():
    """손이 둘이면 확률이 높은 쪽, 방향을 모르는 손은 확률이 높아도 안 고른다."""
    low, high = _hand(), _hand()
    no_ray = _hand(base=(100.0, 100.0), tip=(100.0, 100.0))
    probs = _FixedProb({id(low): 0.6, id(high): 0.8, id(no_ray): 0.99})
    assert pick_pointing_hand([low, no_ray, high], classifier=probs) is high


def test_pick_follows_config_and_falls_back_to_rule(monkeypatch, tmp_path):
    """플래그를 켜도 가중치 파일이 없으면 15° 규칙으로 판정한다."""
    from jarviseo.pointing import hand_classifier

    hand_classifier._load.cache_clear()
    monkeypatch.setattr(config, "ENABLE_HAND_CLASSIFIER", True)
    monkeypatch.setattr(config, "HAND_CLASSIFIER_PATH", tmp_path / "없음.npz")
    pointing = _hand(other_bend=60.0)
    assert pick_pointing_hand([pointing]) is pointing


def test_pick_falls_back_to_rule_when_patch_feature_missing():
    """R2 분류기인데 손에 사진 특징이 없으면(encoder 없이 찾은 손) 15° 규칙으로 판정한다."""
    hand = _hand(other_bend=60.0)
    needs = _FixedProb({id(hand): 0.0})
    needs.needs_patch = True
    assert pick_pointing_hand([hand], classifier=needs) is hand


# --- 손 사진 (R2, #223) ----------------------------------------------------------


def test_patch_box_is_square_with_margin_around_hand():
    """긴 변 100 × (1 + 2 × 0.25) = 150 정사각, 손 상자 가운데."""
    x1, y1, x2, y2 = hand_patch_box([(400.0, 300.0), (500.0, 340.0)], (1280, 720))
    assert x2 - x1 == y2 - y1 == 150
    assert (x1 + x2) / 2 == 450 and (y1 + y2) / 2 == 320


def test_patch_box_shifts_inside_instead_of_cutting():
    """사진 모서리의 손은 잘라 내지 않고 안쪽으로 밀어 정사각을 지킨다(찌그러짐 방지)."""
    x1, y1, x2, y2 = hand_patch_box([(1240.0, 690.0), (1280.0, 720.0)], (1280, 720), margin=1.0)
    assert x2 == 1280 and y2 == 720
    assert x2 - x1 == y2 - y1 == 120
    assert x1 <= 1240 and y1 <= 690  # 손은 상자 안에 남는다


def test_patch_box_never_larger_than_image():
    """손이 화면보다 크게 잡혀도 상자는 사진의 짧은 변까지."""
    assert hand_patch_box([(-100.0, -50.0), (700.0, 500.0)], (640, 360)) == (120, 0, 480, 360)


class _ShapeEncoder:
    """손 사진 크기를 확인하고 손마다 다른 특징을 주는 가짜 특징기."""

    def encode(self, patches):
        assert all(p.shape == (HAND_PATCH_SIZE, HAND_PATCH_SIZE, 3) for p in patches)
        return np.arange(len(patches) * 4, dtype=float).reshape(len(patches), 4)


def test_attach_patch_features_adds_one_vector_per_hand():
    """손마다 224×224 손 사진의 특징이 붙고, 관절은 그대로다."""
    image = np.zeros((720, 1280, 3), np.uint8)
    hands = [_hand(), _hand(base=(300.0, 400.0), tip=(320.0, 380.0))]
    out = attach_patch_features(image, hands, _ShapeEncoder())
    assert [list(h.patch_feature) for h in out] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert out[0].landmarks == hands[0].landmarks
    assert attach_patch_features(image, [], _ShapeEncoder()) == []


def test_hand_pose_with_feature_still_compares():
    """배열 칸이 있어도 == 가 에러 없이 관절로만 비교한다."""
    a = _hand()
    assert attach_patch_features(np.zeros((720, 1280, 3), np.uint8), [a], _ShapeEncoder())[0] == a


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
