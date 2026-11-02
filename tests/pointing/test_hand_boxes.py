"""pointing/hand_boxes.py — 손 박스 검출기 + 관절 모델 직접 실행(#256 ⑤-b).

모델 없이 자르기 좌표 · 손 세우기 각도 · 3D 관절 되돌리기 · 왼오를 확인하고,
가짜 검출기 · 가짜 관절 모델로 read · find 흐름을 본다.
"""

import math

import numpy as np
import pytest

from jarviseo.pointing.hand_boxes import (
    CROP_SCALE,
    MIN_PRESENCE,
    HandBoxFinder,
    crop_matrix,
    crop_square,
    handedness_name,
    rotate_world,
    to_image,
    upright_angle,
)
from jarviseo.types import BBox


def _apply(matrix: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    return matrix[:, :2] @ np.asarray(point, dtype=float) + matrix[:, 2]


# --- 자르기 좌표 ------------------------------------------------------------------


def test_crop_square_centers_on_box_with_scaled_long_side():
    cx, cy, side = crop_square(BBox(100, 100, 200, 150), scale=1.2)
    assert (cx, cy, side) == pytest.approx((150, 125, 120))


def test_crop_matrix_puts_center_in_the_middle_and_to_image_undoes_it():
    matrix = crop_matrix(500, 300, 100, 37.0, size=224)
    assert _apply(matrix, (500, 300)) == pytest.approx((112, 112))
    points = np.array([[10.0, 20.0], [200.0, 50.0]])
    back = to_image(points, matrix)
    assert np.array([_apply(matrix, p) for p in back]) == pytest.approx(points)


def test_crop_matrix_turns_counterclockwise_on_screen():
    """90° 면 원본에서 중심 오른쪽에 있던 점이 크롭에서는 중심 위로 간다."""
    matrix = crop_matrix(500, 300, 224, 90.0, size=224)
    assert _apply(matrix, (510, 300)) == pytest.approx((112, 102))


# --- 손 세우기 · 3D 관절 · 왼오 ----------------------------------------------------------


def _wrist_to_middle(dx: float, dy: float) -> np.ndarray:
    points = np.zeros((21, 2))
    points[9] = (dx, dy)
    return points


@pytest.mark.parametrize(
    ("dx", "dy", "angle"),
    [(0, -10, 0.0), (10, 0, 90.0), (-10, 0, -90.0), (0, 10, 180.0)],
)
def test_upright_angle_turns_fingers_up(dx, dy, angle):
    """손목 → 중지 뿌리가 위면 0, 오른쪽이면 90(시계 반대로 돌림), 아래면 180."""
    assert upright_angle(_wrist_to_middle(dx, dy)) == pytest.approx(angle)


def test_upright_angle_makes_hand_point_up_in_the_crop():
    """구한 각도로 자르면 크롭 안에서 손목 → 중지 뿌리가 정말 위(y 감소)를 향한다."""
    wrist, middle = (300.0, 300.0), (330.0, 280.0)
    points = np.zeros((21, 2))
    points[0], points[9] = wrist, middle
    matrix = crop_matrix(300, 300, 224, upright_angle(points), size=224)
    dx, dy = _apply(matrix, middle) - _apply(matrix, wrist)
    assert dx == pytest.approx(0, abs=1e-6) and dy < 0


def test_rotate_world_matches_how_to_image_turns_screen_points():
    """크롭에서 오른쪽(+x)인 방향을 원본으로 되돌리면, 화면 관절과 같은 방향이 된다."""
    angle = 37.0
    matrix = crop_matrix(500, 300, 224, angle, size=224)
    start, end = to_image(np.array([[100.0, 100.0], [110.0, 100.0]]), matrix)
    screen = (end - start) / np.linalg.norm(end - start)
    world = rotate_world(np.array([[1.0, 0.0, 0.5]]), angle)[0]
    assert world[:2] == pytest.approx(screen)
    assert world[2] == 0.5


def test_rotate_world_quarter_turn():
    """시계 반대로 90° 돌려 자른 크롭의 오른쪽은 원본의 아래쪽이다."""
    assert rotate_world(np.array([[1.0, 0.0, 0.0]]), 90.0)[0] == pytest.approx((0, 1, 0))


def test_handedness_name_uses_mediapipe_labels():
    name, score = handedness_name(0.8)
    assert name == "Right" and score == pytest.approx(0.8)
    name, score = handedness_name(0.2)
    assert name == "Left" and score == pytest.approx(0.8)


# --- read · find (가짜 모델) ------------------------------------------------------------


class _FakeNet:
    """관절 모델 흉내: 크롭 안에 선 손(손목 아래 · 손가락 위)을 늘 같은 자리에 돌려준다."""

    def __init__(self, presence: float = 0.9, right: float = 0.8) -> None:
        landmarks = np.zeros((21, 3))
        landmarks[:, :2] = (112.0, 112.0)
        landmarks[0, :2] = (112.0, 180.0)  # 손목
        landmarks[9, :2] = (112.0, 100.0)  # 중지 뿌리
        landmarks[5, :2], landmarks[8, :2] = (100.0, 110.0), (100.0, 40.0)  # 검지 뿌리 · 끝
        self.outputs = {
            "Identity": landmarks.reshape(1, 63),
            "Identity_1": np.array([[presence]]),
            "Identity_2": np.array([[right]]),
            "Identity_3": np.zeros((1, 63)),
        }
        self.inputs = []

    def setInput(self, blob):  # OpenCV 이름 그대로
        self.inputs.append(blob.shape)

    def forward(self, names):
        return [self.outputs[name] for name in names]


class _List(list):
    def tolist(self):
        return list(self)


class _FakeModel:
    """YOLO 흉내: 확신 순서가 섞인 박스 셋."""

    def predict(self, image, conf, verbose):
        boxes = type(
            "Boxes",
            (),
            {
                "conf": _List([0.3, 0.9, 0.5]),
                "xyxy": _List([[0, 0, 10, 10], [100, 100, 200, 200], [50, 50, 80, 80]]),
            },
        )
        return [type("Result", (), {"boxes": boxes})]


def _finder(net: _FakeNet) -> HandBoxFinder:
    finder = HandBoxFinder.__new__(HandBoxFinder)  # 파일 · ultralytics 없이 흐름만 본다
    finder.model, finder.net = _FakeModel(), net
    return finder


def test_boxes_keeps_two_most_confident():
    boxes = _finder(_FakeNet()).boxes(np.zeros((300, 300, 3), np.uint8))
    assert boxes == [BBox(100, 100, 200, 200), BBox(50, 50, 80, 80)]


def test_read_maps_crop_landmarks_back_into_the_box():
    """선 손이라 2차도 돌리지 않고, 크롭 관절이 박스 둘레의 원본 좌표로 돌아온다."""
    net = _FakeNet()
    box = BBox(400, 300, 500, 400)  # 중심 (450, 350), 정사각 한 변 = 100 × CROP_SCALE
    hand = _finder(net).read(np.zeros((720, 1280, 3), np.uint8), box)

    assert hand is not None and len(net.inputs) == 2
    assert net.inputs[0] == (1, 3, 224, 224)
    wrist, middle = hand.landmarks[0], hand.landmarks[9]
    assert wrist == pytest.approx((450, 350 + (180 - 112) * 100 * CROP_SCALE / 224))
    assert middle[1] < wrist[1]
    assert (hand.handedness, hand.score) == ("Right", pytest.approx(0.8))


def test_read_drops_low_presence():
    hand = _finder(_FakeNet(presence=MIN_PRESENCE - 0.1)).read(
        np.zeros((720, 1280, 3), np.uint8), BBox(400, 300, 500, 400)
    )
    assert hand is None


def test_find_reads_each_box():
    hands = _finder(_FakeNet()).find(np.zeros((720, 1280, 3), np.uint8))
    assert len(hands) == 2


def test_finder_missing_weights_explains_how_to_get_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="손 박스 가중치가 없다"):
        HandBoxFinder(weights=tmp_path / "hand_box.pt", landmark_model=tmp_path / "x.task")


def test_finder_missing_landmark_model_points_to_readme(tmp_path):
    weights = tmp_path / "hand_box.pt"
    weights.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="pointing/README.md"):
        HandBoxFinder(weights=weights, landmark_model=tmp_path / "x.task")


def test_upright_angle_is_finite_for_degenerate_hand():
    """손목과 중지 뿌리가 겹쳐도(관절이 다 한 점) 각도가 숫자로 나온다."""
    assert math.isfinite(upright_angle(np.zeros((21, 2))))
