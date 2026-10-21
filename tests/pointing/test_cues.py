"""단서 점수 — 가운데 단서(v1)와 손 방향 단서(v2)가 맞는 후보에 높은 점수(0~1)를 주는지."""

import math

import pytest

from jarviseo.pointing.cues import POINT_SIGMA_DEG, score_center, score_point
from jarviseo.types import BBox, CueKind, Detection

FRAME = (1920, 1080)


def _det(x1: float, y1: float, x2: float, y2: float, label: str = "cup") -> Detection:
    return Detection(label=label, confidence=0.9, bbox=BBox(x1, y1, x2, y2))


def test_center_beats_corner():
    """같은 크기면 가운데 박스가 구석 박스보다 높다."""
    middle = _det(860, 440, 1060, 640)
    corner = _det(0, 0, 200, 200)

    mid_score, corner_score = (s.score for s in score_center([middle, corner], FRAME))

    assert mid_score > corner_score


def test_symmetric_boxes_tie():
    """좌우 대칭인 두 박스는 점수가 같다. 이때 resolver 가 되묻는다."""
    left = _det(400, 440, 600, 640)
    right = _det(1320, 440, 1520, 640)

    left_score, right_score = (s.score for s in score_center([left, right], FRAME))

    assert left_score == pytest.approx(right_score)


def test_bigger_box_wins_at_same_center():
    """중심이 같으면 크게 찍힌(가까운) 물건이 높다."""
    small = _det(910, 490, 1010, 590)
    big = _det(760, 340, 1160, 740)

    small_score, big_score = (s.score for s in score_center([small, big], FRAME))

    assert big_score > small_score


def test_full_frame_box_scores_one():
    """화면 전체를 채운 박스는 가운데이고 가장 크므로 1.0 이다."""
    [full] = score_center([_det(0, 0, *FRAME)], FRAME)
    assert full.score == pytest.approx(1.0)


def test_scores_stay_in_range_and_order():
    """점수는 0~1 이고, 후보 순서와 개수를 그대로 따른다."""
    boxes = [_det(0, 0, 64, 64), _det(860, 440, 1060, 640), _det(1856, 1016, 1920, 1080)]

    scores = score_center(boxes, FRAME)

    assert len(scores) == 3
    assert all(0.0 <= s.score <= 1.0 for s in scores)
    assert scores[1].score == max(s.score for s in scores)


def test_score_kind_and_detail():
    """단서 이름은 center 이고, 실패 분석용으로 위치·크기 점수를 따로 남긴다."""
    [score] = score_center([_det(860, 440, 1060, 640)], FRAME)

    assert score.kind is CueKind.CENTER
    assert set(score.detail) == {"closeness", "size"}


def test_no_candidates():
    """후보가 없으면 빈 리스트다(기권인 '전부 0점'과 다르다)."""
    assert score_center([], FRAME) == []


# --- 손 방향 단서 (v2) -----------------------------------------------------------

# 검지 끝 (0, 0) 에서 오른쪽(+x)을 가리키는 선
RAY = ((0.0, 0.0), (1.0, 0.0))


def _at(cx: float, cy: float, label: str = "cup") -> Detection:
    """중심이 (cx, cy) 인 20×20 박스."""
    return _det(cx - 10, cy - 10, cx + 10, cy + 10, label)


def test_point_straight_ahead_scores_one():
    """선 위에 중심이 있는 후보는 1점이다."""
    [score] = score_point([_at(300, 0)], RAY)
    assert score.score == pytest.approx(1.0)
    assert score.detail["angle"] == pytest.approx(0.0)


def test_point_drops_like_bell_curve():
    """σ 만큼 벗어나면 e^-0.5(약 0.61), 두 배면 e^-2(약 0.14)."""
    one_sigma = _at(100, 100 * math.tan(math.radians(POINT_SIGMA_DEG)))
    two_sigma = _at(100, 100 * math.tan(math.radians(2 * POINT_SIGMA_DEG)))

    first, second = score_point([one_sigma, two_sigma], RAY)

    assert first.score == pytest.approx(math.exp(-0.5))
    assert second.score == pytest.approx(math.exp(-2.0))


def test_point_behind_hand_is_almost_zero():
    """손 뒤쪽(180°) 후보는 0 에 가깝다."""
    [score] = score_point([_at(-300, 0)], RAY)
    assert score.score < 1e-6


def test_point_center_on_fingertip_scores_one():
    """후보 중심이 검지 끝과 같은 자리면 손가락이 그 위에 있는 것이라 1점이다."""
    [score] = score_point([_at(0, 0)], RAY)
    assert score.score == 1.0
    assert score.detail["angle"] is None


def test_point_kind_and_order():
    """단서 이름은 point 이고, 후보 순서와 개수를 그대로 따른다."""
    candidates = [_at(-300, 0), _at(300, 0), _at(300, 300)]

    scores = score_point(candidates, RAY)

    assert [s.kind for s in scores] == [CueKind.POINT] * 3
    assert scores[1].score == max(s.score for s in scores)
    assert all(0.0 <= s.score <= 1.0 for s in scores)


def test_point_no_candidates():
    """후보가 없으면 빈 리스트다."""
    assert score_point([], RAY) == []
