"""가운데 단서 — 가운데·크게 찍힌 후보가 높은 점수를 받는지, 점수가 0~1 인지."""

import pytest

from jarviseo.pointing.cues import score_center
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
