"""단서 점수 — 가운데(v1) · 손 방향(v2) · 말(v4) 단서가 맞는 후보에 높은 점수(0~1)를 주는지.

말 단서는 질문 읽기(parse_lang)와 점수(score_lang)를 따로 본다. 색은 칠한 가짜 사진으로 잰다.
"""

import math

import numpy as np
import pytest

from jarviseo.pointing.cues import (
    POINT_SIGMA_DEG,
    LangQuery,
    parse_lang,
    score_center,
    score_lang,
    score_point,
)
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


# --- 말 단서 (v4): 질문 읽기 ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("노란 캔 먹어도 돼?", LangQuery(frozenset({"product"}), ("yellow",))),
        ("흰색 컵 뭐야?", LangQuery(frozenset({"cup"}), ("white",))),
        ("왼쪽 컵 뭐야?", LangQuery(frozenset({"cup"}), position="left")),
        ("위에 있는 병 뭐야?", LangQuery(frozenset({"bottle"}), position="top")),
        ("큰 병 뭐야?", LangQuery(frozenset({"bottle"}), size="big")),
        ("작은 거 뭐야?", LangQuery(size="small")),
        ("이 폰 뭐야?", LangQuery(frozenset({"cell phone"}))),
        ("이 텀블러 뭐야?", LangQuery(frozenset({"bottle"}))),
        ("보라색 뭐야?", LangQuery(colors=("purple",))),
    ],
)
def test_parse_lang_reads_s04_style_questions(text, expected):
    """말 · 맥락 세션(s04)에서 실제로 한 말 모양 그대로 읽는다.

    별칭(폰 · 텀블러 · 캔)도 알아듣는다.
    """
    assert parse_lang(text) == expected


def test_parse_lang_reads_words_not_letters():
    """낱말 단위로 본다. "책상"은 책이 아니고, "컵라면"은 컵이 아니라 식품이다."""
    assert parse_lang("책상 위 컵 뭐야?").kinds == {"cup"}
    assert parse_lang("이 컵라면 먹어도 돼?").kinds == {"product"}


def test_parse_lang_keeps_adjectives_that_end_like_particles():
    """ "작은" · "검은"의 "은"은 조사가 아니다. 먼저 떼면 "작" · "검"이 되어 못 알아듣는다."""
    assert parse_lang("작은 거 뭐야?").size == "small"
    assert parse_lang("검은 컵 뭐야?").colors == ("black",)


def test_parse_lang_strips_particles():
    """ "위에" · "왼쪽에" · "컵을"처럼 조사가 붙어도 낱말을 찾는다."""
    query = parse_lang("왼쪽에 있는 컵을 줘")
    assert (query.position, query.kinds) == ("left", {"cup"})


@pytest.mark.parametrize(
    ("text", "relation"),
    [
        ("컵 옆에 있는 거 뭐야?", True),  # 컵은 답이 아니라 기준점
        ("컵 위에 있는 거 뭐야?", True),  # 종류 이름 바로 뒤의 위 · 아래도 관계
        ("위에 있는 병 뭐야?", False),  # 앞에 종류 이름이 없으면 위치
    ],
)
def test_parse_lang_tells_relation_from_position(text, relation):
    assert parse_lang(text).relation is relation


def test_parse_lang_without_words_is_empty():
    """ "그거"처럼 쓸 말이 없으면 비어 있다(이어 묻기는 v5 몫)."""
    assert parse_lang("그거 먹어도 돼?").empty
    assert not parse_lang("파란 거 먹어도 돼?").empty


# --- 말 단서 (v4): 점수 --------------------------------------------------------------

YELLOW, WHITE = (0, 255, 255), (255, 255, 255)  # OpenCV 는 BGR 순서


def _photo(*paints: tuple[BBox, tuple[int, int, int]]) -> np.ndarray:
    """검은 1080p 사진에 박스마다 색을 칠한다."""
    image = np.zeros((1080, 1920, 3), np.uint8)
    for box, color in paints:
        image[int(box.y1) : int(box.y2), int(box.x1) : int(box.x2)] = color
    return image


def _scores(candidates, text, image=None):
    found = score_lang(candidates, parse_lang(text), FRAME, image)
    return None if found is None else [round(s.score, 3) for s in found]


def test_lang_kind_word_scores_matching_label():
    """종류 말이 맞는 후보는 1, 아니면 0."""
    candidates = [_det(400, 440, 600, 640, "cup"), _det(1320, 440, 1520, 640, "bottle")]
    assert _scores(candidates, "이 컵 뭐야?") == [1.0, 0.0]


def test_lang_color_looks_inside_the_box():
    """색은 박스 안쪽만 본다. 노란 박스 1, 흰 박스 0, 가장자리만 노란 박스(안쪽은 검정)도 0."""
    full, white, rim = (
        BBox(100, 100, 300, 300),
        BBox(800, 100, 1000, 300),
        BBox(1500, 100, 1700, 300),
    )
    image = _photo((full, YELLOW), (white, WHITE), (rim, YELLOW))
    image[140:260, 1540:1660] = 0  # 사방 20% 가장자리만 노랗게 남기고 안쪽을 검게
    candidates = [Detection(label="can", confidence=0.9, bbox=b) for b in (full, white, rim)]

    assert _scores(candidates, "노란 거 뭐야?", image) == [1.0, 0.0, 0.0]


def test_lang_left_cup_is_left_among_cups():
    """ "왼쪽 컵"은 컵 중 왼쪽이다. 더 왼쪽에 있는 병은 0(곱이라 종류가 틀리면 0)."""
    bottle = _det(0, 440, 200, 640, "bottle")
    left_cup, right_cup = _det(400, 440, 600, 640), _det(1320, 440, 1520, 640)
    assert _scores([bottle, left_cup, right_cup], "왼쪽 컵 뭐야?") == [0.0, 1.0, 0.0]


def test_lang_center_and_size_rank_within_pool():
    """ "가운데 컵"은 컵 무리의 가운데, "큰 병"은 병 중 넓이가 큰 것."""
    cups = [_det(100, 440, 300, 640), _det(860, 440, 1060, 640), _det(1620, 440, 1820, 640)]
    assert _scores(cups, "가운데 컵 뭐야?") == [0.0, 1.0, 0.0]
    bottles = [_det(100, 400, 200, 600, "bottle"), _det(1000, 300, 1300, 900, "bottle")]
    assert _scores(bottles, "큰 병 뭐야?") == [0.0, 1.0]


def test_lang_detail_records_each_part():
    """기록(cue_scores)에 무엇이 점수를 냈는지 남는다."""
    cups = [_det(400, 440, 600, 640), _det(1320, 440, 1520, 640)]
    found = score_lang(cups, parse_lang("왼쪽 컵 뭐야?"), FRAME)
    assert found[0].kind is CueKind.LANG
    assert found[0].detail == {"kind": 1.0, "position": 1.0}


@pytest.mark.parametrize(
    "text",
    [
        "컵 옆에 있는 거 뭐야?",  # 관계 말
        "저거 뭐야?",  # 쓸 말이 없음
        "노란 거 뭐야?",  # 색만 말했는데 사진이 없음
        "이 컵 뭐야?",  # 둘 다 컵이라 못 가름
    ],
)
def test_lang_abstains_instead_of_zero(text):
    """기권(None)한다. 모두에게 같은 점수를 주면 총점 차이만 줄어 되묻기가 는다."""
    cups = [_det(400, 440, 600, 640), _det(1320, 440, 1520, 640)]
    assert score_lang(cups, parse_lang(text), FRAME, image=None) is None


def test_lang_no_candidates_abstains():
    assert score_lang([], parse_lang("왼쪽 컵 뭐야?"), FRAME) is None
