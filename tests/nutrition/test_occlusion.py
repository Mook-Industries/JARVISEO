"""성분표 가려짐·잘림 찾기 테스트. OCR 줄 좌표만 만들어 넣는다.

실행: pytest tests/nutrition/test_occlusion.py
"""

from jarviseo.nutrition.occlusion import CUT_EDGE, CUT_HEADER, Borders, find_occlusion
from jarviseo.types import BBox, OCRLine

W, H = 800, 600
INGR = "밀가루, 설탕, 대두, 정제소금, 탈지분유"
PANEL = INGR + "우유, 대두 함유"


def line(text: str, y: int, x1: int = 50, x2: int = 700, h: int = 20) -> OCRLine:
    return OCRLine(text=text, confidence=0.9, bbox=BBox(x1, y, x2, y + h))


LINES = [
    line("원재료명:", 100),
    line("밀가루, 설탕, 대두,", 130),
    line("정제소금, 탈지분유", 160),
    line("우유, 대두 함유", 190),
]


def test_occlusion_none_when_panel_fully_visible():
    """성분표가 다 보이면 가려짐이 없다."""
    assert find_occlusion(LINES, W, H, PANEL) == ""


def test_occlusion_cut_when_panel_line_touches_bottom():
    """성분표 줄이 사진 아래 가장자리에 닿으면 잘린 것이다."""
    lines = [*LINES[:3], line("우유, 대두 함유", H - 20)]
    assert find_occlusion(lines, W, H, PANEL) == CUT_EDGE


def test_occlusion_cut_when_panel_line_touches_side():
    """성분표 줄이 옆 가장자리에 닿아도 잘린 것이다."""
    lines = [LINES[0], line("밀가루, 설탕, 대두,", 130, x2=W), *LINES[2:]]
    assert find_occlusion(lines, W, H, PANEL) == CUT_EDGE


def test_occlusion_ignores_edge_inside_photo():
    """영역만 잘라 읽었고 그 변이 사진 안쪽이면, 가장자리에 닿아도 잘림이 아니다."""
    lines = [line(ln.text, int(ln.bbox.y1) + H - 210) for ln in LINES]  # 아래 가장자리에 붙인다
    inside = Borders(top=True, right=True, bottom=False, left=True)
    assert find_occlusion(lines, W, H, PANEL, inside) == ""


def test_occlusion_ignores_other_text_on_edge():
    """성분표가 아닌 포장 글자가 가장자리에서 잘린 것은 상관없다."""
    lines = [line("맛있는 라면 신제품", 0), *LINES]
    assert find_occlusion(lines, W, H, PANEL) == ""


def test_occlusion_borders_follow_rotation():
    """시계 방향 90도로 돌려 읽으면 돌린 이미지의 위는 원래 왼쪽이다."""
    b = Borders(top=False, right=False, bottom=False, left=True)
    assert b.rotated(90) == Borders(top=True, right=False, bottom=False, left=False)
    assert b.rotated(270) == Borders(top=False, right=False, bottom=True, left=False)
    assert b.rotated(0) == b


def test_occlusion_cut_header_without_first_letter():
    """머리말이 '원' 없이 '재료명'·'대료명'으로 읽히면 성분표 왼쪽이 잘린 것이다 (label_01)."""
    for head in ("재료명 밀가루", "대료명 밀가루(밀 미국산)"):
        lines = [line(head, 100), *LINES[1:]]
        assert find_occlusion(lines, W, H, PANEL) == CUT_HEADER


def test_occlusion_full_header_is_not_cut():
    """'원재료명'이 다 보이면 머리말 잘림이 아니다 (띄어 읽어도)."""
    for head in ("원재료명 밀가루", "원 재료명: 밀가루"):
        lines = [line(head, 100), *LINES[1:]]
        assert find_occlusion(lines, W, H, PANEL) == ""


def test_occlusion_none_with_overlapping_boxes():
    """상자가 겹쳐도 줄 간격이 고르면 가려짐이 아니다."""
    ys = [442, 458, 474, 490, 506]
    words = ["밀가루,", "설탕,", "대두,", "정제소금,", "탈지분유"]
    lines = [line(w, y, h=25) for w, y in zip(words, ys, strict=True)]
    assert find_occlusion(lines, W, H, PANEL) == ""
