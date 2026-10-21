"""OCR 글줄 좌표를 check() 에 들어온 사진 기준으로 되돌린다 (IngredientCheck.ocr_lines)."""

from jarviseo.nutrition.ocr import PanelSections
from jarviseo.nutrition.service import _OcrPass
from jarviseo.types import BBox, OCRLine


def line(text, x1, y1, x2, y2):
    return OCRLine(text=text, confidence=0.9, bbox=BBox(x1, y1, x2, y2))


def ocr_pass(lines, angle, size, region, zoom):
    return _OcrPass(lines, PanelSections(), 0.9, angle, size, region=region, zoom=zoom)


def test_source_lines_undo_zoom_and_region():
    """영역 (100, 50) 부터 2배로 키워 읽은 줄을 원본 좌표로 되돌린다."""
    p = ocr_pass([line("밀", 20, 40, 60, 80)], 0, (400, 200), BBox(100, 50, 300, 150), (2.0, 2.0))
    assert p.source_lines()[0].bbox == BBox(110, 70, 130, 90)


def test_source_lines_undo_rotation():
    """시계 방향 90° 로 돌려 읽은 줄을 돌리기 전 자리로 되돌린다."""
    # 확대본 200×100 (가로×세로) → 돌린 뒤 100×200. 돌린 사진의 (10, 20)~(30, 40) 은
    # 확대본의 x = y' = 20~40, y = H-1-x' = 99-30 ~ 99-10 = 69~89 다.
    p = ocr_pass([line("밀", 10, 20, 30, 40)], 90, (100, 200), BBox(0, 0, 200, 100), (1.0, 1.0))
    assert p.source_lines()[0].bbox == BBox(20, 69, 40, 89)
