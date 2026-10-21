"""사진 위에 찾은 자리 그리기. (글줄 좌표 되돌리기는 tests/nutrition/test_ocr_lines.py)"""

import numpy as np

from jarviseo.api.marks import mark_findings, matching_lines
from jarviseo.types import BBox, OCRLine


def line(text, x1, y1, x2, y2):
    return OCRLine(text=text, confidence=0.9, bbox=BBox(x1, y1, x2, y2))


def test_matching_lines_ignores_spaces():
    lines = [line("원재료명 소 맥분, 설탕", 0, 0, 10, 10), line("우유 함유", 0, 20, 10, 30)]
    assert matching_lines(lines, ["소맥분"]) == [lines[0]]


def test_mark_findings_draws_only_found_allergens():
    img = np.full((100, 200, 3), 255, np.uint8)
    lines = [line("소맥분, 설탕", 10, 10, 190, 30), line("유통기한 뒷면", 10, 60, 190, 80)]
    ev = [
        {
            "allergen": "밀",
            "status": "contains",
            "where": [{"section": "원재료", "terms": ["소맥분"]}],
        },
        {"allergen": "우유", "status": "not_found", "where": []},
    ]
    out = mark_findings(img, lines, ev)
    assert out is not None
    assert (out[20, 100] != 255).any()  # 찾은 줄은 칠했다
    assert (out[70, 100] == 255).all()  # 못 찾은 줄은 그대로


def test_mark_findings_nothing_to_draw():
    img = np.full((50, 50, 3), 255, np.uint8)
    assert mark_findings(img, [], [{"allergen": "밀", "status": "not_found", "where": []}]) is None


def test_mark_findings_draws_barcode():
    img = np.full((100, 200, 3), 255, np.uint8)
    out = mark_findings(img, [], [], barcode=BBox(20, 20, 120, 60), barcode_code="880")
    assert out is not None and (out[20, 60] != 255).any()
