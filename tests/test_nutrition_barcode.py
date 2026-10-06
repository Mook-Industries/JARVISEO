"""바코드 → 제품 조회 테스트.

실제 네트워크·카메라 없이 돈다.
- 바코드 이미지는 EAN-13 규격대로 직접 그린다(사진 파일을 커밋하지 않으려고).

실행: pytest tests/test_nutrition_barcode.py
"""

import numpy as np
import pytest

from jarviseo.nutrition.barcode import is_valid_gtin, read_barcodes

# 체크섬이 맞는 실제 형식의 번호 (880 = 대한민국)
CODE = "8801234567893"

# --- EAN-13 그리기 ---------------------------------------------------------
_L = ["0001101", "0011001", "0010011", "0111101", "0100011",
      "0110001", "0101111", "0111011", "0110111", "0001011"]  # fmt: skip
_R = ["".join("1" if b == "0" else "0" for b in p) for p in _L]
_G = [p[::-1] for p in _R]
_PARITY = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
           "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]  # fmt: skip


def draw_ean13(code: str, module: int = 4, height: int = 160) -> np.ndarray:
    """흰 바탕에 EAN-13 막대를 그린 BGR 이미지."""
    d = [int(c) for c in code]
    left = "".join((_L if p == "L" else _G)[x] for p, x in zip(_PARITY[d[0]], d[1:7], strict=True))
    bits = "101" + left + "01010" + "".join(_R[x] for x in d[7:]) + "101"
    quiet = 12 * module
    width = len(bits) * module + 2 * quiet
    img = np.full((height + 40, width), 255, np.uint8)
    for i, b in enumerate(bits):
        if b == "1":
            x = quiet + i * module
            img[20 : 20 + height, x : x + module] = 0
    return np.dstack([img] * 3)


# --- 체크섬 ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "ok"),
    [
        (CODE, True),
        ("8801234567890", False),  # 마지막 자리만 틀림
        ("96385074", True),  # EAN-8
        ("036000291452", True),  # UPC-A
        ("88012345678a3", False),
        ("12345", False),
    ],
)
def test_gtin_체크섬(code, ok):
    assert is_valid_gtin(code) is ok


# --- 이미지에서 읽기 --------------------------------------------------------


def test_그린_바코드를_읽는다():
    reads = read_barcodes(draw_ean13(CODE))
    assert [r.code for r in reads] == [CODE]
    assert reads[0].symbology == "EAN_13"
    assert reads[0].bbox is not None


@pytest.mark.parametrize(("module", "height"), [(1, 40), (3, 150), (6, 300), (12, 600)])
def test_막대_굵기와_상관없이_읽는다(module, height):
    # 크롭이 바코드에 딱 맞게 잘려 막대가 아주 가늘거나 굵어도 읽어야 한다.
    reads = read_barcodes(draw_ean13(CODE, module=module, height=height))
    assert [r.code for r in reads] == [CODE]


def test_큰_크롭_속_작은_바코드도_읽는다():
    scene = np.full((900, 1200, 3), 200, np.uint8)
    bc = draw_ean13(CODE, module=3, height=150)
    scene[300 : 300 + bc.shape[0], 400 : 400 + bc.shape[1]] = bc
    reads = read_barcodes(scene)
    assert [r.code for r in reads] == [CODE]
    # 좌표는 원본 크롭 기준이어야 한다 (여백·배율을 되돌렸는지)
    box = reads[0].bbox
    assert 380 <= box.x1 <= 480 and 280 <= box.y1 <= 340


def test_바코드가_없으면_빈_목록():
    assert read_barcodes(np.full((300, 400, 3), 255, np.uint8)) == []
    assert read_barcodes(np.zeros((0, 0, 3), np.uint8)) == []
