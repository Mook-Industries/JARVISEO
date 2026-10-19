"""성분표가 잘렸는지 찾는다.  담당: 권용현

성분표 일부가 사진 밖으로 나갔거나 접힌 면에서 잘리면, 보이는 부분만 읽고 '못 찾았어요'라고
말하게 된다. 잘린 자리에 땅콩이 있었다면 위험한 말이다.
그래서 OCR 줄을 보고 잘림을 알아낸 뒤, 판정 문장에 그 사실을 덧붙인다.

찾는 두 가지 (평가 사진 22장에서 오탐 0~1장으로 확인한 것만 넣었다, 2026-10-19)
-------------------------------------------------------------------------------
1. 머리말 잘림(cut_header): "원재료명"이 "재료명"·"대료명"처럼 첫 글자 없이 읽혔다.
   성분표 왼쪽이 접힌 면이나 다른 물건에 잘려, 줄마다 앞 글자가 없다는 뜻이다(label_01 실측).
2. 가장자리 잘림(cut_edge): 성분표 글줄이 사진 가장자리에 닿아 있다. 바깥쪽에 글자가 더 있다.
   성분표 영역만 잘라 읽었으면, 그 영역의 변이 사진 가장자리일 때만 본다.
   포장의 다른 글자가 가장자리에서 잘린 것은 상관없다(성분표 줄만 본다).

손가락이 가운데를 가린 경우(줄 사이 빈 틈)는 아직 못 찾는다. OCR 이 옆 칸 영양정보와
원재료 줄을 한 줄로 합쳐 읽어서 줄 간격이 들쭉날쭉하고, 원래 빈칸이 있는 포장(label_16)과
구별이 안 됐다. 실제로 가린 사진을 모은 뒤 다시 본다.

LLM 을 쓰지 않는다. 줄 글자와 좌표만 본다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from jarviseo.types import OCRLine

__all__ = ["CUT_EDGE", "CUT_HEADER", "Borders", "find_occlusion"]

CUT_EDGE = "cut_edge"
CUT_HEADER = "cut_header"

# 가장자리에 닿았다고 보는 거리: 변 길이의 1% (최소 3px).
_EDGE_RATIO = 0.01
# "원재료명"의 '원'이 빠진 머리말. OCR 이 잘린 '재'를 '대'로 읽기도 한다.
# 앞이 '원'이면 잘린 게 아니다.
_CUT_HEADER = re.compile(r"(?<!원)(?<!원\s)[재대]\s*료\s*명")


@dataclass(frozen=True)
class Borders:
    """읽은 이미지의 각 변이 원래 사진의 가장자리인가. 영역만 잘라 읽었으면 일부만 True."""

    top: bool = True
    right: bool = True
    bottom: bool = True
    left: bool = True

    def rotated(self, angle: int) -> Borders:
        """이미지를 시계 방향으로 angle 만큼 돌려 읽었을 때, 돌린 이미지 기준의 변."""
        if angle == 90:  # 돌린 이미지의 위 = 원래 왼쪽
            return Borders(self.left, self.top, self.right, self.bottom)
        if angle == 270:  # 돌린 이미지의 위 = 원래 오른쪽
            return Borders(self.right, self.bottom, self.left, self.top)
        return self


def _key(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _in(line: OCRLine, text: str) -> bool:
    """줄이 성분표 글에 속하는가. 줄의 첫 낱말(두 글자 이상)이 그 글에 있으면 속한다.

    구간을 나눌 때 "함유" 같은 말은 빠지므로 줄 전체가 아니라 첫 낱말로 본다.
    """
    for word in re.split(r"[\s,.:()]+", line.text):
        if len(word) >= 2:
            return word in text
    return False


def find_occlusion(
    lines: list[OCRLine],
    width: int,
    height: int,
    panel_text: str,
    borders: Borders | None = None,
) -> str:
    """잘림 사유 (CUT_HEADER / CUT_EDGE). 잘린 데가 없으면 빈 문자열.

    Args:
        lines: 읽은 이미지 기준 OCR 줄 (위→아래).
        width, height: 읽은 이미지 크기.
        panel_text: 성분표 글 전체 (원재료명 + 알레르기 표시 + 혼입 문장).
        borders: 읽은 이미지의 어느 변이 원래 사진의 가장자리인가. 없으면 네 변 모두.
    """
    if any(_CUT_HEADER.search(ln.text) for ln in lines):
        return CUT_HEADER

    borders = borders or Borders()
    panel_key = _key(panel_text)
    mx, my = max(3.0, width * _EDGE_RATIO), max(3.0, height * _EDGE_RATIO)
    for ln in lines:
        if not _in(ln, panel_key):
            continue
        b = ln.bbox
        if (
            (borders.left and b.x1 <= mx)
            or (borders.right and b.x2 >= width - mx)
            or (borders.top and b.y1 <= my)
            or (borders.bottom and b.y2 >= height - my)
        ):
            return CUT_EDGE
    return ""
