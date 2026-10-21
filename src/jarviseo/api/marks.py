"""성분 판정 결과를 사진 위에 표시한다 — 어디서 찾았는지.  담당: 권용현

- 알레르기를 찾은 성분표 글줄: 들어 있음 = 주황 칠, 섞였을 수 있음 = 주황 테두리, 위에 알레르기 이름
  (함유 표시에는 없고 원재료에서만 찾은 이름은 뒤에 '?')
- 읽은 바코드: 파란 박스
좌표는 판정기(IngredientChecker.check)에 들어간 사진 기준이다(IngredientCheck.ocr_lines).
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

from jarviseo.nutrition.allergen import normalize_term
from jarviseo.types import BBox, OCRLine

__all__ = ["jpeg_data_url", "mark_findings", "matching_lines"]

ORANGE = (40, 160, 255)  # BGR
BLUE = (255, 150, 40)
_FONTS = ("malgun.ttf", "C:/Windows/Fonts/malgun.ttf", "NanumGothic.ttf", "AppleSDGothicNeo.ttc")


def matching_lines(lines: list[OCRLine], terms: list[str]) -> list[OCRLine]:
    """찾은 표기(예: '소맥분')가 든 글줄. 판정처럼 공백을 지우고 비교한다."""
    keys = [normalize_term(t) for t in terms if t]
    return [ln for ln in lines if any(k and k in normalize_term(ln.text) for k in keys)]


def mark_findings(
    image: np.ndarray,
    lines: list[OCRLine],
    evidence: list[dict],
    barcode: BBox | None = None,
    barcode_code: str | None = None,
) -> np.ndarray | None:
    """사진에 찾은 자리를 그린다. 그릴 것이 없으면 None."""
    marks: list[tuple[BBox, str, bool]] = []  # (박스, 이름, 칠할까)
    by_line: dict[int, tuple[list[str], bool]] = {}
    for e in evidence:
        if e["status"] == "not_found":
            continue
        terms = [t for w in e["where"] for t in w["terms"]]
        for ln in matching_lines(lines, terms):
            names, fill = by_line.setdefault(id(ln), ([], False))
            name = e["allergen"] + ("?" if e.get("ingredient_only") else "")
            if name not in names:
                names.append(name)
            by_line[id(ln)] = (names, fill or e["status"] == "contains")
    for ln in lines:
        if id(ln) in by_line:
            names, fill = by_line[id(ln)]
            marks.append((ln.bbox, ", ".join(names), fill))
    if not marks and barcode is None:
        return None

    out = image.copy()
    h, w = out.shape[:2]
    thick = max(2, round(max(h, w) / 400))
    overlay = out.copy()
    for box, _, fill in marks:
        p1, p2 = (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))
        if fill:
            cv2.rectangle(overlay, p1, p2, ORANGE, -1)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    labels: list[tuple[tuple[int, int], str, tuple[int, int, int]]] = []
    for box, name, _ in marks:
        p1, p2 = (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))
        cv2.rectangle(out, p1, p2, ORANGE, thick)
        labels.append(((p1[0], max(0, p1[1] - 4)), name, ORANGE))
    if barcode is not None:
        p1, p2 = (int(barcode.x1), int(barcode.y1)), (int(barcode.x2), int(barcode.y2))
        cv2.rectangle(out, p1, p2, BLUE, thick + 1)
        labels.append(((p1[0], max(0, p1[1] - 4)), f"바코드 {barcode_code or ''}".strip(), BLUE))
    return _put_labels(out, labels)


def _put_labels(image: np.ndarray, labels) -> np.ndarray:
    """박스 위에 한글 이름을 쓴다. cv2.putText 는 한글을 못 써서 Pillow 로 쓴다."""
    if not labels:
        return image
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return image
    size = max(14, round(max(image.shape[:2]) / 45))
    font = None
    for name in _FONTS:
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    if font is None:
        return image  # 한글 글꼴이 없으면 박스만 남긴다
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for (x, y), text, bgr in labels:
        rgb = (bgr[2], bgr[1], bgr[0])
        left, top, right, bottom = draw.textbbox((x, y), text, font=font, anchor="ld")
        draw.rectangle((left - 3, top - 3, right + 3, bottom + 3), fill=rgb)
        draw.text((x, y), text, font=font, fill=(0, 0, 0), anchor="ld")
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def jpeg_data_url(image: np.ndarray | None, max_side: int = 960) -> str | None:
    """화면에 보여 줄 작은 JPEG data URL."""
    if image is None:
        return None
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        image = cv2.resize(
            image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None
