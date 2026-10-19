"""크롭 사진에서 성분표(원재료명 ~ 알레르기 표시) 영역을 자동으로 찾는다.  (#75)

크롭 사진에는 성분표 말고도 제품명 큰 글씨, 영양성분표, 조리법이 같이 찍힌다.
통째로 OCR 하면 글자가 덜 커지고, 영양성분표 낱말이 원재료 구간에 섞여 구간 분리가 흐트러진다.
그래서 먼저 성분표가 있는 곳만 잘라 낸 뒤 그 영역을 크게 키워 읽는다.

방법 (규칙 기반, 학습 없음)
1) 사진 전체를 한 번 OCR 해서 글자 상자를 얻는다(``word_boxes``).
2) 성분표 단서('함유', '원재료명', '알레르기', '제조시설', 원산지 'OO산')가 든 상자를
   씨앗으로 고른다.
3) 영양성분표('영양정보', '나트륨 115mg' 같은 칸)를 위아래 경계로 삼는다.
4) 씨앗에서 시작해 본문 크기의 글자만 붙인다.
   - 1단계: 가까운 상자 (좌우·위아래)
   - 2단계: 같은 단(가로로 겹치는) 상자는 위아래로 조금 더 멀어도 붙인다.
     알레르기 표시가 테두리 상자 안에 따로 인쇄된 경우를 잡기 위해서다.
5) 단서를 하나도 못 찾으면 사진 전체를 쓴다(호출한 쪽이 기존처럼 통째로 읽는다).

평가 세트 21장(Colab T4, PP-OCRv5): 알레르겐 재현율 사람이 그린 영역 81% / 자동 영역 76%.
수치와 실패 사례는 docs/experiments.md 에 적는다.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence

import cv2
import numpy as np

from jarviseo.types import BBox

__all__ = ["ANCHORS", "PARAMS", "find_region", "locate_panel", "word_boxes"]

WordBox = tuple[BBox, str]

ANCHORS = {
    "함유": re.compile(r"함\s*유(?![량될할])"),
    "원재료": re.compile(r"원\s*재\s*료|재\s*료\s*명"),
    "알레르기": re.compile(r"알\s*레\s*\S\s*기|유\s*발\s*물\s*질"),
    "제조시설": re.compile(r"제\s*조\s*시\s*설|혼\s*입"),
    # 원재료 목록에는 원산지가 거의 항상 붙는다. OCR 이 '함유'를 잘못 읽었을 때의 예비 단서.
    "원산지": re.compile(
        r"(?:국내|국|외국|수입|미국|호주|캐나다|중국|베트남|태국|칠레|브라질|독일|프랑스|스페인|이탈리아"
        r"|네덜란드|덴마크|뉴질랜드|인도네시아|말레이시아|필리핀|아르헨티나|러시아|헝가리|폴란드|일본|멕시코)\s*산"
    ),
}
# 경계: 영양성분표. 원재료와 섞이면 안 되는 덩어리다.
# '글루탐산나트륨'처럼 원재료에도 나오는 낱말이라 '나트륨 115mg'처럼 숫자가 붙은 칸만 본다.
# 포장재질·품목보고번호 같은 다른 머리말은 경계로 쓰지 않는다. 원재료와 혼입 문구 사이에
# 끼어 있는 성분표가 있어서, 경계로 쓰면 혼입 문구를 잘라 버린다.
NUTRITION = re.compile(
    r"영\s*양\s*(?:정\s*보|성\s*분)|\d\s*(?:k\s*c\s*a\s*l|m\s*g)"
    r"|(?:나\s*트\s*륨|탄\s*수\s*화\s*물|단\s*백\s*질|콜\s*레\s*스|포\s*화\s*지\s*방|트\s*랜\s*스|당\s*류|열\s*량)\s*\d"
)
# 글자 높이(h) 배수. 평가 세트로 고른 값.
#   size_lo/hi: 씨앗 글자 높이의 이 범위만 본문으로 본다(제품명 큰 글씨·작은 각주 제외)
#   gap_x/gap_y: 1단계에서 붙일 거리, gap_y2: 2단계(같은 단)에서 위아래로 붙일 거리
#   pad: 최종 영역 여백
PARAMS = dict(size_lo=0.6, size_hi=1.8, gap_x=3.0, gap_y=2.0, gap_y2=5.0, pad=0.8)


def word_boxes(image: np.ndarray, reader, target: int = 1600) -> list[WordBox]:
    """사진 전체 OCR → (상자, 글자) 목록. 좌표는 원본 크기 기준.

    크롭 사진이 500px 정도면 성분표 글자가 8px 밖에 안 돼 1배로는 거의 못 찾는다.
    그래서 긴 변이 ``target`` 이 되게 키워서(최대 3배) 읽는다.
    """
    scale = min(3.0, max(1.0, target / max(image.shape[:2])))
    if scale > 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    out: list[WordBox] = []
    for pts, text, _conf in reader.readtext(image, detail=1, paragraph=False):
        text = str(text).strip()
        if not text:
            continue
        xs = [float(p[0]) / scale for p in pts]
        ys = [float(p[1]) / scale for p in pts]
        out.append((BBox(min(xs), min(ys), max(xs), max(ys)), text))
    return out


def _union(boxes: Sequence[BBox]) -> BBox:
    return BBox(
        min(b.x1 for b in boxes),
        min(b.y1 for b in boxes),
        max(b.x2 for b in boxes),
        max(b.y2 for b in boxes),
    )


def _line_h(b: BBox) -> float:
    """글자 높이. 세로쓰기 상자도 있어서 짧은 변을 쓴다."""
    return max(1.0, min(b.width, b.height))


def _x_overlap(a: BBox, b: BBox) -> float:
    """가로로 겹친 길이 ÷ 좁은 쪽 폭. 좁은 '함유' 줄 위의 넓은 원재료 줄도 같은 단으로 본다."""
    return max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) / max(1.0, min(a.width, b.width))


def _limits(boxes, members, anchors, stops) -> tuple[float, float]:
    """단서 상자 기준으로 가장 가까운 위쪽·아래쪽 영양성분표 위치."""
    top = min(boxes[i][0].y1 for i in anchors)
    bottom = max(boxes[i][0].y2 for i in anchors)
    col = _union([boxes[i][0] for i in members])
    above = [
        boxes[i][0].y2
        for i in stops
        if boxes[i][0].y2 <= top + 1 and _x_overlap(col, boxes[i][0]) > 0
    ]
    below = [
        boxes[i][0].y1
        for i in stops
        if boxes[i][0].y1 >= bottom - 1 and _x_overlap(col, boxes[i][0]) > 0
    ]
    return (max(above) if above else float("-inf")), (min(below) if below else float("inf"))


def _grow(boxes, seed: int, p: dict, stop_idx: set[int], anchor_idx: set[int]) -> set[int]:
    h = _line_h(boxes[seed][0])
    members = {seed}
    lo, hi = p["size_lo"] * h, p["size_hi"] * h
    ok = [i for i, (b, _) in enumerate(boxes) if lo <= _line_h(b) <= hi and i not in stop_idx]
    for gap_y, same_column in ((p["gap_y"], False), (p["gap_y2"], True)):
        changed = True
        while changed:
            changed = False
            region = _union([boxes[i][0] for i in members])
            anchors = [i for i in members if i in anchor_idx] or [seed]
            top, bottom = _limits(boxes, members, anchors, stop_idx)
            gx, gy = p["gap_x"] * h, gap_y * h
            for i in ok:
                if i in members:
                    continue
                b = boxes[i][0]
                if not top <= b.center[1] <= bottom:
                    continue
                if (
                    b.x2 < region.x1 - gx
                    or b.x1 > region.x2 + gx
                    or b.y2 < region.y1 - gy
                    or b.y1 > region.y2 + gy
                ):
                    continue
                if same_column and _x_overlap(region, b) < 0.5:
                    continue
                members.add(i)
                changed = True
    return members


def find_region(
    boxes: Sequence[WordBox], shape: tuple[int, ...], **params: float
) -> tuple[BBox, list[str]]:
    """글자 상자 → (성분표 영역, 찾은 단서 이름). 단서가 없으면 (사진 전체, [])."""
    p = {**PARAMS, **params}
    hgt, wid = shape[:2]
    whole = BBox(0, 0, wid, hgt)
    hits = {i: [n for n, pat in ANCHORS.items() if pat.search(t)] for i, (_, t) in enumerate(boxes)}
    anchor_idx = {i for i, names in hits.items() if names}
    if not anchor_idx:
        return whole, []
    stop_idx = {i for i, (_, t) in enumerate(boxes) if NUTRITION.search(t)} - anchor_idx

    # 단서마다 덩어리를 키워 보고, 단서 종류가 가장 많이 모인 덩어리를 고른다(같으면 큰 쪽).
    clusters = [_grow(boxes, s, p, stop_idx, anchor_idx) for s in sorted(anchor_idx)]
    best = max(clusters, key=lambda c: (len({n for i in c for n in hits[i]}), len(c)))
    region = _union([boxes[i][0] for i in best])
    pad = p["pad"] * statistics.median(_line_h(boxes[i][0]) for i in best)
    names = [n for n in ANCHORS if any(n in hits[i] for i in best)]
    return (
        BBox(
            max(0.0, region.x1 - pad),
            max(0.0, region.y1 - pad),
            min(float(wid), region.x2 + pad),
            min(float(hgt), region.y2 + pad),
        ),
        names,
    )


def locate_panel(image: np.ndarray, reader=None, **params: float) -> tuple[BBox, list[str]]:
    """사진 → (성분표 영역, 찾은 단서 이름). ``word_boxes`` + ``find_region``.

    reader 가 없으면 ocr.py 의 기본 엔진을 쓴다(성분표 읽기와 같은 모델을 한 번만 띄운다).
    """
    from jarviseo.nutrition.ocr import _get_reader

    return find_region(word_boxes(image, reader or _get_reader()), image.shape, **params)
