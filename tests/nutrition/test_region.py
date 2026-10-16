"""성분표 영역 자동 찾기 테스트.  OCR 없이 가짜 글자 상자로 돈다."""

import numpy as np

from jarviseo.nutrition.region import find_region, locate_panel, word_boxes
from jarviseo.types import BBox

SHAPE = (800, 600, 3)

TITLE = (BBox(50, 20, 550, 80), "맛있는 과자")  # 제품명 큰 글씨 (높이 60)
INGREDIENTS = [
    (BBox(50, 200, 550, 220), "원재료명: 밀가루(미국산), 설탕"),
    (BBox(50, 225, 500, 245), "탈지분유, 대두유, 쇼트닝"),
    (BBox(50, 250, 300, 270), "우유, 대두, 밀 함유"),
]
NUTRITION = [
    (BBox(50, 300, 200, 320), "영양정보"),
    (BBox(50, 325, 250, 345), "나트륨 115mg"),
]


def inside(outer: BBox, inner: BBox) -> bool:
    return (
        outer.x1 <= inner.x1
        and outer.y1 <= inner.y1
        and outer.x2 >= inner.x2
        and outer.y2 >= inner.y2
    )


def test_keeps_ingredient_block_only():
    """원재료 덩어리만 잡고 제품명과 영양성분표는 뺀다."""
    region, names = find_region([TITLE, *INGREDIENTS, *NUTRITION], SHAPE)
    assert all(inside(region, b) for b, _ in INGREDIENTS)
    assert region.y1 > TITLE[0].y2  # 제품명 제외
    assert region.y2 < NUTRITION[0][0].y1  # 영양성분표에서 멈춘다
    assert names == ["함유", "원재료", "원산지"]


def test_no_cue_uses_whole_photo():
    """단서가 없으면 사진 전체를 쓴다."""
    region, names = find_region([TITLE, *NUTRITION], SHAPE)
    assert region == BBox(0, 0, 600, 800) and names == []


def test_content_amount_is_not_a_cue():
    """함유량은 단서가 아니다."""
    _, names = find_region([(BBox(50, 200, 300, 220), "단백질 함유량 3g")], SHAPE)
    assert names == []


def test_joins_nearby_allergen_label_in_same_column():
    """같은 단의 알레르기 표시는 조금 떨어져도 붙인다."""
    notice = (BBox(60, 340, 400, 360), "알레르기 유발물질: 우유, 대두")  # 원재료 끝에서 70px(3.5줄)
    side = (BBox(540, 340, 600, 360), "뒷면참조")  # 같은 거리지만 원재료 단과 거의 안 겹친다
    region, names = find_region([*INGREDIENTS, notice, side], SHAPE)
    assert inside(region, notice[0])
    assert region.x2 < side[0].x2  # 옆 단은 붙이지 않는다
    assert "알레르기" in names


def test_empty_photo_does_not_crash():
    """빈 사진에서도 멈추지 않는다."""
    region, names = find_region([], SHAPE)
    assert region == BBox(0, 0, 600, 800) and names == []


class ScaledReader:
    """받은 이미지 크기를 기록하고, 그 크기 기준 좌표로 상자를 돌려주는 가짜 OCR."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.shape = None

    def readtext(self, image, **kwargs):
        self.shape = image.shape
        k = image.shape[0] / SHAPE[0]
        return [
            (
                [
                    [b.x1 * k, b.y1 * k],
                    [b.x2 * k, b.y1 * k],
                    [b.x2 * k, b.y2 * k],
                    [b.x1 * k, b.y2 * k],
                ],
                t,
                0.9,
            )
            for b, t in self.boxes
        ]


def test_upscales_small_photo_and_maps_coords_back():
    """작은 사진은 키워 읽고 좌표는 원래 크기로 돌린다."""
    reader = ScaledReader([*INGREDIENTS, (BBox(0, 0, 10, 10), "  ")])
    boxes = word_boxes(np.zeros(SHAPE, np.uint8), reader)
    assert reader.shape[0] == 1600  # 긴 변 800 → 1600
    assert len(boxes) == 3  # 빈 글자는 버린다
    for (got, _), (want, _) in zip(boxes, INGREDIENTS, strict=True):
        assert abs(got.x1 - want.x1) < 1e-6 and abs(got.y2 - want.y2) < 1e-6


def test_locate_panel_runs_ocr_to_region():
    """`locate_panel` 은 OCR 부터 영역까지 한 번에 한다."""
    reader = ScaledReader([TITLE, *INGREDIENTS, *NUTRITION])
    region, names = locate_panel(np.zeros(SHAPE, np.uint8), reader)
    assert all(inside(region, b) for b, _ in INGREDIENTS) and "함유" in names
