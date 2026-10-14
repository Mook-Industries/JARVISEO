"""OCR 전 사진 품질 검사 테스트. 글자를 그린 합성 이미지로 본다.

실행: pytest tests/nutrition/test_quality.py
"""

import cv2
import numpy as np
import pytest

from jarviseo.nutrition.quality import assess_quality, best_frame


def text_image(width: int = 600, lines: int = 8) -> np.ndarray:
    """성분표처럼 작은 글자가 여러 줄 있는 이미지 (영문으로 그린다. 한글 글꼴이 없어도 되게)."""
    img = np.full((lines * 28 + 20, width, 3), 235, np.uint8)
    for i in range(lines):
        cv2.putText(img, "Ingredients: wheat flour, sugar, milk, soy 12%", (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)  # fmt: skip
    return img


def test_선명한_글자_사진은_통과한다():
    q = assess_quality(text_image())
    assert q.ok, q


@pytest.mark.parametrize("sigma", [1.5, 3.0, 6.0])
def test_흐린_사진은_걸러진다(sigma):
    q = assess_quality(cv2.GaussianBlur(text_image(), (0, 0), sigma))
    assert q.reason == "blurry", q


def test_반사가_심하면_걸러진다():
    img = text_image()
    h, w = img.shape[:2]
    img[:, : w // 3] = 255  # 왼쪽 1/3 이 하얗게 날아감
    q = assess_quality(img)
    assert q.reason == "glare", q


def test_단색_이미지는_흐린_것으로_본다():
    assert assess_quality(np.full((100, 100, 3), 128, np.uint8)).reason == "blurry"


def test_큰_사진도_같은_기준으로_잰다():
    # 해상도가 커져도(카메라 원본) 선명한 사진이 흐린 것으로 나오면 안 된다
    big = cv2.resize(text_image(), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    assert assess_quality(big).ok


def test_가장_선명한_프레임을_고른다():
    sharp = text_image()
    frames = [cv2.GaussianBlur(sharp, (0, 0), 2), sharp, cv2.GaussianBlur(sharp, (0, 0), 1)]
    i, q = best_frame(frames)
    assert i == 1 and q.ok


def test_모두_나쁘면_그중_나은_것을_고른다():
    sharp = text_image()
    frames = [cv2.GaussianBlur(sharp, (0, 0), 4), cv2.GaussianBlur(sharp, (0, 0), 1.5)]
    i, q = best_frame(frames)
    assert i == 1 and not q.ok


def test_프레임이_없으면_예외():
    with pytest.raises(ValueError):
        best_frame([])
