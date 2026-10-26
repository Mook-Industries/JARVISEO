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


def test_sharp_text_photo_passes():
    """선명한 글자 사진은 통과한다."""
    q = assess_quality(text_image())
    assert q.ok, q


@pytest.mark.parametrize("sigma", [1.5, 3.0, 6.0])
def test_blurry_photo_is_rejected(sigma):
    """흐린 사진은 걸러진다."""
    q = assess_quality(cv2.GaussianBlur(text_image(), (0, 0), sigma))
    assert q.reason == "blurry", q


def test_strong_glare_is_rejected():
    """반사가 심하면 걸러진다."""
    img = text_image()
    h, w = img.shape[:2]
    img[:, w // 10 : w // 10 + w // 5] = 255  # 글자 줄 가운데가 하얗게 날아가 글자가 지워짐
    q = assess_quality(img)
    assert q.reason == "glare", q


def test_white_margin_is_not_glare():
    """흰 종이 여백은 반사가 아니다. 글자 영역 바깥의 흰 부분은 세지 않는다."""
    img = text_image()
    h, w = img.shape[:2]
    page = np.full((h + 300, w + 400, 3), 255, np.uint8)  # 흰 종이 위 성분표
    page[150 : 150 + h, 200 : 200 + w] = img
    q = assess_quality(page)
    assert q.reason != "glare", q


def test_flat_image_counts_as_blurry():
    """단색 이미지는 흐린 것으로 본다."""
    assert assess_quality(np.full((100, 100, 3), 128, np.uint8)).reason == "blurry"


def test_large_photo_uses_same_threshold():
    """큰 사진도 같은 기준으로 잰다."""
    # 해상도가 커져도(카메라 원본) 선명한 사진이 흐린 것으로 나오면 안 된다
    big = cv2.resize(text_image(), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    assert assess_quality(big).ok


def test_picks_sharpest_frame():
    """가장 선명한 프레임을 고른다."""
    sharp = text_image()
    frames = [cv2.GaussianBlur(sharp, (0, 0), 2), sharp, cv2.GaussianBlur(sharp, (0, 0), 1)]
    i, q = best_frame(frames)
    assert i == 1 and q.ok


def test_picks_best_of_bad_frames():
    """모두 나쁘면 그중 나은 것을 고른다."""
    sharp = text_image()
    frames = [cv2.GaussianBlur(sharp, (0, 0), 4), cv2.GaussianBlur(sharp, (0, 0), 1.5)]
    i, q = best_frame(frames)
    assert i == 1 and not q.ok


def test_no_frames_raises():
    """프레임이 없으면 예외다."""
    with pytest.raises(ValueError):
        best_frame([])
