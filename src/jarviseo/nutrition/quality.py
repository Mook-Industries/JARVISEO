"""OCR 전에 사진 품질을 본다.  담당: 권용현

흐리거나 반사가 심한 사진은 OCR 이 30초를 써도 제대로 못 읽는다. 그런 사진으로
판정하면 '못 읽은 것'을 '못 찾은 것(미검출)'처럼 말할 위험도 있다.
그래서 OCR 전에 걸러서 바로 다시 보여 달라고 한다. 경고를 놓치는 쪽이 아니라
확인 불가·재촬영 쪽으로 가므로 안전하다 (멘토 피드백 2026-10-14).

재는 것
-------
- 선명도 비율: 사진을 한 번 더 살짝 흐리게 했을 때 경계(라플라시안)가 얼마나 줄어드는가.
  선명한 사진은 많이 줄고(비율 큼) 이미 흐린 사진은 별로 안 준다(비율 작음).
  라플라시안 분산 하나만 쓰면 글자가 적은 성분표가 흐린 사진으로 오인되어 비율을 쓴다.
- 경계 세기: 라플라시안 분산. 아주 심하게 흐리면 비율이 다시 커지는 경우가 있어 함께 본다.
- 반사: 하얗게 날아간(밝고 색이 없는) 픽셀 비율. 소프트웨어 보정으로 되살릴 수 없다.

여러 프레임이 있으면 ``best_frame`` 으로 가장 선명하고 반사가 적은 것을 고른다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from jarviseo import config

__all__ = ["ImageQuality", "assess_quality", "best_frame"]

# 큰 사진은 이 크기로 줄여서 잰다. 해상도에 따라 경계 세기 값이 달라지지 않게 하고 빠르게 하려고.
# 키우지는 않는다(키우면 없는 경계가 생기지 않고 값만 흐려진다).
_MAX_SIDE = 1000


@dataclass
class ImageQuality:
    sharpness: float  # 선명도 비율
    edge: float  # 경계 세기 (라플라시안 분산)
    glare: float  # 반사 픽셀 비율 0~1
    reason: str = ""  # "" / "blurry" / "glare"

    @property
    def ok(self) -> bool:
        return not self.reason

    @property
    def score(self) -> float:
        """프레임 고르기용 점수. 선명할수록, 반사가 적을수록 크다."""
        return self.sharpness * (1.0 - self.glare)


def _gray(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) > _MAX_SIDE:
        s = _MAX_SIDE / max(h, w)
        image = cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return gray.astype(np.float64)


def assess_quality(
    image: np.ndarray,
    min_sharpness: float = config.QUALITY_MIN_SHARPNESS,
    min_edge: float = config.QUALITY_MIN_EDGE,
    max_glare: float = config.QUALITY_MAX_GLARE,
) -> ImageQuality:
    """사진 한 장의 품질. ``reason`` 이 비어 있으면 OCR 해도 된다."""
    gray = _gray(image)
    edge = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edge_blur = float(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.0), cv2.CV_64F).var())
    sharpness = edge / max(edge_blur, 1e-6)

    glare = 0.0
    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        glare = float(((hsv[..., 2] >= 245) & (hsv[..., 1] <= 30)).mean())

    reason = ""
    if sharpness < min_sharpness or edge < min_edge:
        reason = "blurry"
    elif glare > max_glare:
        reason = "glare"
    return ImageQuality(sharpness=sharpness, edge=edge, glare=glare, reason=reason)


def best_frame(frames: list[np.ndarray]) -> tuple[int, ImageQuality]:
    """여러 프레임 중 OCR 에 넣을 하나를 고른다. (인덱스, 그 프레임의 품질)

    품질 검사를 통과한 프레임이 있으면 그중 점수가 가장 높은 것, 없으면 전체에서 가장 높은 것.
    """
    if not frames:
        raise ValueError("프레임이 없습니다")
    scored = [(assess_quality(f), i) for i, f in enumerate(frames)]
    good = [(q, i) for q, i in scored if q.ok] or scored
    q, i = max(good, key=lambda x: x[0].score)
    return i, q
