"""성분표 글자 읽기.  담당: B

책임 지표: OCR 인식률

파이프라인: ROI 크롭 -> 업스케일 -> 전처리 -> OCR

업스케일이 왜 필요한가
----------------------
성분표 글자는 약 2mm 다. 30cm 거리에서 1920x1080 으로 찍어도 글자 높이가
10~15픽셀 정도밖에 안 나온다. OCR 이 안정적으로 읽으려면 보통 30픽셀 이상이
필요하다. 그래서 잘라낸 뒤 2~4배로 키워서 넣는다.

(0.3MP 카메라를 기각한 이유가 이것이다. 640x480 이면 글자가 4픽셀이라
아무리 키워도 없는 정보가 생기지 않는다.)

학습은 하지 않는다. 기성 OCR 엔진을 쓰되, 어느 엔진이 한글 성분표에
가장 잘 맞는지는 W1 스파이크에서 비교해서 정한다.
"""

from __future__ import annotations

import numpy as np

from jarviseo.types import BBox, IngredientPanel, OCRLine

__all__ = ["crop_and_upscale", "read_panel", "parse_ingredients"]


def crop_and_upscale(image: np.ndarray, bbox: BBox, scale: float = 3.0) -> np.ndarray:
    """성분표 영역을 잘라내고 키운다.

    보간 방식은 cv2.INTER_CUBIC 을 쓴다. 기본값인 INTER_LINEAR 는
    글자 경계가 뭉개져서 OCR 에 불리하다.
    """
    raise NotImplementedError


def read_panel(panel_image: np.ndarray) -> list[OCRLine]:
    """확대한 성분표 이미지에서 글자를 줄 단위로 읽는다."""
    raise NotImplementedError


def parse_ingredients(lines: list[OCRLine]) -> list[str]:
    """읽어낸 글자에서 성분명만 뽑아낸다.

    한글 성분표는 보통 "원재료명 : 밀가루(미국산), 설탕, 팜유, ..." 형태다.
    쉼표로 끊되 괄호 안 원산지는 버린다.

    OCR 이 흔히 틀리는 것들을 여기서 보정한다.
    예: "팜유" -> "팜유", "칼슘" 의 'ㄹ' 이 깨지는 경우 등.
    실제 오인식 패턴은 촬영한 300장을 돌려보고 나서 채운다.
    """
    raise NotImplementedError
