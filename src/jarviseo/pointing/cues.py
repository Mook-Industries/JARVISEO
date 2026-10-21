"""단서별 점수 계산.  담당: 최홍묵

후보 물체 하나하나에 "이게 사용자가 가리킨 것일 가능성"을 단서별로 0.0~1.0 점수로 매긴다.
여기서 나온 점수를 resolver.py 가 가중치로 합친다. ablation 의 축이 이 파일이다.

v1 center(화면 가운데 · 크기)와 v2 point(검지 방향)가 있다. 단서를 추가할 때는 types.CueKind 에
있는 이름으로 score_* 함수를 만들고, resolver 의 IMPLEMENTED_CUES 와 단서 → 함수 표에 등록한다.

이 파일의 함수는 딥러닝이 아니다. 좌표로 하는 계산이고, 신경망은 detector.py · hands.py 까지만 쓴다.
"""

from __future__ import annotations

import math

from jarviseo.pointing.hands import Ray, angle_to
from jarviseo.types import CueKind, CueScore, Detection

__all__ = ["CENTER_WEIGHT", "POINT_SIGMA_DEG", "score_center", "score_point"]

# 가운데 단서 안에서 위치를 얼마나 믿을지. 나머지(0.3)는 크기.
CENTER_WEIGHT = 0.7
# 손 방향 점수가 얼마나 빨리 떨어지는지. 15° 면 0.61, 30° 면 0.14.
# valid 에서 σ 10 · 15 · 20 이 같은 결과(13/62)라 가운데 값을 골랐다.
POINT_SIGMA_DEG = 15.0


def score_center(candidates: list[Detection], frame_size: tuple[int, int]) -> list[CueScore]:
    """화면 가운데에 가깝고 크게 찍힌 물체일수록 높은 점수. 후보와 같은 순서로 돌려준다.

    안경 카메라는 머리 방향을 따라가서, 보고 있는 물건이 화면 가운데에 온다.
    크기는 비슷한 거리의 두 물건 중 가까운 쪽을 가르는 보조다.
    다른 단서가 없을 때의 기본값이고, ablation 표의 첫 줄(baseline)이다.
    """
    width, height = frame_size
    half_w, half_h = width / 2, height / 2
    scores = []
    for d in candidates:
        cx, cy = d.bbox.center
        # 가로·세로를 각각 -1~1 로 맞춘 거리. 구석이 √2 라서 나눠 0~1 로 만든다.
        distance = math.hypot((cx - half_w) / half_w, (cy - half_h) / half_h) / math.sqrt(2)
        closeness = 1.0 - distance
        # 넓이 비율은 대개 0.01~0.1 이라 그대로 쓰면 0 근처에 몰린다. 제곱근(변의 비율)으로 편다.
        size = math.sqrt(d.bbox.area / (width * height))
        score = CENTER_WEIGHT * closeness + (1.0 - CENTER_WEIGHT) * size
        scores.append(
            CueScore(
                kind=CueKind.CENTER,
                score=min(1.0, max(0.0, score)),
                detail={"closeness": closeness, "size": size},
            )
        )
    return scores


def score_point(candidates: list[Detection], ray: Ray) -> list[CueScore]:
    """검지 끝에서 뻗은 선이 후보 중심을 향할수록 높은 점수. 후보와 같은 순서로 돌려준다.

    선과 '검지 끝 → 후보 중심' 사이 각도를 종 모양(가우시안)으로 바꾼다. 0° 면 1, 손 뒤쪽은 0 쪽.
    코사인은 30° 에서도 0.87 이라 후보끼리 차이가 안 난다. 선이 박스를 지나는지로 재면 책상의
    큰 박스(노트북 · 모니터)가 겹쳐 비긴다(valid 10/62, 중심 각도는 13/62).
    후보 중심이 검지 끝과 같은 자리면 손가락이 그 물건 위에 있는 것이라 1점이다.
    """
    scores = []
    for d in candidates:
        angle = angle_to(ray, d.bbox.center)
        score = 1.0 if angle is None else math.exp(-0.5 * (angle / POINT_SIGMA_DEG) ** 2)
        scores.append(CueScore(kind=CueKind.POINT, score=score, detail={"angle": angle}))
    return scores
