"""단서별 점수 계산.  담당: 최홍묵

후보 물체 하나하나에 "이게 사용자가 가리킨 것일 가능성"을 단서별로 0.0~1.0 점수로 매긴다.
여기서 나온 점수를 resolver.py 가 가중치로 합친다. ablation 의 축이 이 파일이다.

지금은 v1 center 만 있다. 단서를 추가할 때는 types.CueKind 에 있는 이름으로 score_* 함수를
만들고, resolver 의 IMPLEMENTED_CUES 와 단서 → 함수 표에 등록한다.

이 파일의 함수는 딥러닝이 아니다. 좌표로 하는 계산이고, 신경망은 detector.py 까지만 쓴다.
"""

from __future__ import annotations

import math

from jarviseo.types import CueKind, CueScore, Detection

__all__ = ["CENTER_WEIGHT", "score_center"]

# 가운데 단서 안에서 위치를 얼마나 믿을지. 나머지(0.3)는 크기.
CENTER_WEIGHT = 0.7


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
