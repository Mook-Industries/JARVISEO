"""단서별 점수 계산.  담당: 최홍묵

후보 물체 하나하나에 대해 "이게 사용자가 가리킨 것일 가능성"을 단서별로
0.0~1.0 점수로 매긴다. 여기서 나온 점수들을 resolver.py 가 합친다.

ablation(단서를 하나씩 껐다 켜며 기여도를 재는 실험)의 축이 이 파일이다.
단서를 새로 추가할 때는 types.py 의 CueKind 에 항목을 먼저 넣고,
여기에 score_* 함수를 하나 추가한 뒤, resolver 의 가중치에 등록한다.

주의: 이 파일의 함수들은 딥러닝이 아니다. 좌표와 각도를 쓰는 기하 계산과
규칙이다. 신경망은 detector.py 가 손끝·물체를 찾는 데까지만 쓴다.
"""

from __future__ import annotations

from jarviseo.types import CueScore, Detection

__all__ = [
    "score_hand",
    "score_gaze",
    "score_language",
    "score_salience",
]


def score_hand(
    fingertip: Detection | None,
    candidates: list[Detection],
    frame_size: tuple[int, int],
) -> list[CueScore]:
    """손끝이 뻗은 방향에 있는 물체일수록 높은 점수.

    구현 방향: 손끝 좌표에서 손가락이 향하는 방향으로 가상의 선을 긋고,
    각 후보 물체 중심이 그 선에서 얼마나 벗어나 있는지(각도 차이)로 점수를 낸다.

    fingertip 이 None 이면(손이 안 보이면) 이 단서는 기권한다.
    전부 0.0 을 반환하되, 빈 리스트를 반환하지는 않는다.
    빈 리스트와 "전부 0점"은 의미가 다르고, ablation 에서 구분해야 한다.
    """
    raise NotImplementedError


def score_gaze(
    gaze_vector: tuple[float, float] | None,
    candidates: list[Detection],
    frame_size: tuple[int, int],
) -> list[CueScore]:
    """시선이 향하는 쪽에 있는 물체일수록 높은 점수.

    미결 사항: 시선 추정을 기성 라이브러리(MediaPipe 등)로 할지 직접 학습할지
    아직 안 정했다. 기성 라이브러리를 쓰면 이 단서에는 우리가 학습시킨 모델이
    들어가지 않는다는 점을 발표 때 명확히 구분해야 한다.

    안경 카메라는 착용자 시점이라 화면 중앙이 대략 시선 방향이다.
    그래서 이 단서는 score_salience 와 상관관계가 높을 수 있다.
    ablation 에서 둘을 같이 켰을 때 기여도가 겹치는지 확인할 것.
    """
    raise NotImplementedError


def score_language(
    text: str,
    candidates: list[Detection],
    frame_size: tuple[int, int],
) -> list[CueScore]:
    """발화 속 단서와 맞는 물체일수록 높은 점수.

    "빨간 거" -> 색, "왼쪽 거" -> 위치, "저 가방" -> 물체 종류.
    종류가 명시되면("저 가방") 그것만으로 후보가 하나로 좁혀지기도 한다.

    색 판단은 후보 박스 안의 대표 색을 뽑아서 비교한다.
    위치 판단은 frame_size 기준 상대 좌표로 한다.
    """
    raise NotImplementedError


def score_salience(
    candidates: list[Detection],
    frame_size: tuple[int, int],
) -> list[CueScore]:
    """화면 중앙에 가깝고 크게 찍힌 물체일수록 높은 점수.

    다른 단서가 하나도 없을 때의 기본값 역할을 한다.
    baseline(단서를 아무것도 안 쓴 상태)을 이 단서 하나만 켠 상태로 잡으면
    ablation 표의 첫 줄이 된다.
    """
    raise NotImplementedError
