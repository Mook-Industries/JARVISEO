"""손 관절 → 가리키는 방향 (MediaPipe HandLandmarker).  담당: 최홍묵

책임 지표: 손 검출률 · 가리킴 판정률 · 방향 오차 (안경 시점)

손끝 YOLO 는 학습하지 않는다(10/14 결정, 방향 C). 손끝 박스 하나로는 손가락이 어디를
향하는지 모른다. MediaPipe 관절 21개 중 검지 뿌리(5) → 끝(8)으로 방향을 구한다.

물건을 들고 있거나 키보드에 얹은 손도 MediaPipe 는 손으로 잡는다. 그 손의 방향은 뜻이 없어서
"가리키는 손"인지 따로 가른다. 가리키는 손이 없으면 손 단서(v2)는 0점이 아니라 기권이다.
손이 화면에 없어도 대상은 골라야 하므로(10/14 요구사항) 나머지 단서로 고른다.

여기 함수는 좌표 계산이라 MediaPipe 없이 테스트한다. 신경망은 HandDetector 까지만 쓴다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from jarviseo import config

__all__ = [
    "FULL_GAP_DEG",
    "MIN_HAND_CONFIDENCE",
    "POINTING_MIN_GAP_DEG",
    "HandDetector",
    "HandPose",
    "angle_to",
    "bend_deg",
    "bend_gap_deg",
    "finger_bend",
    "is_pointing",
    "pick_pointing_hand",
    "pointing_ray",
    "pointing_score",
    "poses_from_result",
]

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]
# (시작점 = 검지 끝, 길이 1 방향)
Ray = tuple[Point2, Point2]

NUM_LANDMARKS = 21
INDEX_BASE, INDEX_TIP = 5, 8
# 중지 · 약지 · 소지 뿌리. 손가락마다 뿌리부터 4개 관절이 이어진다.
OTHER_FINGER_BASES = (9, 13, 17)

# 손 검출 기준. 기본값 0.5 는 렌즈에 가까워 손바닥이 잘린 손을 놓친다.
# valid 손 있음 32장 중 0.5 → 15장, 0.3 → 18장, 0.1 → 24장(손 없음 30장에서 잡힌 손 3 → 4).
# 더 잡힌 엉뚱한 손은 아래 가리킴 판정이 거른다.
MIN_HAND_CONFIDENCE = 0.1
# (나머지 손가락 굽힘 평균 - 검지 굽힘)이 이보다 크면 가리키는 손이다.
# valid 기준 0.1 에서 15° 가 손 없음 오탐 0/30 인 기준 중 가장 많이 통과(18/32)시켰다.
POINTING_MIN_GAP_DEG = 15.0
# 이만큼 차이 나면 점수 1. 확실히 가리킨 손이 54~136° 였다.
FULL_GAP_DEG = 90.0


@dataclass(frozen=True)
class HandPose:
    """MediaPipe 가 찾은 손 하나.

    좌표를 두 벌 갖는다. 방향은 화면 좌표로 구한다(후보 박스가 픽셀이다).
    굽힘은 3D 로 구한다. 안경은 손을 위에서 내려다봐서, 접은 손가락도 화면에서는 길게 보인다.

    가리킴 안에서만 쓰여서 types.py 가 아니라 여기에 둔다. 팀원 모듈이 읽게 되면 옮긴다.
    """

    landmarks: tuple[Point2, ...]  # 화면 픽셀 (x, y), 관절 21개
    world: tuple[Point3, ...]  # 손 중심 기준 3D 미터 (x, y, z), 관절 21개
    handedness: str  # "Left" / "Right"
    score: float  # 왼손/오른손 판단의 신뢰도 0.0~1.0

    def __post_init__(self) -> None:
        if len(self.landmarks) != NUM_LANDMARKS or len(self.world) != NUM_LANDMARKS:
            raise ValueError(
                f"관절은 {NUM_LANDMARKS}개여야 한다: 화면 {len(self.landmarks)}개, "
                f"3D {len(self.world)}개"
            )


def bend_deg(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """b 에서 꺾인 각도. 곧으면 0°, 완전히 접히면 180° 에 가깝다. 점이 겹치면 0°."""
    u = [p - q for p, q in zip(a, b, strict=True)]
    v = [p - q for p, q in zip(c, b, strict=True)]
    norm = math.hypot(*u) * math.hypot(*v)
    if norm == 0:
        return 0.0
    cos = sum(p * q for p, q in zip(u, v, strict=True)) / norm
    # 부동소수 오차로 1 을 살짝 넘으면 acos 가 터진다
    return 180.0 - math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def finger_bend(hand: HandPose, base: int) -> float:
    """손가락 하나의 두 마디 꺾임 합. 뿌리 관절은 손등 각도에 흔들려서 뺀다."""
    p = hand.world
    first = bend_deg(p[base], p[base + 1], p[base + 2])
    second = bend_deg(p[base + 1], p[base + 2], p[base + 3])
    return first + second


def bend_gap_deg(hand: HandPose) -> float:
    """나머지 세 손가락 굽힘 평균 - 검지 굽힘. 검지만 펴고 나머지를 접을수록 크다."""
    others = sum(finger_bend(hand, base) for base in OTHER_FINGER_BASES) / len(OTHER_FINGER_BASES)
    return others - finger_bend(hand, INDEX_BASE)


def pointing_score(hand: HandPose) -> float:
    """가리키는 손다운 정도 0.0~1.0. 판정(is_pointing)과 따로 둬서 v2 가중치에 곱할 수 있다."""
    return min(1.0, max(0.0, bend_gap_deg(hand) / FULL_GAP_DEG))


def is_pointing(hand: HandPose, min_gap_deg: float = POINTING_MIN_GAP_DEG) -> bool:
    return bend_gap_deg(hand) > min_gap_deg


def pointing_ray(hand: HandPose) -> Ray | None:
    """검지 끝에서 뻗는 선. 뿌리와 끝이 1픽셀 안에 겹치면 방향을 몰라 None.

    5 → 8 을 쓴다. valid 에서 정답 중심까지 각도 중앙값이 5→8 19°, 6→8 23°, 7→8 32° 였다.
    끝(8)에서 시작해야 손가락 자체를 가리킨 물건처럼 지나가지 않는다.
    """
    (bx, by), (tx, ty) = hand.landmarks[INDEX_BASE], hand.landmarks[INDEX_TIP]
    length = math.hypot(tx - bx, ty - by)
    if length < 1.0:
        return None
    return (tx, ty), ((tx - bx) / length, (ty - by) / length)


def angle_to(ray: Ray, point: Point2) -> float | None:
    """선 방향과 '손끝 → 점' 사이 각도 0~180°. 점이 손끝과 겹치면 None."""
    (ox, oy), (dx, dy) = ray
    vx, vy = point[0] - ox, point[1] - oy
    length = math.hypot(vx, vy)
    if length == 0:
        return None
    cos = (dx * vx + dy * vy) / length
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def pick_pointing_hand(
    hands: Sequence[HandPose], min_gap_deg: float = POINTING_MIN_GAP_DEG
) -> HandPose | None:
    """가리키는 손 하나. 여럿이면 점수가 가장 높은 손, 없으면 None(손 단서 기권).

    한 손으로 물건을 들고 다른 손으로 가리키는 장면이 있다.
    """
    pointing = [h for h in hands if is_pointing(h, min_gap_deg) and pointing_ray(h) is not None]
    return max(pointing, key=pointing_score, default=None)


def poses_from_result(result: Any, frame_size: tuple[int, int]) -> list[HandPose]:
    """MediaPipe 결과를 HandPose 로. 화면 관절은 0~1 → 픽셀, 3D 관절은 그대로.

    세 리스트(화면 관절 · 3D 관절 · 왼손/오른손)는 손 순서가 같다. 길이가 어긋나면
    조용히 짝을 잘못 맞추지 않게 바로 터뜨린다.
    """
    width, height = frame_size
    return [
        HandPose(
            landmarks=tuple((p.x * width, p.y * height) for p in points),
            world=tuple((p.x, p.y, p.z) for p in world),
            handedness=side[0].category_name,
            score=side[0].score,
        )
        for points, world, side in zip(
            result.hand_landmarks, result.hand_world_landmarks, result.handedness, strict=True
        )
    ]


class HandDetector:
    """사진 한 장에서 손을 찾는다. 앱에서 하나만 만들어 돌려 쓴다(모델을 한 번만 읽으려고).

    IMAGE 모드다. 그래프는 턴마다 질문이 시작된 시각의 사진 한 장만 본다.
    VIDEO 모드는 연속 프레임과 그 시각을 넘겨야 하는 추적용이다.
    """

    def __init__(
        self,
        model_path: Path | str = config.HAND_LANDMARKER_PATH,
        min_confidence: float = MIN_HAND_CONFIDENCE,
        num_hands: int = 2,
    ) -> None:
        model_path = Path(model_path)
        # 파일부터 본다. mediapipe 가 없는 CI 에서도 이 안내가 테스트된다.
        if not model_path.exists():
            raise FileNotFoundError(
                f"손 관절 모델 파일이 없다: {model_path} (받는 법: pointing/README.md)"
            )
        # CI 에는 mediapipe 가 없다. 모듈 위에서 import 하면 pointing 테스트가 전부 깨진다.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            # 경로 대신 바이트로 넘긴다. 윈도우에서 경로에 한글이 있으면 모델을 못 연다.
            base_options=BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            # 검출 · 존재 기준을 같이 낮춘다. 하나만 낮추면 다른 하나가 막는다.
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> list[HandPose]:
        """BGR 이미지에서 손을 찾는다. 없으면 빈 리스트.

        MediaPipe 는 RGB 를 받는다. BGR 그대로 받는 YOLO(ObjectDetector)와 반대다.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        height, width = image.shape[:2]
        return poses_from_result(self.landmarker.detect(mp_image), (width, height))
