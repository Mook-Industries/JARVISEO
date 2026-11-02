"""손 박스 검출기 + 관절 모델 직접 실행 — MediaPipe 가 못 찾은 손의 관절을 읽는다.  담당: 최홍묵

MediaPipe HandLandmarker 는 ① 손바닥 찾기(사진 전체를 192px 로 줄여서) → ② 관절 읽기(224 크롭)다.
안경 시점에서는 ① 이 팔을 뻗은 작은 손 · 흐린 손을 놓친다(#256). 여기서는 ① 대신 우리 안경
영상으로 학습한 손 박스 검출기(YOLOv8n, 1종 hand — tools/hand_box_labels.py ·
notebooks/hand_box_finetune.ipynb)가 손을 찾고, 그 박스 둘레에서 ② 관절 모델
(hand_landmarker.task 안의 hand_landmarks_detector.tflite)을 OpenCV 로 직접 돌려 관절 21개를 얻는다.
결과는 MediaPipe 와 같은 HandPose 라서 가리키는 손 판정(R1) · 방향 계산은 그대로다.

HandDetector 가 사진 전체 · 4조각으로도 가리키는 손을 못 고른 사진에서만 부른다
(config.ENABLE_HAND_BOX_DETECTOR).

② 는 손이 선 모양(손목 아래, 손가락 위)으로 잘린 사진을 받는다. MediaPipe 는 ① 에서 그 각도를
구해 돌려 자른다. 여기서는 박스를 그대로 한 번 읽어 손목 → 중지 뿌리 방향을 구하고,
그만큼 돌려 다시 읽는다. 3D 관절도 돌린 만큼 되돌린다 — R1 은 3D 관절 방향을 바로 세우지 않고
그대로 읽어서, 안 되돌리면 학습 때와 다른 방향의 손을 받는다.

여기 순수 함수(자르기 좌표 · 회전 · 왼오)는 모델 없이 테스트한다.
"""

from __future__ import annotations

import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from jarviseo import config
from jarviseo.pointing.hands import HandPose
from jarviseo.types import BBox

__all__ = [
    "BOX_CONF",
    "CROP_SCALE",
    "MAX_BOXES",
    "MIN_PRESENCE",
    "HandBoxFinder",
    "crop_matrix",
    "crop_square",
    "handedness_name",
    "rotate_world",
    "to_image",
    "upright_angle",
]

# 손 박스 확신 기준. 물체 검출기(ObjectDetector)와 같은 ultralytics 기본값.
BOX_CONF = 0.25
# MediaPipe 설정(손 최대 2개)과 같게 해서 지연이 늘어나는 폭을 묶는다.
MAX_BOXES = 2
# 정사각 한 변 = 박스 긴 변 × 이 값. 학습 정답 박스에 이미 사방 15% 여백이 있다.
# 학습 영상 손 317개(MediaPipe 가 찾은 손)를 이 길로 다시 읽어 정했다(평가 사진으로 안 고름):
# 1.0 · 1.2 · 1.4 · 1.6 에서 읽은 손 257 · 293 · 307 · 303, R1 판정이 MediaPipe 와 같은 손
# 241 · 277 · 292 · 284 → 가장 많은 1.4.
CROP_SCALE = 1.4
LANDMARK_SIZE = 224
# 손 있음 점수 기준 = MediaPipe 기본값. 이 단계에 오는 손은 MediaPipe 가 기준 0.1 로도
# 못 찾은 손이라, 헛손을 막으려고 기본값으로 올린다.
MIN_PRESENCE = 0.5
WRIST, MIDDLE_BASE = 0, 9
LANDMARK_MODEL = "hand_landmarks_detector.tflite"
# ② 의 출력 이름: 화면 관절(224 크롭 픽셀, 21 × xyz) · 손 있음 · 오른손 점수 · 3D 관절(미터)
OUTPUTS = ("Identity", "Identity_1", "Identity_2", "Identity_3")


@dataclass(frozen=True)
class _Reading:
    landmarks: np.ndarray  # (21, 3) 크롭 픽셀
    presence: float
    right: float
    world: np.ndarray  # (21, 3)


def crop_square(box: BBox, scale: float = CROP_SCALE) -> tuple[float, float, float]:
    """박스 중심 (cx, cy) 와 정사각 한 변(박스 긴 변 × scale)."""
    cx, cy = box.center
    return cx, cy, max(box.x2 - box.x1, box.y2 - box.y1) * scale


def crop_matrix(
    cx: float, cy: float, side: float, angle_deg: float, size: int = LANDMARK_SIZE
) -> np.ndarray:
    """원본 → 크롭(size × size) 아핀 행렬(2 × 3). (cx, cy) 가 크롭 가운데로 가고, 화면에서
    시계 반대 방향으로 angle_deg 만큼 돌고, 한 변 side 가 size 로 늘어난다.
    """
    matrix = cv2.getRotationMatrix2D((cx, cy), angle_deg, size / side)
    matrix[0, 2] += size / 2 - cx
    matrix[1, 2] += size / 2 - cy
    return matrix


def to_image(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """크롭 좌표(n × 2) → 원본 좌표. crop_matrix 를 거꾸로 한다."""
    inverse = cv2.invertAffineTransform(matrix)
    return np.asarray(points, dtype=float) @ inverse[:, :2].T + inverse[:, 2]


def upright_angle(points: np.ndarray) -> float:
    """손목(0) → 중지 뿌리(9)가 화면 위를 향하게 하려면 돌릴 각도(crop_matrix 의 angle_deg).

    -180 ~ 180°. 이미 손가락이 위면 0, 오른쪽을 향하면 90(시계 반대로 90° 돌린다).
    """
    dx = points[MIDDLE_BASE][0] - points[WRIST][0]
    dy = points[MIDDLE_BASE][1] - points[WRIST][1]
    # 화면 좌표는 y 가 아래로 커서, 수학 각도로 바꾸려면 dy 부호를 뒤집는다
    angle = 90.0 - math.degrees(math.atan2(-dy, dx))
    return (angle + 180.0) % 360.0 - 180.0 if angle != 180.0 else 180.0


def rotate_world(world: np.ndarray, angle_deg: float) -> np.ndarray:
    """크롭에서 읽은 3D 관절(n × 3)을 원본 방향으로 되돌린다(화면 평면 x · y 만, z 는 그대로).

    크롭은 원본을 화면에서 시계 반대로 angle_deg 돌린 것이라, 반대로 돌린다.
    to_image 가 화면 관절에 하는 회전과 같은 방향이다.
    """
    theta = math.radians(angle_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    world = np.asarray(world, dtype=float)
    x, y = world[:, 0], world[:, 1]
    return np.stack([x * cos - y * sin, x * sin + y * cos, world[:, 2]], axis=1)


def handedness_name(right: float) -> tuple[str, float]:
    """② 의 오른손 점수 → (이름, 그 이름의 확신). 이름은 MediaPipe 결과와 같은 Left · Right."""
    return ("Right", right) if right > 0.5 else ("Left", 1.0 - right)


def _load_landmark_net(task_path: Path):
    """hand_landmarker.task(zip) 안의 ② 모델을 OpenCV 로 읽는다.

    메모리 버퍼로 읽으면 engine 인자가 안 먹어 새 엔진이 출력을 1개(3D 관절)만 낸다.
    파일 경로 + ENGINE_CLASSIC 으로 읽어야 4개가 다 나와서, 임시 파일로 꺼내 읽고 지운다.
    윈도우에서 임시 폴더 경로에 한글이 있으면 못 열 수 있다(서버 · 맥은 문제없다).
    """
    with zipfile.ZipFile(task_path) as z:
        data = z.read(LANDMARK_MODEL)
    # OpenCV 4 에는 이 엔진 하나뿐이라 engine 인자가 없다
    engine = getattr(cv2.dnn, "ENGINE_CLASSIC", None)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / LANDMARK_MODEL
        path.write_bytes(data)
        if engine is None:
            return cv2.dnn.readNetFromTFLite(str(path))
        return cv2.dnn.readNetFromTFLite(str(path), engine=engine)


class HandBoxFinder:
    """손 박스 검출기 + ② 관절 모델. 앱에서 하나만 만들어 돌려 쓴다(모델을 한 번만 읽으려고)."""

    def __init__(
        self, weights: Path | str | None = None, landmark_model: Path | str | None = None
    ) -> None:
        # 기본값 자리에 config 값을 바로 쓰면 정의할 때 한 번만 읽혀서, 바꾼 설정이 안 따라온다.
        weights = Path(weights if weights is not None else config.HAND_BOX_WEIGHTS)
        landmark_model = Path(
            landmark_model if landmark_model is not None else config.HAND_LANDMARKER_PATH
        )
        if not weights.exists():
            raise FileNotFoundError(
                f"손 박스 가중치가 없다: {weights} "
                "(Colab 학습 결과 runs/h1/weights/best.pt 를 받는다 — tools/README.md)"
            )
        if not landmark_model.exists():
            raise FileNotFoundError(
                f"손 관절 모델 파일이 없다: {landmark_model} (받는 법: pointing/README.md)"
            )
        # CI 에는 ultralytics 가 없다. 모듈 위에서 import 하면 pointing 테스트가 전부 깨진다.
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.weights_path = weights
        self.net = _load_landmark_net(landmark_model)

    def boxes(self, image: np.ndarray) -> list[BBox]:
        """BGR 사진에서 손 박스. 확신 높은 순으로 MAX_BOXES 개까지."""
        result = self.model.predict(image, conf=BOX_CONF, verbose=False)[0]
        found = zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist(), strict=True)
        ranked = sorted(found, key=lambda pair: pair[0], reverse=True)
        return [BBox(*xyxy) for _, xyxy in ranked[:MAX_BOXES]]

    def read(self, image: np.ndarray, box: BBox) -> HandPose | None:
        """박스 하나에서 관절 21개. 손 있음 점수가 MIN_PRESENCE 미만이면 None."""
        cx, cy, side = crop_square(box, CROP_SCALE)
        first = crop_matrix(cx, cy, side, 0.0)
        angle = upright_angle(to_image(self._forward(image, first).landmarks[:, :2], first))
        matrix = crop_matrix(cx, cy, side, angle)
        reading = self._forward(image, matrix)
        if reading.presence < MIN_PRESENCE:
            return None
        landmarks = to_image(reading.landmarks[:, :2], matrix)
        world = rotate_world(reading.world, angle)
        name, score = handedness_name(reading.right)
        return HandPose(
            landmarks=tuple((float(x), float(y)) for x, y in landmarks),
            world=tuple((float(x), float(y), float(z)) for x, y, z in world),
            handedness=name,
            score=score,
        )

    def find(self, image: np.ndarray) -> list[HandPose]:
        """손 박스 검출기 + ② 만으로 찾은 손(MediaPipe 손바닥 찾기 없이)."""
        found = (self.read(image, box) for box in self.boxes(image))
        return [hand for hand in found if hand is not None]

    def _forward(self, image: np.ndarray, matrix: np.ndarray) -> _Reading:
        crop = cv2.warpAffine(image, matrix, (LANDMARK_SIZE, LANDMARK_SIZE))
        # ② 는 RGB 0~1 을 받는다. OpenCV 사진은 BGR 이라 채널을 바꾼다.
        blob = cv2.dnn.blobFromImage(crop, 1 / 255.0, (LANDMARK_SIZE, LANDMARK_SIZE), swapRB=True)
        self.net.setInput(blob)
        out = dict(zip(OUTPUTS, self.net.forward(list(OUTPUTS)), strict=True))
        return _Reading(
            landmarks=np.asarray(out["Identity"], dtype=float).reshape(21, 3),
            presence=float(np.asarray(out["Identity_1"]).reshape(-1)[0]),
            right=float(np.asarray(out["Identity_2"]).reshape(-1)[0]),
            world=np.asarray(out["Identity_3"], dtype=float).reshape(21, 3),
        )
