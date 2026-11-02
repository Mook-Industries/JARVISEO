"""손 관절 → 가리키는 방향 (MediaPipe HandLandmarker).  담당: 최홍묵

책임 지표: 손 검출률 · 가리킴 판정률 · 방향 오차 (안경 시점)

손끝 YOLO 는 학습하지 않는다(10/14 결정, 방향 C). 손끝 박스 하나로는 손가락이 어디를
향하는지 모른다. MediaPipe 관절 21개 중 검지 뿌리(5) → 끝(8)으로 방향을 구한다.

물건을 들고 있거나 키보드에 얹은 손도 MediaPipe 는 손으로 잡는다. 그 손의 방향은 뜻이 없어서
"가리키는 손"인지 따로 가른다. 판정은 15° 굽힘 규칙이고, ENABLE_HAND_CLASSIFIER 를 켜면
손 영상으로 학습한 분류기(hand_classifier.py, #223)가 대신한다.
가리키는 손이 없으면 손 단서(v2)는 0점이 아니라 기권이다.
손이 화면에 없어도 대상은 골라야 하므로(10/14 요구사항) 나머지 단서로 고른다.

MediaPipe 는 손바닥을 먼저 찾고 그 둘레에서 관절을 읽는다. 손바닥은 사진 전체를
192px 로 줄여서 찾아서, 팔을 뻗은 작은 손 · 화면 끝에 걸린 손은 그 크기에서 사라진다.
ENABLE_HAND_TILES 를 켜면 가리키는 손을 못 고른 사진만 4조각으로 나눠 다시 찾는다(#256).
ENABLE_HAND_BOX_DETECTOR 를 켜면 그래도 못 고른 사진은 우리 안경 영상으로 학습한 손 박스
검출기가 찾은 박스에서 관절을 읽는다(hand_boxes.py, #256 ⑤-b).

여기 함수는 좌표 계산이라 MediaPipe 없이 테스트한다. 신경망은 HandDetector 까지만 쓴다.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from jarviseo import config
from jarviseo.pointing.hand_classifier import HandClassifier, PatchEncoder, load_default
from jarviseo.types import BBox

__all__ = [
    "FULL_GAP_DEG",
    "MIN_HAND_CONFIDENCE",
    "POINTING_MIN_GAP_DEG",
    "SAME_HAND_IOU",
    "TILE_RATIO",
    "HandDetector",
    "HAND_PATCH_SIZE",
    "HandPose",
    "attach_patch_features",
    "angle_to",
    "bend_deg",
    "bend_gap_deg",
    "finger_bend",
    "hand_patch",
    "hand_patch_box",
    "is_pointing",
    "landmark_box",
    "load_box_finder",
    "merge_hands",
    "pick_pointing_hand",
    "pointing_ray",
    "pointing_score",
    "poses_from_result",
    "search_tiles",
    "shift_hand",
    "tile_boxes",
]

log = logging.getLogger(__name__)

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
# 손 사진(R2 분류기 입력). 224 = ImageNet 사전학습 CNN 의 입력 크기, 여백은 손가락 끝이 잘리지 않게.
HAND_PATCH_SIZE = 224
HAND_PATCH_MARGIN = 0.25
# 조각 확대(#256). 조각 한 변 = 사진 가로 · 세로의 60%, 네 모서리에 하나씩이라 가운데 20% 가 겹친다.
# 겹쳐야 조각 경계에 걸린 손도 어느 한 조각에는 통째로 든다. 3 × 2 조각은 손이 더 커지는데도
# 4조각보다 덜 찾았다(사전 측정, docs/experiments.md "조각 확대").
TILE_RATIO = 0.6
# 조각에서 찾은 손의 관절 상자가 이미 찾은 손과 이만큼 겹치면 같은 손으로 보고 버린다.
# 조각이 겹쳐서 한 손이 두 조각에서 잡힌다.
SAME_HAND_IOU = 0.3


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
    # 손 사진의 CNN 특징(R2 분류기, #223). 분류기가 필요로 할 때만 HandDetector 가 채운다.
    # 배열을 == 로 비교하면 참 · 거짓이 안 정해져 에러가 나서 비교 · 출력에서 뺀다.
    patch_feature: np.ndarray | None = field(default=None, compare=False, repr=False)

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


def hand_patch_box(
    points: Sequence[Point2], image_size: tuple[int, int], margin: float = HAND_PATCH_MARGIN
) -> tuple[int, int, int, int]:
    """손 관절을 감싸는 정사각형 (x1, y1, x2, y2). 한 변 = 관절 상자의 긴 변 × (1 + 2 × 여백).

    사진 밖으로 나가면 잘라 내지 않고 안쪽으로 민다. 잘라 내면 정사각이 아니게 돼서
    224×224 로 줄일 때 손이 찌그러진다. 한 변이 관절 상자보다 크니 밀어도 손은 안에 남는다.
    학습용 손 사진(tools/hand_clips.py)과 실행 때 손 사진이 이 함수 하나에서 나온다.
    """
    width, height = image_size
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    long_side = max(max(xs) - min(xs), max(ys) - min(ys))
    side = max(1, min(round(long_side * (1 + 2 * margin)), width, height))
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    x1 = min(max(round(cx - side / 2), 0), width - side)
    y1 = min(max(round(cy - side / 2), 0), height - side)
    return x1, y1, x1 + side, y1 + side


def hand_patch(image: np.ndarray, hand: HandPose) -> np.ndarray:
    """손 사진 224×224 (BGR)."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = hand_patch_box(hand.landmarks, (width, height))
    size = (HAND_PATCH_SIZE, HAND_PATCH_SIZE)
    return cv2.resize(image[y1:y2, x1:x2], size, interpolation=cv2.INTER_AREA)


def attach_patch_features(image: np.ndarray, hands: list[HandPose], encoder: Any) -> list[HandPose]:
    """손마다 손 사진의 CNN 특징을 붙인 새 HandPose. 손이 없으면 그대로."""
    if not hands:
        return hands
    vectors = encoder.encode([hand_patch(image, h) for h in hands])
    return [replace(h, patch_feature=v) for h, v in zip(hands, vectors, strict=True)]


def tile_boxes(
    width: int, height: int, ratio: float = TILE_RATIO
) -> list[tuple[int, int, int, int]]:
    """사진을 덮는 조각 4개 (x1, y1, x2, y2) — 왼위 · 오른위 · 왼아래 · 오른아래."""
    tile_w, tile_h = int(width * ratio), int(height * ratio)
    return [
        (x, y, x + tile_w, y + tile_h) for y in (0, height - tile_h) for x in (0, width - tile_w)
    ]


def landmark_box(hand: HandPose) -> BBox:
    """화면 관절 21개를 감싸는 상자."""
    xs = [p[0] for p in hand.landmarks]
    ys = [p[1] for p in hand.landmarks]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def shift_hand(hand: HandPose, dx: float, dy: float) -> HandPose:
    """조각 좌표로 찾은 손을 원본 사진 좌표로 옮긴다. 3D 관절은 손 중심 기준이라 그대로다."""
    return replace(hand, landmarks=tuple((x + dx, y + dy) for x, y in hand.landmarks))


def merge_hands(
    kept: Sequence[HandPose], extra: Sequence[HandPose], same_iou: float = SAME_HAND_IOU
) -> list[HandPose]:
    """kept 뒤에 extra 중 새 손만 붙인다.

    관절 상자가 앞의 손과 same_iou 이상 겹치면 같은 손이라 버린다.
    같은 손이면 먼저 찾은 쪽(사진 전체)을 남긴다.
    """
    merged = list(kept)
    for hand in extra:
        box = landmark_box(hand)
        if all(box.iou(landmark_box(h)) < same_iou for h in merged):
            merged.append(hand)
    return merged


def search_tiles(
    image: np.ndarray,
    find: Callable[[np.ndarray], list[HandPose]],
    ratio: float = TILE_RATIO,
) -> list[HandPose]:
    """조각마다 find 로 찾은 손을 원본 좌표로 옮겨 모은다.

    두 조각에 걸친 같은 손은 둘 다 들어 있다 — merge_hands 가 거른다.
    """
    height, width = image.shape[:2]
    found: list[HandPose] = []
    for x1, y1, x2, y2 in tile_boxes(width, height, ratio):
        found += [shift_hand(h, x1, y1) for h in find(image[y1:y2, x1:x2])]
    return found


_FROM_CONFIG: Any = object()


def pick_pointing_hand(
    hands: Sequence[HandPose],
    min_gap_deg: float = POINTING_MIN_GAP_DEG,
    classifier: HandClassifier | None = _FROM_CONFIG,
) -> HandPose | None:
    """가리키는 손 하나. 여럿이면 점수가 가장 높은 손, 없으면 None(손 단서 기권).

    한 손으로 물건을 들고 다른 손으로 가리키는 장면이 있다.
    classifier 를 안 주면 설정을 따른다(플래그가 꺼졌거나 가중치가 없으면 15° 규칙).
    None 을 주면 15° 규칙, 분류기를 주면 그 확률로 판정 · 고르기를 한다.
    부르는 쪽(그래프 · 웹 · 측정 스크립트)은 이 함수만 부르므로 판정을 여기서 한 번에 바꾼다.
    """
    if classifier is _FROM_CONFIG:
        classifier = load_default()
    if classifier is not None and classifier.needs_patch:
        # 사진 특징을 안 붙인 손(가짜 손 · encoder 없이 만든 검출기)이면 규칙으로 판정한다
        if any(h.patch_feature is None for h in hands):
            classifier = None
    if classifier is None:
        pointing = [h for h in hands if is_pointing(h, min_gap_deg) and pointing_ray(h) is not None]
        return max(pointing, key=pointing_score, default=None)
    probs = [(classifier.prob(h), h) for h in hands if pointing_ray(h) is not None]
    passed = [(p, h) for p, h in probs if p >= classifier.threshold]
    return max(passed, key=lambda ph: ph[0], default=(0.0, None))[1]


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


def load_box_finder(strict: bool) -> Any:
    """손 박스 검출기(hand_boxes.HandBoxFinder). 가중치 · 모델 파일이 없을 때
    strict 면 그대로 터뜨리고, 아니면 한 번 알리고 None(이 단계만 끈다).

    설정(플래그)으로 켠 비서는 가중치 하나 때문에 전체가 서면 안 된다. 측정 스크립트처럼
    직접 켜라고 한 쪽은 조용히 꺼지면 손 박스 없이 잰 숫자가 나와서 터뜨린다.
    """
    from jarviseo.pointing.hand_boxes import HandBoxFinder  # hand_boxes 가 이 모듈을 불러 순환

    try:
        return HandBoxFinder()
    except (FileNotFoundError, ImportError) as e:
        if strict:
            raise
        log.warning("손 박스 검출기를 못 읽어 그 단계를 끕니다: %s", e)
        return None


class HandDetector:
    """사진 한 장에서 손을 찾는다. 앱에서 하나만 만들어 돌려 쓴다(모델을 한 번만 읽으려고).

    IMAGE 모드다. 그래프는 턴마다 질문이 시작된 시각의 사진 한 장만 본다.
    VIDEO 모드는 연속 프레임과 그 시각을 넘겨야 하는 추적용이다.

    tiles 를 켜면 사진 전체에서 가리키는 손을 못 고른 사진만 4조각에서 다시 찾는다(#256).
    늘 조각까지 돌리면 사진마다 MediaPipe 가 다섯 번 돌아, 이미 찾은 사진까지 느려진다.
    boxes 를 켜면 그래도 못 고른 사진만 손 박스 검출기가 찾은 박스에서 관절을 읽는다(#256 ⑤-b).
    이미 가리키는 손을 고른 사진은 앞 단계 그대로라, 뒤 단계가 바꿀 수 있는 건 못 고른 사진뿐이다.
    """

    def __init__(
        self,
        model_path: Path | str = config.HAND_LANDMARKER_PATH,
        min_confidence: float = MIN_HAND_CONFIDENCE,
        num_hands: int = 2,
        encoder: Any = _FROM_CONFIG,
        classifier: HandClassifier | None = _FROM_CONFIG,
        tiles: bool = _FROM_CONFIG,
        boxes: bool = _FROM_CONFIG,
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
        # 조각을 돌릴지 정하는 손 판정. 부르는 쪽의 pick_pointing_hand 와 같은 설정을 따른다.
        # 둘이 다르면 고를 손이 있는데 조각을 돌리거나, 못 고르는데 안 돌린다.
        if classifier is _FROM_CONFIG:
            classifier = load_default()
        self.classifier = classifier
        # 손 사진 특징기. 분류기가 사진 특징을 쓸 때(R2)만 만든다 — 꺼진 서버에서
        # CNN 을 돌리면 손 하나에 수 ms 가 공짜로 더해진다.
        if encoder is _FROM_CONFIG:
            encoder = PatchEncoder() if classifier is not None and classifier.needs_patch else None
        self.encoder = encoder
        self.tiles = config.ENABLE_HAND_TILES if tiles is _FROM_CONFIG else tiles
        use_boxes = config.ENABLE_HAND_BOX_DETECTOR if boxes is _FROM_CONFIG else boxes
        self.box_finder = load_box_finder(strict=boxes is not _FROM_CONFIG) if use_boxes else None

    def detect(self, image: np.ndarray) -> list[HandPose]:
        """BGR 이미지에서 손을 찾는다. 없으면 빈 리스트.

        사진 전체 → (가리키는 손을 못 고르면) 4조각 → (그래도 못 고르면) 손 박스 순서다.
        뒤 단계에서 찾은 손은 앞에서 찾은 손 뒤에 붙는다(같은 손은 하나만).
        """
        hands = self._with_patch(image, self._landmarks(image))
        if self.tiles and not self._has_pointing(hands):
            # 손 사진 특징은 원본에서 자른다. 조각에서 자르면 조각 경계에서 손이 한쪽으로 밀린다.
            tile_hands = search_tiles(image, self._landmarks)
            hands = merge_hands(hands, self._with_patch(image, tile_hands))
        if self.box_finder is not None and not self._has_pointing(hands):
            hands = merge_hands(hands, self._with_patch(image, self._box_hands(image, hands)))
        return hands

    def _has_pointing(self, hands: list[HandPose]) -> bool:
        """부르는 쪽 pick_pointing_hand 와 같은 판정으로 가리키는 손을 골랐나."""
        return pick_pointing_hand(hands, classifier=self.classifier) is not None

    def _box_hands(self, image: np.ndarray, found: list[HandPose]) -> list[HandPose]:
        """이미 찾은 손과 겹치지 않는 손 박스에서만 관절을 읽는다(같은 손을 두 번 안 읽게)."""
        known = [landmark_box(hand) for hand in found]
        fresh = [
            box
            for box in self.box_finder.boxes(image)
            if all(box.iou(other) < SAME_HAND_IOU for other in known)
        ]
        read = (self.box_finder.read(image, box) for box in fresh)
        return [hand for hand in read if hand is not None]

    def _landmarks(self, image: np.ndarray) -> list[HandPose]:
        """MediaPipe 한 번. 관절은 넘겨받은 이미지의 픽셀 좌표다.

        MediaPipe 는 RGB 를 받는다. BGR 그대로 받는 YOLO(ObjectDetector)와 반대다.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        height, width = image.shape[:2]
        return poses_from_result(self.landmarker.detect(mp_image), (width, height))

    def _with_patch(self, image: np.ndarray, hands: list[HandPose]) -> list[HandPose]:
        if self.encoder is None:
            return hands
        return attach_patch_features(image, hands, self.encoder)
