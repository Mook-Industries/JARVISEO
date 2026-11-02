"""가리키는 손 분류기 (R1, 손 관절 MLP).  담당: 최홍묵

15° 굽힘 규칙(hands.is_pointing)을 대신하는 학습 모델이다(#223). 손 영상 9개(장소 3 × 3명)의
손 관절로 학습했고(tools/train_hand_classifier.py), 사람 단위 3겹으로 문턱을 골랐다.

실행 쪽은 numpy 만 쓴다. 서버(리눅스)에는 scikit-learn 이 깔리지 않고(requirements.lock),
pickle 은 sklearn 버전이 바뀌면 못 여는 경우가 있어서 숫자 배열만 담은 .npz 로 주고받는다.

학습할 때와 실행할 때 특징이 어긋나지 않게, 특징 계산(features)은 여기 한 곳에만 둔다.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from jarviseo import config

if TYPE_CHECKING:  # hands.py 가 이 모듈을 불러서, 실행 중에 되불러오면 순환이 된다
    from jarviseo.pointing.hands import HandPose

__all__ = ["HandClassifier", "features", "load_default"]

log = logging.getLogger(__name__)

WRIST, MIDDLE_BASE = 0, 9


def features(hand: HandPose) -> np.ndarray:
    """3D 관절 21개 → 63칸. 손목을 원점으로, 손목 → 중지 뿌리 길이를 1로, 왼손은 거울로 뒤집는다.

    손 크기 · 위치가 아니라 손가락 모양만 보게 하려는 것이다. 영상의 세 사람은 주로 오른손으로
    찍어서, 뒤집지 않으면 왼손으로 가리킨 손이 처음 보는 모양이 된다.
    """
    points = np.asarray(hand.world, dtype=float)
    if hand.handedness == "Left":
        points = points * np.array([-1.0, 1.0, 1.0])
    points = points - points[WRIST]
    scale = float(np.linalg.norm(points[MIDDLE_BASE]))
    if scale > 0:
        points = points / scale
    return points.reshape(-1)


@dataclass(frozen=True)
class HandClassifier:
    """표준화 → 은닉층(ReLU) → 출력 한 칸(시그모이드).

    sklearn MLPClassifier 의 이진 분류와 같은 계산이다.
    """

    mean: np.ndarray
    scale: np.ndarray
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    threshold: float
    note: str = ""

    @classmethod
    def from_file(cls, path: Path | str) -> HandClassifier:
        with np.load(Path(path)) as data:
            layers = int(data["layers"])
            return cls(
                mean=data["mean"],
                scale=data["scale"],
                weights=tuple(data[f"w{i}"] for i in range(layers)),
                biases=tuple(data[f"b{i}"] for i in range(layers)),
                threshold=float(data["threshold"]),
                note=str(data["note"]) if "note" in data else "",
            )

    def save(self, path: Path | str) -> None:
        arrays = {f"w{i}": w for i, w in enumerate(self.weights)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez(
            Path(path),
            mean=self.mean,
            scale=self.scale,
            threshold=self.threshold,
            layers=len(self.weights),
            note=self.note,
            **arrays,
        )

    def prob(self, hand: HandPose) -> float:
        """가리키는 손일 확률 0~1."""
        x = (features(hand) - self.mean) / self.scale
        for w, b in zip(self.weights[:-1], self.biases[:-1], strict=True):
            x = np.maximum(x @ w + b, 0.0)
        logit = float((x @ self.weights[-1] + self.biases[-1])[0])
        # exp 가 넘치지 않게 부호에 따라 나눠 쓴다
        if logit >= 0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        z = np.exp(logit)
        return float(z / (1.0 + z))

    def is_pointing(self, hand: HandPose) -> bool:
        return self.prob(hand) >= self.threshold


def load_default() -> HandClassifier | None:
    """설정을 따른 분류기. 플래그가 꺼져 있거나 파일을 못 읽으면 None(15° 규칙으로 돌아간다)."""
    if not config.ENABLE_HAND_CLASSIFIER:
        return None
    return _load(config.HAND_CLASSIFIER_PATH)


@functools.cache
def _load(path: Path) -> HandClassifier | None:
    # 가중치 하나 때문에 비서 전체가 서면 안 된다. 못 읽으면 한 번만 알리고 규칙으로 간다.
    try:
        return HandClassifier.from_file(path)
    except (OSError, KeyError, ValueError) as e:
        log.warning("손 분류기 가중치를 못 읽어 15° 규칙으로 판정합니다: %s (%s)", path, e)
        return None
