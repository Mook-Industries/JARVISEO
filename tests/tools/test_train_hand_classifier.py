"""tools/train_hand_classifier.py — sklearn 없이 검사할 수 있는 부분.

틀리면 회차 비교가 조용히 망가지는 곳을 잡는다 — 표 줄을 손 모양으로 되돌리기(실행과 같은 특징) ·
문턱 고르기(R0 재현율 이상 중 가장 높은 것) · 가리킴 줄 늘리기 · 가중치 덮어쓰기 거부.

tools/ 는 패키지가 아니라서 파일 경로로 직접 불러온다.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from jarviseo.pointing.hand_classifier import features

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "train_hand_classifier.py"
_spec = importlib.util.spec_from_file_location("train_hand_classifier", _TOOL_PATH)
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)


def _row(handedness: str = "Right") -> dict[str, str]:
    row = {"handedness": handedness, "hand_score": "0.9"}
    for i in range(21):
        row.update({f"w{i}_x": str(0.01 * i), f"w{i}_y": str(0.002 * i * i), f"w{i}_z": "0.0"})
        row.update({f"p{i}_x": str(10.0 * i), f"p{i}_y": str(5.0 * i)})
    return row


def test_pose_from_row_gives_same_features_as_runtime():
    """표 줄 → HandPose 가 관절 순서를 지켜, 실행 때와 같은 특징이 나온다."""
    pose = train.pose_from_row(_row())
    assert pose.world[3] == pytest.approx((0.03, 0.018, 0.0))
    assert pose.landmarks[2] == (20.0, 10.0)
    assert features(pose).shape == (63,)


def test_rates_recall_and_false_alarm():
    y = np.array([1, 1, 0, 0, 0, 0])
    pred = np.array([True, False, True, False, False, False])
    assert train.rates(pred, y) == (0.5, 0.25)


def test_pick_threshold_highest_that_reaches_target():
    """재현율 0.75 이상이 되는 가장 높은 문턱 — 0.3 에서 처음 넘는다."""
    y = np.array([1, 1, 1, 1, 0, 0])
    prob = np.array([0.9, 0.6, 0.35, 0.05, 0.4, 0.01])
    assert train.pick_threshold(prob, y, target_recall=0.75) == 0.3


def test_pick_threshold_lowest_when_never_reached():
    y = np.array([1, 1])
    prob = np.array([0.0, 0.0])
    assert train.pick_threshold(prob, y, target_recall=0.9) == train.THRESHOLDS[-1]


def test_balanced_index_matches_negatives():
    """가리킴 줄을 아님 줄 수만큼 늘린다(아님 줄은 그대로 한 번씩)."""
    y = np.array([1, 0, 0, 0, 0, 1, 0])
    index = train.balanced_index(y, np.random.default_rng(0))
    assert (y[index] == 1).sum() == (y[index] == 0).sum() == 5
    assert sorted(index[y[index] == 0]) == [1, 2, 3, 4, 6]


def test_main_refuses_to_overwrite_weights(monkeypatch, tmp_path):
    """회차 가중치가 조용히 바뀌지 않게, 이미 있으면 --force 없이는 멈춘다."""
    out = tmp_path / "hand.npz"
    out.write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", ["train_hand_classifier.py", "--out", str(out)])
    with pytest.raises(SystemExit, match="--force"):
        train.main()
