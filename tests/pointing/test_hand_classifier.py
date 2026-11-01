"""pointing/hand_classifier.py — 특징 · numpy 계산 · 가중치 읽기 · 못 읽으면 규칙으로.

틀리면 학습 때와 실행 때 손 모양을 다르게 읽어 분류기가 조용히 엉뚱한 답을 낸다.
sklearn 과 계산이 같은지는 sklearn 이 깔린 곳(맥 · 윈도우)에서만 본다. CI(ubuntu)에는 없다.
"""

import numpy as np
import pytest

from jarviseo import config
from jarviseo.pointing import hand_classifier
from jarviseo.pointing.hand_classifier import HandClassifier, features, load_default
from jarviseo.pointing.hands import HandPose


def _pose(world, handedness: str = "Right") -> HandPose:
    return HandPose(
        landmarks=tuple((float(i), float(i)) for i in range(21)),
        world=tuple(tuple(map(float, p)) for p in world),
        handedness=handedness,
        score=0.9,
    )


def _world(seed: int = 0) -> np.ndarray:
    world = np.random.default_rng(seed).normal(size=(21, 3)) * 0.03
    world[9] = world[0] + (0.0, 0.08, 0.0)  # 손목 → 중지 뿌리 8cm
    return world


# --- 특징 ------------------------------------------------------------------------


def test_features_put_wrist_at_origin_and_unit_palm():
    """손목이 원점, 손목 → 중지 뿌리 길이가 1."""
    f = features(_pose(_world())).reshape(21, 3)
    assert np.allclose(f[0], 0.0)
    assert np.linalg.norm(f[9]) == pytest.approx(1.0)


def test_features_ignore_hand_size_and_position():
    """크기 · 위치만 다른 같은 모양은 특징이 같다."""
    world = _world()
    moved = world * 1.7 + (0.1, -0.2, 0.05)
    assert np.allclose(features(_pose(world)), features(_pose(moved)))


def test_left_hand_is_mirrored_to_right():
    """왼손은 x 를 뒤집어 같은 모양의 오른손과 같은 특징이 된다."""
    world = _world()
    mirrored = world * (-1.0, 1.0, 1.0)
    assert np.allclose(features(_pose(mirrored, "Left")), features(_pose(world, "Right")))


# --- 계산 · 저장 ------------------------------------------------------------------


def _tiny(threshold: float = 0.5) -> HandClassifier:
    """은닉 2칸짜리 손 계산용 분류기. 특징 0번 칸만 본다."""
    w0 = np.zeros((63, 2))
    w0[0, 0] = 1.0
    return HandClassifier(
        mean=np.zeros(63),
        scale=np.ones(63),
        weights=(w0, np.array([[2.0], [0.0]])),
        biases=(np.zeros(2), np.array([-1.0])),
        threshold=threshold,
    )


def test_prob_matches_hand_computation():
    """relu → 시그모이드: 특징 0번 칸이 0(손목)이라 logit = -1 → 확률 0.269."""
    assert _tiny().prob(_pose(_world())) == pytest.approx(1 / (1 + np.e))


def test_is_pointing_uses_threshold():
    pose = _pose(_world())
    assert not _tiny(threshold=0.5).is_pointing(pose)
    assert _tiny(threshold=0.2).is_pointing(pose)


def test_save_and_load_round_trip(tmp_path):
    """npz 로 내보내고 읽어도 같은 확률 · 문턱 · 기록."""
    path = tmp_path / "hand.npz"
    original = HandClassifier(**{**_tiny().__dict__, "note": "R1 test"})
    original.save(path)
    loaded = HandClassifier.from_file(path)
    pose = _pose(_world(3))
    assert loaded.prob(pose) == pytest.approx(original.prob(pose))
    assert loaded.threshold == 0.5 and loaded.note == "R1 test"


def test_prob_matches_sklearn(tmp_path):
    """sklearn 으로 학습한 것을 numpy 로 계산해도 predict_proba 와 같다."""
    pytest.importorskip("sklearn")
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    poses = [_pose(_world(i)) for i in range(40)]
    x = np.array([features(p) for p in poses])
    y = np.array([i % 2 for i in range(40)])
    model = make_pipeline(StandardScaler(), MLPClassifier((8, 4), max_iter=300, random_state=0))
    model.fit(x, y)
    scaler, mlp = model[0], model[-1]
    ours = HandClassifier(
        scaler.mean_, scaler.scale_, tuple(mlp.coefs_), tuple(mlp.intercepts_), 0.5
    )
    expected = model.predict_proba(x)[:, 1]
    assert np.allclose([ours.prob(p) for p in poses], expected, atol=1e-6)


# --- 설정 ------------------------------------------------------------------------


def test_load_default_none_when_flag_off(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_HAND_CLASSIFIER", False)
    assert load_default() is None


def test_load_default_none_when_file_missing(monkeypatch, tmp_path, caplog):
    """플래그를 켜도 가중치가 없으면 None + 경고 — 비서는 15° 규칙으로 계속 돈다."""
    hand_classifier._load.cache_clear()
    monkeypatch.setattr(config, "ENABLE_HAND_CLASSIFIER", True)
    monkeypatch.setattr(config, "HAND_CLASSIFIER_PATH", tmp_path / "없음.npz")
    assert load_default() is None
    assert "15° 규칙" in caplog.text


def test_load_default_reads_file_when_flag_on(monkeypatch, tmp_path):
    hand_classifier._load.cache_clear()
    path = tmp_path / "hand.npz"
    _tiny().save(path)
    monkeypatch.setattr(config, "ENABLE_HAND_CLASSIFIER", True)
    monkeypatch.setattr(config, "HAND_CLASSIFIER_PATH", path)
    assert load_default().threshold == 0.5


def test_shipped_weights_load():
    """레포에 들어간 가중치(hand_r1.npz)가 읽히고 문턱이 0~1 이다(서버가 이 파일을 그대로 쓴다)."""
    loaded = HandClassifier.from_file(config.HAND_CLASSIFIER_PATH)
    assert 0 < loaded.threshold < 1
    assert loaded.weights[0].shape[0] == 63
