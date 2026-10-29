"""v3 시선 분포 맞추기 — 학습용 줄 고르기, 0~1 좌표, 평균 · 공분산, 모자란 점 · 한 줄 점,
상수 줄 모양, 실행(학습용 줄 없음 · 사진 없음 · 정상).

사진은 작은 빈 그림을 tmp 에 만들어 쓴다. YOLO 는 쓰지 않는다.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
# fit_gaze 가 같은 폴더의 label_targets 를 import 한다. 스크립트로 실행하면 그 폴더가 경로에 든다.
sys.path.append(str(_TOOLS))
_spec = importlib.util.spec_from_file_location("fit_gaze", _TOOLS / "fit_gaze.py")
fit_gaze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fit_gaze)

COLUMNS = [
    "file",
    "session",
    "split",
    "utterance",
    "gt_x1",
    "gt_y1",
    "gt_x2",
    "gt_y2",
    "gt_label",
    "has_hand",
    "ambiguous",
    "prev_label",
    "excluded",
]


def _row(file="s05/a.jpg", session=5, split="train", box=(80, 40, 120, 60), excluded=0):
    x1, y1, x2, y2 = box if box is not None else ("", "", "", "")
    return {
        "file": file,
        "session": str(session),
        "split": split,
        "utterance": "이거 뭐야?",
        "gt_x1": str(x1),
        "gt_y1": str(y1),
        "gt_x2": str(x2),
        "gt_y2": str(y2),
        "gt_label": "cup" if box is not None else "",
        "has_hand": "0",
        "ambiguous": "0",
        "prev_label": "",
        "excluded": str(excluded),
    }


def _spread_points(n: int = 12) -> np.ndarray:
    """한 줄에 놓이지 않는 점 n 개(0~1 안)."""
    rng = np.random.default_rng(0)
    return rng.uniform(0.3, 0.7, size=(n, 2))


# --- 학습용 줄 · 좌표 ------------------------------------------------------------


def test_train_rows_keep_train_session_with_box_only():
    """valid · 다른 세션 · 뺀 사진 · 박스 없는 줄은 맞추는 데 쓰지 않는다."""
    rows = [
        _row(file="s05/keep.jpg"),
        _row(file="s01/valid.jpg", session=1, split="valid"),
        _row(file="s06/other.jpg", session=6),
        _row(file="s05/excluded.jpg", box=None, excluded=1),
        _row(file="s05/nobox.jpg", box=None),
    ]

    kept = fit_gaze.train_rows(rows, [5])

    assert [r["file"] for r in kept] == ["s05/keep.jpg"]


def test_centers_are_divided_by_image_size():
    """같은 자리면 사진 크기가 달라도 같은 0~1 좌표가 된다."""
    rows = [
        _row(file="s05/big.jpg", box=(800, 400, 1120, 680)),  # 1920×1080 의 (0.5, 0.5)
        _row(file="s05/small.jpg", box=(560, 260, 720, 460)),  # 1280×720 의 (0.5, 0.5)
    ]
    sizes = {"s05/big.jpg": (1920, 1080), "s05/small.jpg": (1280, 720)}

    points = fit_gaze.centers(rows, sizes)

    assert points.shape == (2, 2)
    assert np.allclose(points, [[0.5, 0.5], [0.5, 0.5]])


# --- 평균 · 공분산 ----------------------------------------------------------------


def test_fit_returns_mean_and_maximum_likelihood_covariance():
    """공분산은 n 으로 나눈다(최대우도). numpy 의 bias=True 와 같아야 한다."""
    points = _spread_points()

    mean, cov = fit_gaze.fit(points)

    assert np.allclose(mean, points.mean(axis=0))
    assert np.allclose(cov, np.cov(points, rowvar=False, bias=True))
    assert cov[0][1] == cov[1][0]


def test_fit_refuses_too_few_points():
    """점이 MIN_POINTS 보다 적으면 운에 좌우되므로 맞추지 않는다."""
    with pytest.raises(ValueError, match="개뿐이다"):
        fit_gaze.fit(_spread_points(fit_gaze.MIN_POINTS - 1))


def test_fit_refuses_points_on_one_line():
    """한 줄에 놓인 점은 공분산을 뒤집을 수 없어 시선 점수를 못 낸다."""
    xs = np.linspace(0.3, 0.7, 12)
    points = np.stack([xs, 0.2 + 0.5 * xs], axis=1)

    with pytest.raises(ValueError, match="한 줄"):
        fit_gaze.fit(points)


def test_constant_lines_are_ready_to_paste():
    """cues.py 에 붙일 두 줄은 파이썬으로 읽히고 소수 4자리다."""
    lines = fit_gaze.constant_lines((0.51234, 0.6), ((0.012345, 0.001), (0.001, 0.02)))

    assert lines == [
        "GAZE_MEAN = (0.5123, 0.6000)",
        "GAZE_COV = ((0.0123, 0.0010), (0.0010, 0.0200))",
    ]
    namespace: dict = {}
    exec("\n".join(lines), namespace)
    assert namespace["GAZE_MEAN"] == (0.5123, 0.6)


def test_summary_shows_offset_from_center():
    """화면 가운데(가운데 단서의 기준)보다 얼마나 어긋났는지 보인다."""
    lines = fit_gaze.summary(12, (0.52, 0.61), ((0.01, 0.0), (0.0, 0.04)))

    assert lines[0] == "점 12개"
    assert "x +0.020 · y +0.110" in lines[1]
    assert "표준편차 x 0.100 · y 0.200 · 상관 +0.00" == lines[2]


# --- 실행 ------------------------------------------------------------------------


def _write_labels(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_image(path: Path, size=(200, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    ok, jpeg = cv2.imencode(".jpg", np.zeros((height, width, 3), np.uint8))
    assert ok
    jpeg.tofile(path)


def _run(monkeypatch, labels: Path, raw: Path) -> None:
    argv = ["fit_gaze.py", "--labels", str(labels), "--raw", str(raw)]
    monkeypatch.setattr(sys, "argv", argv)
    fit_gaze.main()


def test_main_stops_without_train_rows(tmp_path, monkeypatch):
    """학습용 줄이 없으면 표시부터 하라고 알리고 멈춘다(valid 로 맞추지 않는다)."""
    labels = _write_labels(tmp_path / "labels.csv", [_row(session=1, split="valid")])

    with pytest.raises(SystemExit, match="--split train"):
        _run(monkeypatch, labels, tmp_path / "raw")


def test_main_stops_when_image_is_missing(tmp_path, monkeypatch):
    """사진이 없으면 그 파일 이름을 대고 멈춘다(크기를 모르면 0~1 좌표를 못 낸다)."""
    labels = _write_labels(tmp_path / "labels.csv", [_row(file="s05/none.jpg")])

    with pytest.raises(SystemExit, match="none.jpg"):
        _run(monkeypatch, labels, tmp_path / "raw")


def test_main_prints_summary_and_constants(tmp_path, monkeypatch, capsys):
    """정상: 요약과 붙일 상수 두 줄을 찍는다."""
    raw = tmp_path / "raw"
    rows = []
    for i, (x, y) in enumerate(_spread_points()):
        file = f"s05/p{i:02d}.jpg"
        _write_image(raw / file)
        cx, cy = x * 200, y * 100
        rows.append(_row(file=file, box=(cx - 5, cy - 5, cx + 5, cy + 5)))
    labels = _write_labels(tmp_path / "labels.csv", rows)

    _run(monkeypatch, labels, raw)

    out = capsys.readouterr().out
    assert "점 12개" in out
    assert "GAZE_MEAN = (" in out
    assert "GAZE_COV = ((" in out
