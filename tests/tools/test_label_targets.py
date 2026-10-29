"""가리킴 정답 표시 도구 — 표시 파일 읽고 쓰기, 남은 사진 고르기, 미리보기 좌표 되돌리기.

화면과 키 입력은 사람이 하는 부분이라 빼고, 떼어 낸 순수 함수만 본다. YOLO 도 쓰지 않는다.
"""

import importlib.util
from pathlib import Path

import numpy as np

from jarviseo.types import BBox, Detection

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "label_targets.py"
_spec = importlib.util.spec_from_file_location("label_targets", _TOOL)
label_targets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(label_targets)


def _det(label: str, x1: float) -> Detection:
    return Detection(label=label, confidence=0.9, bbox=BBox(x1, 100, x1 + 200, 300))


# --- 표시 파일 읽고 쓰기 -----------------------------------------------------------


def test_append_then_read_keeps_rows_and_header_once(tmp_path):
    """두 줄을 쓰고 다시 읽으면 두 줄이고, 헤더는 처음 한 번만 들어간다. 한글도 그대로다."""
    path = tmp_path / "pointing" / "labels.csv"  # 없는 폴더도 만든다
    row = label_targets.make_row("s01/a.jpg", 1, "valid", "저거 뭐야?", BBox(1, 2, 3, 4), "cup")

    label_targets.append_label(path, row)
    label_targets.append_label(path, {**row, "file": "s01/b.jpg"})

    rows = label_targets.read_csv(path)
    assert [r["file"] for r in rows] == ["s01/a.jpg", "s01/b.jpg"]
    assert rows[0]["utterance"] == "저거 뭐야?"
    assert path.read_text(encoding="utf-8").count("file,session") == 1


def test_read_missing_file_is_empty(tmp_path):
    """표시 파일이 아직 없으면 빈 목록이다."""
    assert label_targets.read_csv(tmp_path / "none.csv") == []


def test_make_row_writes_flags_as_zero_one():
    """참/거짓은 0/1 로 쓴다. 좌표는 소수 한 자리로 줄인다."""
    row = label_targets.make_row(
        "s01/a.jpg",
        1,
        "valid",
        "왼쪽 거",
        BBox(10.04, 20.06, 30.0, 40.0),
        "cup",
        has_hand=True,
        ambiguous=False,
        prev_label="bottle",
    )

    assert (row["has_hand"], row["ambiguous"], row["excluded"]) == (1, 0, 0)
    assert (row["gt_x1"], row["gt_y1"]) == (10.0, 20.1)
    assert row["prev_label"] == "bottle"


def test_excluded_row_has_no_box():
    """뺀 사진은 좌표가 비고 excluded 가 1 이다."""
    row = label_targets.make_row("s01/a.jpg", 1, "valid", "흐림", None, "", excluded=True)

    assert row["excluded"] == 1
    assert row["gt_x1"] == row["gt_x2"] == ""


# --- 남은 사진 · 후보 번호 -----------------------------------------------------------


def test_pending_skips_already_labeled():
    """이미 표시한 사진은 다시 묻지 않는다. 찍은 순서는 그대로다."""
    manifest = [
        {"file": "s01_desk_0001.jpg"},
        {"file": "s01_desk_0002.jpg"},
        {"file": "s01_desk_0003.jpg"},
    ]
    labeled = {"s01/s01_desk_0002.jpg"}

    todo = label_targets.pending(manifest, labeled, session=1)

    assert [r["file"] for r in todo] == ["s01_desk_0001.jpg", "s01_desk_0003.jpg"]


def test_pending_is_per_session():
    """다른 세션에서 같은 파일 이름을 표시했어도 이 세션에서는 남은 사진이다."""
    manifest = [{"file": "x.jpg"}]
    assert label_targets.pending(manifest, {"s02/x.jpg"}, session=1) == manifest


def test_candidates_are_numbered_left_to_right():
    """화면 왼쪽 후보부터 1, 2, 3 번이 된다(YOLO 는 신뢰도 순으로 준다)."""
    right, left, middle = _det("cup", 1500), _det("bottle", 100), _det("book", 800)

    assert label_targets.left_to_right([right, left, middle]) == [left, middle, right]


# --- 미리보기 좌표 되돌리기 -------------------------------------------------------------


def test_roi_is_scaled_back_to_original():
    """960 폭 미리보기에서 그린 박스는 원본(1920) 좌표로 두 배가 된다."""
    box = label_targets.roi_to_box((100, 50, 200, 100), scale=2.0)
    assert box == BBox(200, 100, 600, 300)


def test_empty_roi_is_cancel():
    """그리다 만 박스(폭이나 높이 0)는 취소로 본다."""
    assert label_targets.roi_to_box((0, 0, 0, 0), scale=2.0) is None


def test_draw_returns_preview_width_and_scale():
    """미리보기는 960 폭이고, 원본으로 되돌리는 배율을 같이 준다."""
    image = np.zeros((1080, 1920, 3), np.uint8)

    preview, scale = label_targets.draw(image, [_det("cup", 100)], "hand=0")

    assert preview.shape[:2] == (540, 960)
    assert scale == 2.0


def test_read_image_handles_missing_and_hangul_path(tmp_path):
    """한글 경로의 사진도 읽고, 빈 파일은 None 이다."""
    import cv2

    path = tmp_path / "한글 폴더" / "사진.jpg"
    path.parent.mkdir()
    ok, buf = cv2.imencode(".jpg", np.full((10, 20, 3), 128, np.uint8))
    buf.tofile(path)
    empty = tmp_path / "빈.jpg"
    empty.write_bytes(b"")

    assert label_targets.read_image(path).shape == (10, 20, 3)
    assert label_targets.read_image(empty) is None


# --- 후보 번호로 고른 정답의 이름 ---------------------------------------------------


def test_chosen_label_uses_yolo_name_when_input_is_blank():
    """Enter 만 치면 YOLO 이름 그대로. 공백만 친 것도 빈칸이다."""
    assert label_targets.chosen_label("bottle", "") == "bottle"
    assert label_targets.chosen_label("bottle", "   ") == "bottle"


def test_chosen_label_prefers_typed_name_and_strips_spaces():
    """박스는 맞고 이름이 틀린 후보(안경 케이스를 remote 로 부름)는 입력한 이름으로 남긴다."""
    assert label_targets.chosen_label("remote", "  glasses case ") == "glasses case"


def test_split_accepts_train_for_gaze_session():
    """학습용 세션(s05, v3 시선 분포)을 train 으로 표시한다. valid 로 두면 채점에 섞인다."""
    assert label_targets.SPLITS == ("valid", "test", "train")
    row = label_targets.make_row("s05/a.jpg", 5, "train", "이거 뭐야?", BBox(1, 2, 3, 4), "cup")
    assert row["split"] == "train"
