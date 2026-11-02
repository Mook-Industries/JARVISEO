"""tools/hand_box_labels.py — 손 박스 자동 정답 · 학습 묶음(#256 ⑤-b).

MediaPipe 없이 가짜 손 찾기 함수와 작은 사진으로 단계 · 이어 읽기 · 라벨 · 묶음을 확인한다.
"""

import importlib.util
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from jarviseo.pointing.hands import HandPose, landmark_box
from jarviseo.types import BBox

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.append(str(_TOOLS))  # 도구가 같은 폴더의 hand_clips 를 불러온다
_spec = importlib.util.spec_from_file_location("hand_box_labels", _TOOLS / "hand_box_labels.py")
labels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(labels)


def _hand(x: float, y: float, size: float = 20.0) -> HandPose:
    """(x, y) 에서 오른쪽 아래로 size 만큼 펼친 가짜 손. 관절 상자 = (x, y, x+size, y+size)."""
    points = tuple((x + size * i / 20, y + size * i / 20) for i in range(21))
    world = tuple((0.0, 0.0, 0.0) for _ in range(21))
    return HandPose(landmarks=points, world=world, handedness="Right", score=0.9)


def _shot(t_sec: float, boxes: list[BBox], cls: str = "point") -> "labels.Shot":
    stage = "full" if boxes else "none"
    return labels.Shot(
        image_id=f"v_{t_sec}",
        source="hand",
        video="v",
        stage=stage,
        boxes=boxes,
        cls=cls,
        t_sec=t_sec,
    )


# --- 박스 · 조각 ------------------------------------------------------------------


def test_padded_box_adds_margin_on_each_side():
    """너비 100 · 높이 50 상자에 15% → 좌우 15px · 위아래 7.5px 씩 넓어진다."""
    box = labels.padded_box(BBox(100, 100, 200, 150), (1000, 1000))
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((85, 92.5, 215, 157.5))


def test_padded_box_clips_to_image():
    """사진 끝에 걸린 손은 여백이 사진 밖으로 나가지 않는다."""
    box = labels.padded_box(BBox(0, 5, 55, 40), (60, 45))
    assert box.x1 == 0 and box.y1 == pytest.approx(0)
    assert box.x2 == 60 and box.y2 == 45


def test_square_tiles_landscape_takes_three_height_squares():
    assert labels.square_tiles(1280, 720) == [
        (0, 0, 720, 720),
        (280, 0, 1000, 720),
        (560, 0, 1280, 720),
    ]


def test_square_tiles_portrait_slides_down():
    assert labels.square_tiles(720, 1280) == [
        (0, 0, 720, 720),
        (0, 280, 720, 1000),
        (0, 560, 720, 1280),
    ]


def test_staged_hands_stops_at_full_image():
    image = np.zeros((720, 1280, 3), np.uint8)
    stage, hands = labels.staged_hands(image, lambda crop: [_hand(5, 5)])
    assert stage == "full" and len(hands) == 1


def test_staged_hands_falls_back_to_four_tiles():
    """사진 전체에선 못 찾고 왼쪽 위 4조각(60%)에서 찾으면 g2x2 — 3조각까지 안 간다."""
    image = np.zeros((720, 1280, 3), np.uint8)
    image[0, 0] = 9

    def find(crop):
        return [_hand(10, 10)] if crop.shape[:2] == (432, 768) and crop[0, 0, 0] == 9 else []

    stage, hands = labels.staged_hands(image, find)
    assert stage == "g2x2" and len(hands) == 1


def test_staged_hands_falls_back_to_squares_and_shifts_to_image():
    """오른쪽 정사각에서만 찾은 손은 그 정사각 시작점(560)만큼 옮겨 원본 좌표가 된다."""
    image = np.zeros((720, 1280, 3), np.uint8)
    image[0, 560] = 7

    def find(crop):
        return [_hand(10, 10)] if crop.shape[:2] == (720, 720) and crop[0, 0, 0] == 7 else []

    stage, hands = labels.staged_hands(image, find)
    assert stage == "sq3"
    assert landmark_box(hands[0]).x1 == pytest.approx(570)


def test_staged_hands_none_when_nothing_found():
    image = np.zeros((720, 1280, 3), np.uint8)
    assert labels.staged_hands(image, lambda crop: []) == ("none", [])


# --- 이어 읽기 ------------------------------------------------------------------


def test_fill_holes_fills_single_gap_with_average():
    shots = [_shot(0.0, [BBox(0, 0, 10, 10)]), _shot(0.2, []), _shot(0.4, [BBox(10, 0, 20, 10)])]
    assert labels.fill_holes(shots) == 1
    box = shots[1].boxes[0]
    assert shots[1].stage == "fill"
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((5, 0, 15, 10))


def test_fill_holes_leaves_two_frame_gap():
    """두 장 연달아 빈 칸은 손이 움직였을 수 있어 채우지 않는다."""
    box = BBox(0, 0, 10, 10)
    shots = [_shot(0.0, [box]), _shot(0.2, []), _shot(0.4, []), _shot(0.6, [box])]
    assert labels.fill_holes(shots) == 0
    assert not shots[1].boxes and not shots[2].boxes


def test_fill_holes_needs_same_segment_close_frames_and_hand_count():
    box = BBox(0, 0, 10, 10)
    other_segment = [_shot(0.0, [box]), _shot(0.2, []), _shot(0.4, [box], cls="hold")]
    time_gap = [_shot(0.0, [box]), _shot(0.6, []), _shot(0.8, [box])]
    count = [_shot(0.0, [box]), _shot(0.2, []), _shot(0.4, [box, BBox(50, 0, 60, 10)])]
    assert labels.fill_holes(other_segment) == 0
    assert labels.fill_holes(time_gap) == 0
    assert labels.fill_holes(count) == 0


def test_fill_holes_pairs_two_hands_left_to_right():
    left, right = BBox(0, 0, 10, 10), BBox(100, 0, 110, 10)
    shots = [_shot(0.0, [right, left]), _shot(0.2, []), _shot(0.4, [left, right])]
    labels.fill_holes(shots)
    xs = sorted(b.x1 for b in shots[1].boxes)
    assert xs == pytest.approx([0, 100])


# --- 쓰임 · 라벨 · 나누기 ----------------------------------------------------------


def test_s05_role_trusts_human_no_hand_mark():
    assert labels.s05_role(has_hand=False, found=1) == labels.BACKGROUND
    assert labels.s05_role(has_hand=True, found=1) == "positive"
    assert labels.s05_role(has_hand=True, found=0) == "skip"


def test_yolo_line_is_normalized_center_and_size():
    line = labels.yolo_line(BBox(100, 50, 300, 250), (640, 360))
    assert line == "0 0.312500 0.416667 0.312500 0.555556"


def test_split_of_holds_out_place_and_video():
    assert labels.split_of("hand", "loby", "h_loby_p1") == "val"
    assert labels.split_of("hand", "hall", "h_hall_p1") == "train"
    assert labels.split_of("detector", "", "d05") == "val"
    assert labels.split_of("detector", "", "d01") == "train"
    assert labels.split_of("s05", "", "s05") == "train"


def test_is_kept_every_other_frame():
    assert [labels.is_kept(t) for t in (0.0, 0.2, 0.4, 0.6)] == [True, False, True, False]


def test_shrink_keeps_aspect_and_leaves_small_images():
    small, scale = labels.shrink(np.zeros((720, 1280, 3), np.uint8))
    assert small.shape[:2] == (360, 640) and scale == pytest.approx(0.5)
    same, one = labels.shrink(np.zeros((100, 200, 3), np.uint8))
    assert same.shape[:2] == (100, 200) and one == 1.0


def test_label_rows_scales_then_pads():
    """원본 박스를 저장 사진 배율로 줄인 뒤 여백을 더한다."""
    shot = _shot(0.4, [BBox(100, 100, 200, 200)])
    shot.scale, shot.size = 0.5, (640, 360)
    lines, rows = labels.label_rows(shot)
    assert len(lines) == 1 and len(rows) == 1
    assert (rows[0]["x1"], rows[0]["x2"]) == (42.5, 107.5)
    assert lines[0].startswith("0 0.117188 0.208333")


def test_label_rows_background_writes_empty_label_and_one_row():
    shot = labels.Shot(image_id="d01_0001", source="detector", video="d01", stage="background")
    shot.size = (640, 360)
    lines, rows = labels.label_rows(shot)
    assert lines == []
    assert len(rows) == 1 and rows[0]["x1"] == "" and rows[0]["stage"] == "background"


def test_format_report_counts_labeled_frames():
    counts = Counter({("hand", "full"): 2, ("hand", "g2x2"): 1, ("hand", "none"): 1})
    report = labels.format_report([], counts)
    assert "정답 붙은 프레임 3/4 (75.0%)" in report


# --- 학습 묶음 ------------------------------------------------------------------


def _boxes_dir(root: Path) -> Path:
    """사진 셋(손 영상 hall · loby, 검출기 d05 배경)의 자동 정답 폴더."""
    boxes = root / "boxes"
    for kind in ("images", "labels"):
        (boxes / kind).mkdir(parents=True)
    shots = [
        labels.Shot(
            image_id="h_hall_p1_000400",
            source="hand",
            video="h_hall_p1",
            stage="full",
            boxes=[BBox(10, 10, 50, 50)],
            place="hall",
            person="p1",
            cls="point",
        ),
        labels.Shot(
            image_id="h_loby_p2_000800",
            source="hand",
            video="h_loby_p2",
            stage="sq3",
            boxes=[BBox(20, 20, 40, 40)],
            place="loby",
            person="p2",
            cls="open",
        ),
        labels.Shot(image_id="d05_0001", source="detector", video="d05", stage="background"),
    ]
    rows = []
    for shot in shots:
        shot.size = (640, 360)
        (boxes / "images" / f"{shot.image_id}.jpg").write_bytes(b"jpg")
        rows += labels.write_label(boxes, shot)
    labels.write_rows(boxes / "boxes.csv", rows)
    return boxes


def test_run_build_splits_by_place_and_zips_with_notebook(tmp_path):
    boxes = _boxes_dir(tmp_path)
    notebook = tmp_path / "hand_box_finetune.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    out, upload = tmp_path / "yolo_h1", tmp_path / "drive_upload"

    labels.run_build(boxes, out, upload, notebook)

    assert (out / "images" / "train" / "h_hall_p1_000400.jpg").exists()
    assert (out / "images" / "val" / "h_loby_p2_000800.jpg").exists()
    assert (out / "labels" / "val" / "d05_0001.txt").read_text(encoding="utf-8") == ""
    yaml = (out / "data.yaml").read_text(encoding="utf-8")
    assert "path:" not in yaml and '0: "hand"' in yaml
    assert "train: 사진 1 (배경 0) · 손 박스 1" in (out / "prep_report.txt").read_text("utf-8")
    with zipfile.ZipFile(upload / labels.ZIP_NAME) as z:
        names = set(z.namelist())
    assert {"yolo_h1/data.yaml", "yolo_h1/images/val/d05_0001.jpg"} <= names
    assert (upload / notebook.name).exists()


def test_run_build_refuses_to_overwrite(tmp_path):
    boxes = _boxes_dir(tmp_path)
    notebook = tmp_path / "hand_box_finetune.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    out, upload = tmp_path / "yolo_h1", tmp_path / "drive_upload"
    labels.run_build(boxes, out, upload, notebook)
    with pytest.raises(SystemExit):
        labels.run_build(boxes, out, upload, notebook)
