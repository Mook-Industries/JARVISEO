"""tools/auto_label.py — 모델 없이 검사할 수 있는 부분과, 가짜 선생으로 명령 한 바퀴.

CI 에는 ultralytics 가 없다. 틀리면 학습 라벨이 조용히 망가지는 곳을 잡는다 —
종류 번호 순서 · 글귀 표 · YOLO 좌표 · 겹친 박스 합치기 · Label Studio 형식 · 선생 성적 계산 ·
이미 만든 라벨 덮어쓰기.

tools/ 는 패키지가 아니라서 파일 경로로 직접 불러온다.
"""

import csv
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from jarviseo.pointing.detector import FINETUNE_CLASSES
from jarviseo.types import BBox, Detection

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "auto_label.py"
_spec = importlib.util.spec_from_file_location("auto_label", _TOOL_PATH)
auto_label = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(auto_label)


def _set_argv(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["auto_label.py", *args])


def _det(label: str, box: tuple[float, float, float, float], conf: float = 0.9) -> Detection:
    return Detection(label=label, confidence=conf, bbox=BBox(*box))


def _write_jpg(path: Path, width: int = 200, height: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    buffer.tofile(path)


# --- 종류 · 글귀 표 ----------------------------------------------------------------


def test_finetune_classes_order_is_fixed():
    """라벨 파일에는 번호가 적힌다. 순서가 바뀌면 이미 만든 라벨이 전부 다른 물건이 된다."""
    assert FINETUNE_CLASSES == (
        "bottle",
        "cup",
        "bowl",
        "laptop",
        "mouse",
        "keyboard",
        "cell phone",
        "remote",
        "book",
        "tv",
        "product",
    )


def test_prompt_table_covers_every_class():
    """모든 종류에 글귀가 있고, 글귀 번호가 원래 종류로 돌아간다."""
    phrases, owners = auto_label.prompt_table()
    assert len(phrases) == len(owners)
    assert set(owners) == set(range(len(FINETUNE_CLASSES)))
    product = FINETUNE_CLASSES.index("product")
    assert phrases[owners.index(product)] == "snack bag"
    assert owners[phrases.index("computer monitor")] == FINETUNE_CLASSES.index("tv")


def test_prompt_table_rejects_unknown_or_missing_class():
    """표가 종류 표와 어긋나면 멈춘다. 그대로 두면 그 종류는 학습에서 조용히 사라진다."""
    with pytest.raises(ValueError, match="모르는 종류"):
        auto_label.prompt_table({**auto_label.PROMPTS, "chair": ("chair",)})
    missing = {k: v for k, v in auto_label.PROMPTS.items() if k != "mouse"}
    with pytest.raises(ValueError, match="mouse"):
        auto_label.prompt_table(missing)


def test_coco_to_class_keeps_only_our_classes():
    """COCO 선생에서는 11종 안 이름만 남는다. person · chair 는 버리고 product 는 원래 없다."""
    names = {0: "person", 39: "bottle", 56: "chair", 62: "tv", 67: "cell phone"}
    assert auto_label.coco_to_class(names) == {39: "bottle", 62: "tv", 67: "cell phone"}


# --- YOLO 라벨 ---------------------------------------------------------------------


def test_to_yolo_line_normalizes_center_and_size():
    """200×100 사진의 (50,25)~(150,75) → 중심 0.5 0.5, 크기 0.5 0.5."""
    line = auto_label.to_yolo_line(3, BBox(50, 25, 150, 75), 200, 100)
    assert line == "3 0.500000 0.500000 0.500000 0.500000"


def test_to_yolo_line_clips_box_outside_image_first():
    """사진 밖으로 나간 부분은 먼저 잘라낸다. (-20,0)~(100,100) → (0,0)~(100,100)."""
    line = auto_label.to_yolo_line(0, BBox(-20, 0, 100, 100), 200, 100)
    assert line == "0 0.250000 0.500000 0.500000 1.000000"


# --- 겹친 박스 합치기 ---------------------------------------------------------------


def test_merge_boxes_keeps_higher_confidence_across_classes():
    """같은 캔에 cup(0.6) 과 product(0.8) 가 겹치면 종류와 상관없이 product 하나만."""
    cup = _det("cup", (0, 0, 100, 100), 0.6)
    product = _det("product", (2, 2, 100, 100), 0.8)
    assert auto_label.merge_boxes([cup, product]) == [product]


def test_merge_boxes_keeps_separate_objects():
    left, right = _det("cup", (0, 0, 50, 50)), _det("cup", (100, 0, 150, 50), 0.5)
    assert auto_label.merge_boxes([right, left]) == [left, right]


# --- Label Studio ------------------------------------------------------------------


def test_to_label_studio_uses_percent_from_top_left():
    """좌표는 왼쪽 위 기준 % 단위, 이름은 설정 파일과 같은 label · image."""
    task = auto_label.to_label_studio(
        "frames/d01/d01_0001.jpg", [_det("bottle", (50, 25, 150, 75), 0.7)], 200, 100, "world"
    )
    assert task["data"]["image"] == "/data/local-files/?d=frames/d01/d01_0001.jpg"
    (prediction,) = task["predictions"]
    assert prediction["model_version"] == "world"
    (box,) = prediction["result"]
    assert (box["from_name"], box["to_name"], box["type"]) == ("label", "image", "rectanglelabels")
    assert box["value"]["rectanglelabels"] == ["bottle"]
    assert [box["value"][k] for k in ("x", "y", "width", "height")] == [25.0, 25.0, 50.0, 50.0]
    assert (box["original_width"], box["original_height"]) == (200, 100)


def test_label_config_lists_classes_in_order():
    """설정 파일은 올바른 XML 이고 종류를 FINETUNE_CLASSES 순서로 담는다."""
    root = ET.fromstring(auto_label.label_config())
    assert root.find("Image").get("name") == "image"
    labels = root.find("RectangleLabels")
    assert (labels.get("name"), labels.get("toName")) == ("label", "image")
    assert [x.get("value") for x in labels.findall("Label")] == list(FINETUNE_CLASSES)


# --- 선생 비교 ---------------------------------------------------------------------


def _write_labels(path: Path, rows: list[dict]) -> None:
    columns = ["file", "session", "split", "gt_x1", "gt_y1", "gt_x2", "gt_y2", "gt_label"]
    columns += ["excluded"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({"gt_x1": 0, "gt_y1": 0, "gt_x2": 10, "gt_y2": 10, **row})


def test_load_samples_reads_only_chosen_valid_sessions(tmp_path):
    """test 세션 · 제외한 사진 · 고르지 않은 세션(말 세션 s04)은 읽지 않는다."""
    base = {"gt_label": "cup", "excluded": "0"}
    _write_labels(
        tmp_path / "labels.csv",
        [
            {**base, "file": "s01/a.jpg", "session": "1", "split": "valid"},
            {**base, "file": "s01/b.jpg", "session": "1", "split": "valid", "excluded": "1"},
            {**base, "file": "s03/c.jpg", "session": "3", "split": "test"},
            {**base, "file": "s04/d.jpg", "session": "4", "split": "valid"},
        ],
    )
    samples = auto_label.load_samples(tmp_path, [1, 2])
    assert [s["file"] for s in samples] == ["s01/a.jpg"]
    assert samples[0]["gt"] == BBox(0, 0, 10, 10)


def test_load_samples_without_labels_file_is_empty(tmp_path):
    assert auto_label.load_samples(tmp_path, [1, 2]) == []


def test_recall_counts_box_class_and_product_separately():
    """박스는 맞고 종류가 틀린 사진, 11종 밖 정답, product 를 따로 센다."""
    gt = BBox(0, 0, 100, 100)
    samples = [
        {"file": "a", "gt": gt, "gt_label": "bottle"},  # 박스 · 종류 다 맞음
        {"file": "b", "gt": gt, "gt_label": "can"},  # 박스는 맞고 종류 틀림(cup 으로 잡음)
        {"file": "c", "gt": gt, "gt_label": "glasses"},  # 11종 밖: 박스만 센다
        {"file": "d", "gt": gt, "gt_label": "snack bag"},  # 못 잡음
    ]
    detections = {
        "a": [_det("bottle", (0, 0, 100, 100))],
        "b": [_det("cup", (5, 5, 100, 100)), _det("book", (300, 300, 400, 400))],
        "c": [_det("cup", (0, 0, 90, 100))],
        "d": [],
    }
    r = auto_label.recall(samples, detections)
    assert r["box_recall"] == 3 / 4
    assert (r["class_total"], r["class_recall"]) == (3, 1 / 3)
    assert (r["product_total"], r["product_recall"]) == (2, 0.0)
    assert r["boxes_per_photo"] == 4 / 4


def test_recall_with_no_samples_has_no_rates():
    r = auto_label.recall([], {})
    assert r["box_recall"] is None and r["class_recall"] is None


def test_format_recall_is_a_table_row():
    r = {
        "photos": 62,
        "box_recall": 0.5,
        "class_total": 54,
        "class_recall": 0.25,
        "product_total": 11,
        "product_recall": None,
        "boxes_per_photo": 3.25,
    }
    row = auto_label.format_recall("world", r)
    assert row == "| world | 50.0% (62장) | 25.0% (54장) | - (11장) | 3.2 |"


# --- 명령 한 바퀴 (가짜 선생) ---------------------------------------------------------


def _fake_teacher(name: str, conf: float):
    """사진마다 왼쪽 위 bottle 하나를 찾는 선생. 이름은 기록용으로 받기만 한다."""
    return lambda image: [_det("bottle", (0, 0, 100, 50), 0.8)]


def test_label_writes_yolo_labels_tasks_and_config(monkeypatch, tmp_path, capsys):
    _write_jpg(tmp_path / "frames" / "d01" / "d01_0001.jpg")
    _write_jpg(tmp_path / "frames" / "d02" / "d02_0001.jpg")
    monkeypatch.setattr(auto_label, "make_teacher", _fake_teacher)
    _set_argv(monkeypatch, "label", "--root", str(tmp_path))
    auto_label.main()

    out = tmp_path / "auto"
    assert (out / "labels" / "d01_0001.txt").read_text(encoding="utf-8") == (
        "0 0.250000 0.250000 0.500000 0.500000\n"
    )
    tasks = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    assert [t["data"]["image"] for t in tasks] == [
        "/data/local-files/?d=frames/d01/d01_0001.jpg",
        "/data/local-files/?d=frames/d02/d02_0001.jpg",
    ]
    assert (out / "label_config.xml").exists()
    printed = capsys.readouterr().out
    assert "사진 2장 · 박스 2개" in printed
    assert "bottle: 2" in printed


def test_label_writes_empty_file_for_photo_without_boxes(monkeypatch, tmp_path):
    """박스가 없는 사진도 빈 라벨 파일을 둔다. "물건 없는 사진"이라는 뜻이다."""
    _write_jpg(tmp_path / "frames" / "d01" / "d01_0001.jpg")
    monkeypatch.setattr(auto_label, "make_teacher", lambda name, conf: lambda image: [])
    _set_argv(monkeypatch, "label", "--root", str(tmp_path))
    auto_label.main()
    assert (tmp_path / "auto" / "labels" / "d01_0001.txt").read_text(encoding="utf-8") == ""


def test_label_refuses_to_overwrite_existing_labels(monkeypatch, tmp_path):
    """R1 이 학습한 라벨을 덮어쓰면 R1 을 다시 만들 수 없다."""
    _write_jpg(tmp_path / "frames" / "d01" / "d01_0001.jpg")
    (tmp_path / "auto").mkdir()
    monkeypatch.setattr(auto_label, "make_teacher", _fake_teacher)
    _set_argv(monkeypatch, "label", "--root", str(tmp_path))
    with pytest.raises(SystemExit, match="이미 있습니다"):
        auto_label.main()


def test_label_without_frames_stops(monkeypatch, tmp_path):
    _set_argv(monkeypatch, "label", "--root", str(tmp_path))
    with pytest.raises(SystemExit, match="extract_frames"):
        auto_label.main()


def test_compare_prints_one_row_per_teacher(monkeypatch, tmp_path, capsys):
    _write_jpg(tmp_path / "raw" / "s01" / "s01_desk_0001.jpg")
    _write_labels(
        tmp_path / "labels.csv",
        [
            {
                "file": "s01/s01_desk_0001.jpg",
                "session": "1",
                "split": "valid",
                "gt_x1": 0,
                "gt_y1": 0,
                "gt_x2": 100,
                "gt_y2": 50,
                "gt_label": "bottle",
                "excluded": "0",
            }
        ],
    )
    monkeypatch.setattr(auto_label, "make_teacher", _fake_teacher)
    _set_argv(monkeypatch, "compare", "--pointing", str(tmp_path))
    auto_label.main()
    printed = capsys.readouterr().out
    assert "valid 1장 (세션 1 2)" in printed
    assert "| world | 100.0% (1장) | 100.0% (1장) | - (0장) | 1.0 |" in printed
    assert "| coco |" in printed


# --- 인자 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [["label", "--conf", "0"], ["label", "--conf", "1.5"], ["label", "--teacher", "coco"], []],
)
def test_parse_args_rejects_bad_values(monkeypatch, args):
    """conf 는 0~1 사이, label 의 선생은 world · mix 만(coco 는 product 가 없다), 명령은 꼭."""
    _set_argv(monkeypatch, *args)
    with pytest.raises(SystemExit):
        auto_label.parse_args()


def test_read_image_rejects_broken_file(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="사진을 읽지 못했습니다"):
        auto_label.read_image(path)


@pytest.mark.parametrize("args", [["build", "--labels", "x", "--out", "y"], ["build"]])
def test_parse_args_build_needs_labels_val_and_out(monkeypatch, args):
    """build 는 라벨 폴더 · 검증 영상 · 회차 폴더를 꼭 받는다."""
    _set_argv(monkeypatch, *args)
    with pytest.raises(SystemExit):
        auto_label.parse_args()


def test_parse_args_build_rejects_val_that_is_not_video_number(monkeypatch):
    """--val 5 처럼 쓰면 아무 영상도 검증으로 안 빠져 조용히 틀린다."""
    _set_argv(monkeypatch, "build", "--labels", "x", "--val", "5", "--out", "y")
    with pytest.raises(SystemExit):
        auto_label.parse_args()


# --- 학습 묶음 (build) --------------------------------------------------------------


def test_split_of_uses_video_number_not_photo():
    """같은 영상의 사진은 모두 같은 쪽으로 간다."""
    assert auto_label.split_of("d05_0001", {"d05"}) == "val"
    assert auto_label.split_of("d05_0099", {"d05"}) == "val"
    assert auto_label.split_of("d01_0001", {"d05"}) == "train"


def test_class_names_file_looks_in_folder_then_parent(tmp_path):
    """Label Studio 내보내기는 classes.txt 를 labels 위 폴더에 둔다."""
    labels = tmp_path / "export" / "labels"
    labels.mkdir(parents=True)
    assert auto_label.class_names_file(labels) is None
    (tmp_path / "export" / "classes.txt").write_text("cup\n", encoding="utf-8")
    assert auto_label.class_names_file(labels) == tmp_path / "export" / "classes.txt"
    (labels / "classes.txt").write_text("cup\n", encoding="utf-8")
    assert auto_label.class_names_file(labels) == labels / "classes.txt"


def test_remap_table_matches_by_name_not_order():
    """내보낸 순서가 달라도 이름으로 우리 번호를 찾는다."""
    table = auto_label.remap_table(["product", "bottle", "cell phone"])
    assert table == {0: 10, 1: 0, 2: 6}


def test_remap_table_rejects_unknown_name():
    with pytest.raises(ValueError, match="chair"):
        auto_label.remap_table(["bottle", "chair"])


def test_remap_line_changes_only_class_number():
    assert auto_label.remap_line("1 0.5 0.5 0.2 0.2", {1: 0}) == "0 0.5 0.5 0.2 0.2"
    assert auto_label.remap_line("10 0.1 0.2 0.3 0.4", None) == "10 0.1 0.2 0.3 0.4"


@pytest.mark.parametrize(
    ("line", "table"), [("11 0.5 0.5 0.1 0.1", None), ("3 0.5 0.5 0.1 0.1", {0: 0})]
)
def test_remap_line_rejects_number_out_of_range(line, table):
    """우리 표 밖 번호(11번)나 classes.txt 에 없는 번호는 멈춘다."""
    with pytest.raises(ValueError, match="밖입니다"):
        auto_label.remap_line(line, table)


def test_data_yaml_has_no_path_key_and_lists_classes_in_order():
    """path 를 넣으면 상대 경로가 지금 작업 폴더 기준이 돼 맥과 Colab 에서 다른 곳을 가리킨다."""
    text = auto_label.data_yaml()
    assert "path:" not in text
    assert text.startswith("train: images/train\nval: images/val\nnames:\n")
    assert '  6: "cell phone"' in text
    yaml = pytest.importorskip("yaml")
    loaded = yaml.safe_load(text)
    assert loaded["names"] == dict(enumerate(FINETUNE_CLASSES))


def test_rare_classes_compares_with_most_common_class():
    """가장 많은 종류(100)의 10% 미만인 종류만. 0개인 종류도 걸린다."""
    counts = Counter({name: 100 for name in FINETUNE_CLASSES})
    counts["remote"] = 9
    counts["mouse"] = 10
    del counts["tv"]
    assert auto_label.rare_classes(counts) == ["remote", "tv"]


def _write_label(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_root(tmp_path: Path) -> Path:
    """d01 사진 2장 · d05 사진 1장 · d02 사진 1장(라벨 없음)."""
    for name in ("d01/d01_0001", "d01/d01_0002", "d05/d05_0001", "d02/d02_0001"):
        _write_jpg(tmp_path / "frames" / f"{name}.jpg")
    return tmp_path


def _build_argv(monkeypatch, root: Path, labels: Path, out: Path, val: str = "d05") -> None:
    _set_argv(
        monkeypatch,
        "build",
        "--root",
        str(root),
        "--labels",
        str(labels),
        "--val",
        val,
        "--out",
        str(out),
    )


def test_build_splits_by_video_copies_and_writes_data_yaml(monkeypatch, tmp_path, capsys):
    root = _build_root(tmp_path)
    labels = root / "auto" / "labels"
    _write_label(labels / "d01_0001.txt", "0 0.5 0.5 0.2 0.2\n")
    _write_label(labels / "d01_0002.txt", "")  # 물건 없는 사진
    _write_label(labels / "d05_0001.txt", "10 0.5 0.5 0.2 0.2\n6 0.1 0.1 0.1 0.1\n")
    out = root / "yolo_r1"
    _build_argv(monkeypatch, root, labels, out)
    auto_label.main()

    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == [
        "d01_0001.jpg",
        "d01_0002.jpg",
    ]
    assert [p.name for p in (out / "images" / "val").iterdir()] == ["d05_0001.jpg"]
    assert (out / "labels" / "train" / "d01_0002.txt").read_text(encoding="utf-8") == ""
    assert (out / "labels" / "val" / "d05_0001.txt").read_text(encoding="utf-8") == (
        "10 0.5 0.5 0.2 0.2\n6 0.1 0.1 0.1 0.1\n"
    )
    assert (out / "data.yaml").read_text(encoding="utf-8") == auto_label.data_yaml()
    printed = capsys.readouterr().out
    assert "학습 2장 · 검증 1장 (검증 영상 d05) · 라벨 없어 뺀 사진 1장" in printed
    assert "  bottle | 1 | 0" in printed
    assert "  product | 0 | 1" in printed


def test_build_remaps_label_studio_export_by_name(monkeypatch, tmp_path, capsys):
    """Label Studio 가 product=0, bottle=1 로 내보내도 우리 번호(10, 0)로 맞춘다."""
    root = _build_root(tmp_path)
    export = root / "reviewed"
    _write_label(export / "classes.txt", "product\nbottle\n")
    _write_label(export / "labels" / "d01_0001.txt", "1 0.5 0.5 0.2 0.2\n")
    _write_label(export / "labels" / "d05_0001.txt", "0 0.5 0.5 0.2 0.2\n")
    out = root / "yolo_r2"
    _build_argv(monkeypatch, root, export / "labels", out)
    auto_label.main()

    assert (out / "labels" / "train" / "d01_0001.txt").read_text(encoding="utf-8") == (
        "0 0.5 0.5 0.2 0.2\n"
    )
    assert (out / "labels" / "val" / "d05_0001.txt").read_text(encoding="utf-8") == (
        "10 0.5 0.5 0.2 0.2\n"
    )
    assert "종류 번호를 이름으로 다시 맞춤" in capsys.readouterr().out


def test_build_refuses_existing_round_folder(monkeypatch, tmp_path):
    """회차 폴더를 덮어쓰면 그 회차를 다시 학습할 수 없다."""
    root = _build_root(tmp_path)
    (root / "yolo_r1").mkdir()
    _build_argv(monkeypatch, root, root, root / "yolo_r1")
    with pytest.raises(SystemExit, match="이미 있습니다"):
        auto_label.main()


def test_build_stops_without_validation_photos(monkeypatch, tmp_path):
    """검증 영상 번호를 잘못 적으면(사진이 없는 d09) 학습 전에 멈춘다."""
    root = _build_root(tmp_path)
    labels = root / "auto" / "labels"
    _write_label(labels / "d01_0001.txt", "0 0.5 0.5 0.2 0.2\n")
    out = root / "yolo_r1"
    _build_argv(monkeypatch, root, labels, out, val="d09")
    with pytest.raises(SystemExit, match="d09"):
        auto_label.main()
    assert not out.exists()


def test_build_stops_on_unknown_class_before_writing(monkeypatch, tmp_path):
    """classes.txt 에 우리 표에 없는 이름이 있으면 아무것도 쓰기 전에 멈춘다."""
    root = _build_root(tmp_path)
    export = root / "reviewed"
    _write_label(export / "classes.txt", "chair\n")
    _write_label(export / "labels" / "d01_0001.txt", "0 0.5 0.5 0.2 0.2\n")
    out = root / "yolo_r2"
    _build_argv(monkeypatch, root, export / "labels", out)
    with pytest.raises(SystemExit, match="chair"):
        auto_label.main()
    assert not out.exists()
