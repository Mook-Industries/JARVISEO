"""tools/auto_label.py — 모델 없이 검사할 수 있는 부분과, 가짜 선생으로 명령 한 바퀴.

CI 에는 ultralytics · CLIP 이 없다. 틀리면 학습 라벨이 조용히 망가지는 곳을 잡는다 —
종류 번호 순서 · 글귀 표 · YOLO 좌표 · 겹친 박스 합치기 · Label Studio 형식(오가는 변환) ·
선생 성적 계산 · 검수 초안 만들기 · 이미 만든 라벨 덮어쓰기.

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
# 아래 autouse 픽스처가 바꿔 끼우기 전의 진짜 검사. 검사 자체는 이것으로 테스트한다.
_require_clip = auto_label.require_clip


@pytest.fixture(autouse=True)
def _skip_clip_check(monkeypatch):
    """CI 에는 CLIP 이 없다. 명령 테스트는 가짜 모델을 쓰므로 CLIP 검사를 건너뛴다."""
    monkeypatch.setattr(auto_label, "require_clip", lambda: None)


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
    """라벨 파일에는 번호가 적힌다. 순서가 바뀌면 이미 만든 라벨이 전부 다른 물건이 된다.

    R2 부터 10종이다(remote 를 뺐다). R1 의 11종 번호와 다르다.
    """
    assert FINETUNE_CLASSES == (
        "bottle",
        "cup",
        "bowl",
        "laptop",
        "mouse",
        "keyboard",
        "cell phone",
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


def test_prompts_do_not_send_instant_rice_to_product():
    """햇반은 bowl 이다(이름은 모양). product 글귀로 찾으면 초안이 product 로 붙는다."""
    assert not any("rice" in phrase for phrase in auto_label.PROMPTS["product"])


def test_prompt_table_rejects_unknown_or_missing_class():
    """표가 종류 표와 어긋나면 멈춘다. 그대로 두면 그 종류는 학습에서 조용히 사라진다."""
    with pytest.raises(ValueError, match="모르는 종류"):
        auto_label.prompt_table({**auto_label.PROMPTS, "chair": ("chair",)})
    missing = {k: v for k, v in auto_label.PROMPTS.items() if k != "mouse"}
    with pytest.raises(ValueError, match="mouse"):
        auto_label.prompt_table(missing)


def test_coco_to_class_keeps_only_our_classes():
    """COCO 선생에서는 10종 안 이름만 남는다. person · chair · remote 는 버린다."""
    names = {0: "person", 39: "bottle", 56: "chair", 62: "tv", 65: "remote", 67: "cell phone"}
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


def test_label_config_turns_on_brightness_and_contrast():
    """검수 화면에서 밝기 · 대비를 조절할 수 있다.

    어두운 d04 사진의 물건을 못 보고 지나치지 않게.
    """
    image = ET.fromstring(auto_label.label_config()).find("Image")
    assert (image.get("brightnessControl"), image.get("contrastControl")) == ("true", "true")


def test_label_config_turns_off_box_rotation():
    """YOLO 박스는 돌릴 수 없다. 검수 중에 돌린 박스는 변환할 때 틀어진다."""
    labels = ET.fromstring(auto_label.label_config()).find("RectangleLabels")
    assert labels.get("canRotate") == "false"


def test_label_config_shows_note_only_when_asked():
    """review 작업만 note 가 있다. 없는 칸을 설정에 넣으면 R1 작업을 가져올 때 걸린다."""
    assert ET.fromstring(auto_label.label_config()).find("Header") is None
    header = ET.fromstring(auto_label.label_config(with_note=True)).find("Header")
    assert header.get("value") == "$note"


def test_from_label_studio_round_trips_to_label_studio():
    """픽셀 → % → 픽셀로 돌아오면 같은 박스 · 같은 이름 · 같은 신뢰도."""
    dets = [_det("bottle", (50, 25, 150, 75), 0.7), _det("product", (0, 0, 40, 100), 0.3)]
    task = auto_label.to_label_studio("frames/d01/d01_0001.jpg", dets, 200, 100, "world")
    back = auto_label.from_label_studio(task["predictions"][0]["result"], 200, 100)
    assert [d.label for d in back] == ["bottle", "product"]
    assert [d.confidence for d in back] == [0.7, 0.3]
    for got, want in zip(back, dets, strict=True):
        got_box = (got.bbox.x1, got.bbox.y1, got.bbox.x2, got.bbox.y2)
        want_box = (want.bbox.x1, want.bbox.y1, want.bbox.x2, want.bbox.y2)
        assert got_box == pytest.approx(want_box)


def test_from_label_studio_clips_to_image_and_skips_other_results():
    """사진 끝을 조금 넘긴 박스는 사진 안으로 자르고, 박스가 아닌 결과는 건너뛴다.

    사람이 그린 박스에는 신뢰도(score)가 없어서 1.0 이다.
    """
    result = [
        {
            "type": "rectanglelabels",
            "value": {"x": 90, "y": -2, "width": 15, "height": 50, "rectanglelabels": ["cup"]},
        },
        {"type": "choices", "value": {"choices": ["ok"]}},
    ]
    (box,) = auto_label.from_label_studio(result, 200, 100)
    assert (box.bbox.x1, box.bbox.y1, box.bbox.x2, box.bbox.y2) == pytest.approx((180, 0, 200, 48))
    assert (box.label, box.confidence) == ("cup", 1.0)


def test_task_file_reads_path_from_local_files_url():
    task = {"data": {"image": "/data/local-files/?d=frames/d05/d05_0002.jpg"}}
    assert auto_label.task_file(task) == "frames/d05/d05_0002.jpg"


def test_prediction_result_without_predictions_is_empty():
    assert auto_label.prediction_result({"data": {}}) == []
    assert auto_label.prediction_result({"predictions": []}) == []


# --- 검수 초안 (review) --------------------------------------------------------------


def test_too_small_keeps_box_exactly_at_min_side():
    """짧은 변이 40px 미만이면 뺀다. 딱 40px 은 남긴다."""
    assert not auto_label.too_small(BBox(0, 0, 40, 300))
    assert auto_label.too_small(BBox(0, 0, 39.9, 300))
    assert auto_label.too_small(BBox(0, 0, 300, 10), min_side=20)


def test_fill_gaps_adds_only_where_base_has_no_box():
    """R1 초안과 절반 넘게 겹치는 식품 박스는 버리고(신뢰도가 높아도), 빈 자리 것만 더한다."""
    keyboard = _det("keyboard", (0, 0, 100, 40), 0.3)
    over_keyboard = _det("product", (2, 0, 100, 40), 0.9)
    in_gap = _det("product", (120, 0, 180, 60), 0.2)
    assert auto_label.fill_gaps([keyboard], [over_keyboard, in_gap]) == [keyboard, in_gap]


def test_review_order_starts_with_even_val_then_alternates_training_videos():
    """d05 짝수 → d01 · d02 를 번갈아(영상 안에서는 짝수 먼저) → d05 홀수."""
    stems = [
        "d01_0001",
        "d01_0002",
        "d01_0003",
        "d01_0004",
        "d02_0001",
        "d02_0002",
        "d05_0001",
        "d05_0002",
        "d05_0003",
        "d05_0004",
    ]
    order = auto_label.review_order(stems, {"d05"})
    assert sorted(order, key=order.get) == [
        "d05_0002",
        "d05_0004",
        "d01_0002",
        "d02_0002",
        "d01_0004",
        "d02_0001",
        "d01_0001",
        "d01_0003",
        "d05_0001",
        "d05_0003",
    ]
    assert sorted(order.values()) == list(range(1, len(stems) + 1))


def test_require_clip_stops_when_clip_is_missing(monkeypatch):
    """CLIP 이 없으면 모델을 부르기 전에 멈춘다. 그대로 두면 ultralytics 가 팀 환경에 설치한다."""
    monkeypatch.setattr(auto_label.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(SystemExit, match="CLIP"):
        _require_clip()


def test_require_clip_passes_when_clip_is_installed(monkeypatch):
    monkeypatch.setattr(auto_label.importlib.util, "find_spec", lambda name: object())
    _require_clip()


def _write_drafts(path: Path, tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks), encoding="utf-8")


def _review_root(tmp_path: Path) -> Path:
    """d01 사진 1장(remote · bottle · 작은 cup 초안) · d05 사진 2장(초안 없음)."""
    for name in ("d01/d01_0001", "d05/d05_0001", "d05/d05_0002"):
        _write_jpg(tmp_path / "frames" / f"{name}.jpg")
    d01 = [
        _det("remote", (0, 0, 60, 50), 0.4),
        _det("bottle", (100, 0, 160, 50), 0.6),
        _det("cup", (0, 60, 20, 80), 0.5),  # 짧은 변 20px — 뺀다
    ]
    _write_drafts(
        tmp_path / "auto" / "tasks.json",
        [
            auto_label.to_label_studio("frames/d01/d01_0001.jpg", d01, 200, 100, "world"),
            auto_label.to_label_studio("frames/d05/d05_0001.jpg", [], 200, 100, "world"),
            auto_label.to_label_studio("frames/d05/d05_0002.jpg", [], 200, 100, "world"),
        ],
    )
    return tmp_path


def _fake_product_finder(conf: float):
    """bottle 과 겹치는 식품 · 빈 자리(remote 를 뺀 자리) 식품 · 작은 식품을 찾는 YOLOE."""
    return lambda image: [
        _det("product", (102, 2, 160, 50), 0.9),
        _det("product", (0, 0, 60, 50), 0.2),
        _det("product", (170, 60, 190, 80), 0.8),
    ]


def test_review_builds_tasks_in_review_order_with_note(monkeypatch, tmp_path, capsys):
    root = _review_root(tmp_path)
    monkeypatch.setattr(auto_label, "make_product_finder", _fake_product_finder)
    _set_argv(monkeypatch, "review", "--root", str(root))
    auto_label.main()

    tasks = json.loads((root / "review" / "tasks.json").read_text(encoding="utf-8"))
    assert [auto_label.task_file(t) for t in tasks] == [
        "frames/d05/d05_0002.jpg",
        "frames/d01/d01_0001.jpg",
        "frames/d05/d05_0001.jpg",
    ]
    d01 = tasks[1]
    assert (d01["data"]["video"], d01["data"]["order"]) == ("d01", 2)
    assert "remote 1개" in d01["data"]["note"]
    assert tasks[0]["data"]["note"] == ""
    (prediction,) = d01["predictions"]
    assert prediction["model_version"] == "r1+yoloe"
    labels = [box["value"]["rectanglelabels"][0] for box in prediction["result"]]
    assert labels == ["bottle", "product"]  # remote · 작은 cup 은 빠지고, 빈 자리 식품만 더해짐
    config_xml = (root / "review" / "label_config.xml").read_text(encoding="utf-8")
    assert ET.fromstring(config_xml).find("Header").get("value") == "$note"

    printed = capsys.readouterr().out
    assert "검수 작업 3장 (검증 d05 2장)" in printed
    assert "10종 밖이라 뺀 R1 초안: remote 1개 · 사진 1장 d01_0001" in printed
    assert "R1 초안 1개 · YOLOE 3개" in printed


def test_review_refuses_to_overwrite_existing_review(monkeypatch, tmp_path):
    """검수 도중 덮어쓰면 Label Studio 로 이미 가져간 작업과 어긋난다."""
    root = _review_root(tmp_path)
    (root / "review").mkdir()
    monkeypatch.setattr(auto_label, "make_product_finder", _fake_product_finder)
    _set_argv(monkeypatch, "review", "--root", str(root))
    with pytest.raises(SystemExit, match="이미 있습니다"):
        auto_label.main()


def test_review_without_drafts_stops(monkeypatch, tmp_path):
    _write_jpg(tmp_path / "frames" / "d01" / "d01_0001.jpg")
    _set_argv(monkeypatch, "review", "--root", str(tmp_path))
    with pytest.raises(SystemExit, match="R1 초안이 없습니다"):
        auto_label.main()
    assert not (tmp_path / "review").exists()


def test_review_stops_when_draft_photo_is_missing(monkeypatch, tmp_path):
    """초안이 가리키는 사진이 없으면(--root 를 잘못 줌) 모델을 받기 전에 멈춘다."""
    _write_drafts(
        tmp_path / "auto" / "tasks.json",
        [auto_label.to_label_studio("frames/d01/d01_0001.jpg", [], 200, 100, "world")],
    )
    monkeypatch.setattr(auto_label, "make_product_finder", _fake_product_finder)
    _set_argv(monkeypatch, "review", "--root", str(tmp_path))
    with pytest.raises(SystemExit, match="1장이 없습니다"):
        auto_label.main()


# --- 가중치 초안 (R3) ----------------------------------------------------------------


def test_weight_class_table_accepts_our_ten_classes_in_any_order():
    names = dict(enumerate(reversed(FINETUNE_CLASSES)))
    assert auto_label.weight_class_table(names) == names


@pytest.mark.parametrize(
    "names",
    [
        dict(enumerate([*FINETUNE_CLASSES[:7], "remote", *FINETUNE_CLASSES[7:]])),  # R1 의 11종
        {0: "person", 1: "bicycle", 2: "bottle"},  # 사전학습 COCO 처럼 product 가 없다
    ],
)
def test_weight_class_table_rejects_r1_or_coco_weights(names):
    """R1(remote 포함)이나 COCO 가중치로 초안을 그리면 초안 기준이 회차마다 흔들린다."""
    with pytest.raises(SystemExit, match="10종 표와 다릅니다"):
        auto_label.weight_class_table(names)


def _weights_root(tmp_path: Path) -> Path:
    """d07 사진 2장 · d08 사진 1장 · 고르지 않을 d01 사진 1장 · 가중치 파일."""
    for name in ("d07/d07_0001", "d07/d07_0002", "d08/d08_0001", "d01/d01_0001"):
        _write_jpg(tmp_path / "frames" / f"{name}.jpg")
    (tmp_path / "detector_r2.pt").write_bytes(b"fake weights")
    return tmp_path


def _fake_weight_teacher(weights: Path, conf: float):
    """mouse 하나와 짧은 변 20px 짜리 cup 하나를 찾는 가중치."""
    return lambda image: [
        _det("mouse", (10, 10, 70, 60), 0.8),
        _det("cup", (100, 10, 120, 30), 0.7),  # 짧은 변 20px — 뺀다
    ]


def _weights_argv(monkeypatch, root: Path, *videos: str) -> None:
    _set_argv(
        monkeypatch,
        "review",
        "--root",
        str(root),
        "--weights",
        str(root / "detector_r2.pt"),
        "--videos",
        *(videos or ("d07", "d08")),
        "--out",
        str(root / "review_r3"),
    )


def test_review_with_weights_drafts_only_chosen_videos(monkeypatch, tmp_path, capsys):
    root = _weights_root(tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("R3 초안은 CLIP · YOLOE 를 부르지 않는다")

    monkeypatch.setattr(auto_label, "require_clip", fail)
    monkeypatch.setattr(auto_label, "make_product_finder", fail)
    seen = {}

    def teacher(weights: Path, conf: float):
        seen["args"] = (weights.name, conf)
        return _fake_weight_teacher(weights, conf)

    monkeypatch.setattr(auto_label, "make_weight_teacher", teacher)
    _weights_argv(monkeypatch, root)
    auto_label.main()

    out = root / "review_r3"
    tasks = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    # 고르지 않은 d01 은 빠지고, 영상을 번갈아 나온다(영상 안에서는 짝수 번 먼저).
    assert [auto_label.task_file(t) for t in tasks] == [
        "frames/d07/d07_0002.jpg",
        "frames/d08/d08_0001.jpg",
        "frames/d07/d07_0001.jpg",
    ]
    assert [(t["data"]["video"], t["data"]["order"], t["data"]["note"]) for t in tasks] == [
        ("d07", 1, ""),
        ("d08", 2, ""),
        ("d07", 3, ""),
    ]
    (prediction,) = tasks[0]["predictions"]
    assert prediction["model_version"] == "detector_r2"
    assert [box["value"]["rectanglelabels"][0] for box in prediction["result"]] == ["mouse"]
    assert seen["args"] == ("detector_r2.pt", auto_label.CONF)
    assert (out / "label_config.xml").exists()
    assert not (root / "review").exists()  # R2 검수 원본 자리는 건드리지 않는다

    printed = capsys.readouterr().out
    assert "검수 작업 3장 (d07 2장 · d08 1장) · 초안 박스 3개" in printed
    assert "  mouse: 3" in printed
    assert "초안: detector_r2.pt (conf 0.25) · 짧은 변 40px 미만이라 뺀 박스 3개" in printed


def test_review_with_weights_refuses_existing_out(monkeypatch, tmp_path):
    """R3 작업도 Label Studio 로 가져간 뒤 덮어쓰면 어긋난다."""
    root = _weights_root(tmp_path)
    (root / "review_r3").mkdir()
    monkeypatch.setattr(auto_label, "make_weight_teacher", _fake_weight_teacher)
    _weights_argv(monkeypatch, root)
    with pytest.raises(SystemExit, match="이미 있습니다"):
        auto_label.main()


def test_review_with_weights_stops_before_loading_when_video_folder_is_missing(
    monkeypatch, tmp_path
):
    """d09 사진을 아직 안 뽑았으면 가중치를 읽기 전에 멈춘다."""
    root = _weights_root(tmp_path)

    def must_not_load(weights: Path, conf: float):
        pytest.fail("가중치를 읽기 전에 멈춰야 한다")

    monkeypatch.setattr(auto_label, "make_weight_teacher", must_not_load)
    _weights_argv(monkeypatch, root, "d07", "d09")
    with pytest.raises(SystemExit, match="d09"):
        auto_label.main()
    assert not (root / "review_r3").exists()


def test_review_with_weights_stops_when_weights_file_is_missing(monkeypatch, tmp_path):
    root = _weights_root(tmp_path)
    (root / "detector_r2.pt").unlink()
    monkeypatch.setattr(auto_label, "make_weight_teacher", _fake_weight_teacher)
    _weights_argv(monkeypatch, root)
    with pytest.raises(SystemExit, match="가중치가 없습니다"):
        auto_label.main()


@pytest.mark.parametrize(
    "args",
    [
        ["review", "--weights", "w.pt"],  # 영상을 안 고름
        ["review", "--videos", "d07"],  # 가중치 없이 영상만
        ["review", "--weights", "w.pt", "--videos", "7"],  # 영상 번호 모양이 아님
    ],
)
def test_parse_args_review_weights_and_videos_go_together(monkeypatch, args):
    _set_argv(monkeypatch, *args)
    with pytest.raises(SystemExit):
        auto_label.parse_args()


def test_parse_args_review_conf_default_depends_on_draft_source(monkeypatch):
    """가중치 초안은 비서 검출과 같은 0.25, R2 갈래(YOLOE 식품 박스)는 그대로 0.1."""
    _set_argv(monkeypatch, "review", "--weights", "w.pt", "--videos", "d07")
    assert auto_label.parse_args().conf == auto_label.CONF
    _set_argv(monkeypatch, "review")
    assert auto_label.parse_args().conf == auto_label.REVIEW_CONF


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
    """박스는 맞고 종류가 틀린 사진, 10종 밖 정답, product 를 따로 센다."""
    gt = BBox(0, 0, 100, 100)
    samples = [
        {"file": "a", "gt": gt, "gt_label": "bottle"},  # 박스 · 종류 다 맞음
        {"file": "b", "gt": gt, "gt_label": "can"},  # 박스는 맞고 종류 틀림(cup 으로 잡음)
        {"file": "c", "gt": gt, "gt_label": "glasses"},  # 10종 밖: 박스만 센다
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
    [
        ["label", "--conf", "0"],
        ["label", "--conf", "1.5"],
        ["label", "--teacher", "coco"],
        ["review", "--conf", "1"],
        ["review", "--val", "5"],
        [],
    ],
)
def test_parse_args_rejects_bad_values(monkeypatch, args):
    """conf 는 0~1 사이, label 의 선생은 world · mix 만(coco 는 product 가 없다), review 의
    검증 영상은 d05 모양, 명령은 꼭."""
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
    assert table == {0: 9, 1: 0, 2: 6}


def test_remap_table_rejects_unknown_name():
    with pytest.raises(ValueError, match="chair"):
        auto_label.remap_table(["bottle", "chair"])


def test_remap_line_changes_only_class_number():
    assert auto_label.remap_line("1 0.5 0.5 0.2 0.2", {1: 0}) == "0 0.5 0.5 0.2 0.2"
    assert auto_label.remap_line("9 0.1 0.2 0.3 0.4", None) == "9 0.1 0.2 0.3 0.4"


@pytest.mark.parametrize(
    ("line", "table"), [("10 0.5 0.5 0.1 0.1", None), ("3 0.5 0.5 0.1 0.1", {0: 0})]
)
def test_remap_line_rejects_number_out_of_range(line, table):
    """우리 표 밖 번호(10번 — R1 의 11종 번호로는 product)나 classes.txt 에 없는 번호는 멈춘다."""
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
    counts["book"] = 9
    counts["mouse"] = 10
    del counts["tv"]
    assert auto_label.rare_classes(counts) == ["book", "tv"]


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
    _write_label(labels / "d05_0001.txt", "9 0.5 0.5 0.2 0.2\n6 0.1 0.1 0.1 0.1\n")
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
        "9 0.5 0.5 0.2 0.2\n6 0.1 0.1 0.1 0.1\n"
    )
    assert (out / "data.yaml").read_text(encoding="utf-8") == auto_label.data_yaml()
    printed = capsys.readouterr().out
    assert "학습 2장 · 검증 1장 (검증 영상 d05) · 라벨 없어 뺀 사진 1장" in printed
    assert "  bottle | 1 | 0" in printed
    assert "  product | 0 | 1" in printed


def test_build_remaps_label_studio_export_by_name(monkeypatch, tmp_path, capsys):
    """Label Studio 가 product=0, bottle=1 로 내보내도 우리 번호(9, 0)로 맞춘다."""
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
        "9 0.5 0.5 0.2 0.2\n"
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


def test_build_reads_validation_labels_from_val_labels_folder(monkeypatch, tmp_path, capsys):
    """대조군: 학습 사진은 자동 라벨, 검증 사진은 사람 라벨. 자동 라벨 폴더의 d05 는 안 쓴다."""
    root = _build_root(tmp_path)
    auto = root / "review" / "auto_labels"
    human = root / "review" / "labels"
    _write_label(auto / "d01_0001.txt", "0 0.5 0.5 0.2 0.2\n")
    _write_label(auto / "d05_0001.txt", "1 0.5 0.5 0.2 0.2\n")
    _write_label(human / "d05_0001.txt", "9 0.5 0.5 0.2 0.2\n")
    out = root / "yolo_r2auto"
    _build_argv(monkeypatch, root, auto, out)
    sys.argv += ["--val-labels", str(human)]
    auto_label.main()

    assert (out / "labels" / "train" / "d01_0001.txt").read_text(encoding="utf-8") == (
        "0 0.5 0.5 0.2 0.2\n"
    )
    assert (out / "labels" / "val" / "d05_0001.txt").read_text(encoding="utf-8") == (
        "9 0.5 0.5 0.2 0.2\n"
    )
    printed = capsys.readouterr().out
    assert "학습 1장 · 검증 1장" in printed
    assert f"검증 라벨은 {human} 에서 읽음" in printed


def _r3_argv(monkeypatch, root: Path, out: Path, *labels: Path) -> None:
    _set_argv(
        monkeypatch,
        "build",
        "--root",
        str(root),
        "--labels",
        *map(str, labels),
        "--val",
        "d05",
        "--out",
        str(out),
    )


def test_build_reads_labels_from_several_folders(monkeypatch, tmp_path, capsys):
    """R3 = R2 검수본 + R3 검수본. 사진마다 라벨이 있는 폴더 하나에서 읽는다."""
    root = _build_root(tmp_path)
    r2 = root / "review" / "labels"
    r3 = root / "review_r3" / "labels"
    _write_label(r2 / "d01_0001.txt", "0 0.5 0.5 0.2 0.2\n")
    _write_label(r2 / "d05_0001.txt", "9 0.5 0.5 0.2 0.2\n")
    _write_label(r3 / "d02_0001.txt", "4 0.5 0.5 0.2 0.2\n")
    out = root / "yolo_r3"
    _r3_argv(monkeypatch, root, out, r2, r3)
    auto_label.main()

    assert sorted(p.name for p in (out / "images" / "train").iterdir()) == [
        "d01_0001.jpg",
        "d02_0001.jpg",
    ]
    assert (out / "labels" / "train" / "d02_0001.txt").read_text(encoding="utf-8") == (
        "4 0.5 0.5 0.2 0.2\n"
    )
    assert [p.name for p in (out / "images" / "val").iterdir()] == ["d05_0001.jpg"]
    printed = capsys.readouterr().out
    assert "학습 2장 · 검증 1장 (검증 영상 d05) · 라벨 없어 뺀 사진 1장" in printed
    assert f"라벨을 가져온 폴더: {r2} 2장 · {r3} 1장" in printed


def test_build_stops_when_photo_has_labels_in_two_folders(monkeypatch, tmp_path):
    """같은 사진의 라벨이 두 폴더에 있으면 한쪽을 조용히 고르지 않고, 쓰기 전에 멈춘다."""
    root = _build_root(tmp_path)
    r2 = root / "review" / "labels"
    r3 = root / "review_r3" / "labels"
    _write_label(r2 / "d01_0001.txt", "0 0.5 0.5 0.2 0.2\n")
    _write_label(r2 / "d05_0001.txt", "9 0.5 0.5 0.2 0.2\n")
    _write_label(r3 / "d01_0001.txt", "1 0.5 0.5 0.2 0.2\n")
    out = root / "yolo_r3"
    _r3_argv(monkeypatch, root, out, r2, r3)
    with pytest.raises(SystemExit, match="두 폴더에 있습니다: d01_0001"):
        auto_label.main()
    assert not out.exists()


# --- 검수 결과 → YOLO 라벨 (import) ---------------------------------------------------


def _annotation(
    ann_id: int,
    dets: list[Detection],
    *,
    cancelled: bool = False,
    lead_time: float = 30.0,
) -> dict:
    """사람이 제출한 annotation 하나(200×100 사진). 사람이 그린 박스에는 score 가 없다."""
    task = auto_label.to_label_studio("frames/x.jpg", dets, 200, 100, "human")
    result = task["predictions"][0]["result"]
    for item in result:
        del item["score"]
    return {"id": ann_id, "was_cancelled": cancelled, "lead_time": lead_time, "result": result}


def _export_task(stem: str, annotations: list[dict], predictions: list[Detection] = ()) -> dict:
    video = stem.split("_")[0]
    task = auto_label.to_label_studio(
        f"frames/{video}/{stem}.jpg", list(predictions), 200, 100, "r1+yoloe"
    )
    return {**task, "annotations": annotations}


def _import_argv(monkeypatch, source: str, path: Path, out: Path, *extra: str) -> None:
    _set_argv(monkeypatch, "import", source, str(path), "--out", str(out), *extra)


def test_import_export_uses_last_submitted_annotation(monkeypatch, tmp_path, capsys):
    """취소(Skip)한 것은 건너뛰고, 여럿이면 마지막(번호가 가장 큰) 것. 빈 제출은 빈 파일,
    검수 안 한 작업은 파일을 만들지 않는다(build 에서 빠지게)."""
    tasks = [
        _export_task(
            "d01_0001",
            [
                _annotation(1, [_det("bottle", (0, 0, 100, 50))], lead_time=20.0),
                _annotation(3, [_det("book", (0, 0, 100, 50))], cancelled=True),
                _annotation(2, [_det("cup", (0, 0, 100, 50))], lead_time=30.0),
            ],
        ),
        _export_task("d05_0002", [_annotation(1, [], lead_time=90.0)]),
        _export_task("d01_0002", []),
        _export_task("d02_0002", [_annotation(1, [_det("cup", (0, 0, 50, 50))], cancelled=True)]),
    ]
    export = tmp_path / "export.json"
    export.write_text(json.dumps(tasks), encoding="utf-8")
    out = tmp_path / "review" / "labels"
    _import_argv(monkeypatch, "--export", export, out)
    auto_label.main()

    assert sorted(p.name for p in out.iterdir()) == ["d01_0001.txt", "d05_0002.txt"]
    assert (out / "d01_0001.txt").read_text(encoding="utf-8") == (
        "1 0.250000 0.250000 0.500000 0.500000\n"
    )
    assert (out / "d05_0002.txt").read_text(encoding="utf-8") == ""
    printed = capsys.readouterr().out
    assert "라벨 2장" in printed
    assert "영상별: d01 1장 · d05 1장" in printed
    assert "검수 안 한 작업 2장" in printed
    assert "검수 시간: 합계 2분 · 사진당 중앙값 60초" in printed


def test_import_export_stops_on_name_outside_classes(monkeypatch, tmp_path):
    """사람이 고른 이름이 10종 밖이면 Label Studio 에 다른 설정이 올라간 것이라 멈춘다."""
    tasks = [_export_task("d01_0001", [_annotation(1, [_det("remote", (0, 0, 100, 50))])])]
    export = tmp_path / "export.json"
    export.write_text(json.dumps(tasks), encoding="utf-8")
    out = tmp_path / "labels"
    _import_argv(monkeypatch, "--export", export, out)
    with pytest.raises(SystemExit, match="10종 밖"):
        auto_label.main()
    assert not out.exists()


def test_import_predictions_drops_remote_and_counts_it(monkeypatch, tmp_path, capsys):
    """대조군(R1 초안): 원래 있던 remote 초안은 멈추지 않고 빼고 센다."""
    drafts = [
        _export_task(
            "d01_0001", [], [_det("remote", (0, 0, 60, 50)), _det("bottle", (100, 0, 200, 50))]
        )
    ]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(drafts), encoding="utf-8")
    out = tmp_path / "auto_labels"
    _import_argv(monkeypatch, "--predictions", path, out)
    auto_label.main()

    assert (out / "d01_0001.txt").read_text(encoding="utf-8") == (
        "0 0.750000 0.250000 0.500000 0.500000\n"
    )
    assert "10종 밖 remote 1개" in capsys.readouterr().out


def test_import_drops_boxes_shorter_than_min_side(monkeypatch, tmp_path, capsys):
    """짧은 변 40px 은 남기고 39px 은 뺀다. 사람 규칙과 자동 라벨에 같은 기준을 쓴다."""
    drafts = [
        _export_task(
            "d01_0001", [], [_det("bottle", (0, 0, 40, 50)), _det("cup", (100, 0, 139, 50))]
        )
    ]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(drafts), encoding="utf-8")
    out = tmp_path / "auto_labels"
    _import_argv(monkeypatch, "--predictions", path, out)
    auto_label.main()

    lines = (out / "d01_0001.txt").read_text(encoding="utf-8").splitlines()
    assert [line.split()[0] for line in lines] == ["0"]
    assert "짧은 변 40px 미만 1개" in capsys.readouterr().out


def test_import_like_keeps_only_photos_labeled_in_that_folder(monkeypatch, tmp_path):
    """대조군은 사람이 검수한 사진과 같은 사진만. 사진 수가 다르면 라벨 말고 양도 달라진다."""
    like = tmp_path / "review" / "labels"
    _write_label(like / "d01_0001.txt", "")
    drafts = [
        _export_task("d01_0001", [], [_det("bottle", (0, 0, 100, 50))]),
        _export_task("d01_0002", [], [_det("bottle", (0, 0, 100, 50))]),
    ]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(drafts), encoding="utf-8")
    out = tmp_path / "review" / "auto_labels"
    _import_argv(monkeypatch, "--predictions", path, out, "--like", str(like))
    auto_label.main()
    assert [p.name for p in out.iterdir()] == ["d01_0001.txt"]


def test_import_refuses_existing_out(monkeypatch, tmp_path):
    out = tmp_path / "labels"
    out.mkdir()
    path = tmp_path / "tasks.json"
    path.write_text("[]", encoding="utf-8")
    _import_argv(monkeypatch, "--predictions", path, out)
    with pytest.raises(SystemExit, match="이미 있습니다"):
        auto_label.main()


def test_import_stops_when_same_photo_was_reviewed_twice(monkeypatch, tmp_path):
    """가져오기를 두 번 해서 같은 사진의 작업이 둘이면, 어느 쪽을 쓸지 몰라 멈춘다."""
    tasks = [
        _export_task("d01_0001", [_annotation(1, [_det("cup", (0, 0, 100, 50))])]),
        _export_task("d01_0001", [_annotation(2, [])]),
    ]
    export = tmp_path / "export.json"
    export.write_text(json.dumps(tasks), encoding="utf-8")
    out = tmp_path / "labels"
    _import_argv(monkeypatch, "--export", export, out)
    with pytest.raises(SystemExit, match="d01_0001"):
        auto_label.main()
    assert not out.exists()


def test_import_without_any_submitted_photo_stops(monkeypatch, tmp_path):
    export = tmp_path / "export.json"
    export.write_text(json.dumps([_export_task("d01_0001", [])]), encoding="utf-8")
    _import_argv(monkeypatch, "--export", export, tmp_path / "labels")
    with pytest.raises(SystemExit, match="쓸 라벨이 없습니다"):
        auto_label.main()


@pytest.mark.parametrize(
    "args",
    [
        ["import", "--out", "x"],
        ["import", "--export", "a.json", "--predictions", "b.json", "--out", "x"],
        ["import", "--export", "a.json"],
    ],
)
def test_parse_args_import_needs_exactly_one_source_and_out(monkeypatch, args):
    """검수 결과(--export)와 R1 초안(--predictions) 중 하나만, 그리고 --out 은 꼭."""
    _set_argv(monkeypatch, *args)
    with pytest.raises(SystemExit):
        auto_label.parse_args()
