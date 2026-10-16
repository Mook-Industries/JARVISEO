"""물체 후보 검출 — 사람·작은 박스를 빼고, YOLO 결과를 Detection 으로 바꾸는지.

CI 에는 ultralytics 가 없어서 실제 모델 대신 결과 모양만 같은 가짜를 끼운다.
"""

import numpy as np

from jarviseo.pointing.detector import MIN_AREA_RATIO, ObjectDetector, filter_detections
from jarviseo.pointing.labels import korean_name
from jarviseo.types import BBox, Detection

FRAME = (1920, 1080)


def _det(label: str, bbox: BBox, confidence: float = 0.9) -> Detection:
    return Detection(label=label, confidence=confidence, bbox=bbox)


class _List:
    """텐서 흉내. 실제 결과처럼 tolist() 로만 값을 꺼낸다."""

    def __init__(self, values) -> None:
        self.values = values

    def tolist(self):
        return self.values


class _Boxes:
    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy, self.conf, self.cls = _List(xyxy), _List(conf), _List(cls)


class _Result:
    names = {0: "person", 41: "cup", 39: "bottle"}

    def __init__(self, boxes: _Boxes) -> None:
        self.boxes = boxes


class _Model:
    """ultralytics YOLO 의 predict() 와 같은 모양. 받은 인자를 기록한다."""

    def __init__(self, result: _Result) -> None:
        self.result = result
        self.calls: list[dict] = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def _detector(result: _Result, conf_threshold: float = 0.25) -> ObjectDetector:
    # __init__ 은 ultralytics 를 import 하므로 건너뛰고 가짜 모델을 넣는다.
    detector = ObjectDetector.__new__(ObjectDetector)
    detector.model = _Model(result)
    detector.conf_threshold = conf_threshold
    return detector


# --- filter_detections ---------------------------------------------------------


def test_filter_drops_person():
    """사용자 손·팔이 person 으로 잡혀도 후보에 들어가지 않는다."""
    cup = _det("cup", BBox(800, 400, 1100, 700))
    kept = filter_detections([_det("person", BBox(0, 0, 900, 1080)), cup], FRAME)
    assert kept == [cup]


def test_filter_drops_tiny_box():
    """화면 넓이의 0.2% 보다 작은 박스는 뺀다."""
    tiny = _det("cup", BBox(0, 0, 10, 10))
    assert filter_detections([tiny], FRAME) == []


def test_filter_keeps_box_at_exact_min_area():
    """기준과 넓이가 딱 같으면 남긴다."""
    side = (FRAME[0] * FRAME[1] * MIN_AREA_RATIO) ** 0.5
    edge = _det("cup", BBox(0, 0, side, side))
    assert filter_detections([edge], FRAME) == [edge]


def test_filter_keeps_order():
    """걸러도 남은 후보의 순서는 그대로다."""
    a = _det("cup", BBox(0, 0, 300, 300))
    b = _det("bottle", BBox(500, 0, 800, 300))
    assert filter_detections([b, a], FRAME) == [b, a]


def test_filter_empty_input():
    """후보가 하나도 없으면 빈 리스트다."""
    assert filter_detections([], FRAME) == []


# --- ObjectDetector.detect -----------------------------------------------------


def test_detect_converts_result_to_detections():
    """라벨 번호를 이름으로, 좌표를 BBox 로 바꾸고 person 은 뺀다."""
    boxes = _Boxes(
        xyxy=[[0.0, 0.0, 900.0, 1080.0], [800.0, 400.0, 1100.0, 700.0]],
        conf=[0.95, 0.8],
        cls=[0.0, 41.0],  # 실제 결과처럼 클래스 번호가 float 로 온다
    )
    detector = _detector(_Result(boxes))

    found = detector.detect(np.zeros((1080, 1920, 3), np.uint8))

    assert found == [_det("cup", BBox(800.0, 400.0, 1100.0, 700.0), confidence=0.8)]


def test_detect_passes_conf_threshold():
    """신뢰도 문턱을 모델에 넘기고, 진행 출력을 끈다."""
    detector = _detector(_Result(_Boxes([], [], [])), conf_threshold=0.4)

    assert detector.detect(np.zeros((1080, 1920, 3), np.uint8)) == []
    assert detector.model.calls == [{"conf": 0.4, "verbose": False}]


def test_detect_uses_image_size_for_tiny_filter():
    """작은 박스 기준은 들어온 사진 크기로 잰다. 작은 사진에서는 같은 박스도 남는다."""
    boxes = _Boxes(xyxy=[[0.0, 0.0, 30.0, 30.0]], conf=[0.9], cls=[39.0])
    detector = _detector(_Result(boxes))

    assert detector.detect(np.zeros((1080, 1920, 3), np.uint8)) == []
    assert len(detector.detect(np.zeros((200, 200, 3), np.uint8))) == 1


# --- labels --------------------------------------------------------------------


def test_korean_name_known_label():
    """표에 있는 라벨은 한글 이름으로 바꾼다."""
    assert korean_name("cup") == "컵"


def test_korean_name_falls_back_to_label():
    """표에 없는 라벨(새로 학습한 클래스 등)은 그대로 돌려준다."""
    assert korean_name("product") == "product"
