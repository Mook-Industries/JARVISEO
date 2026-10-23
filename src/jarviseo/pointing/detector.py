"""물체 후보 검출 (기본은 사전학습 COCO yolov8n, ``.env`` 로 학습한 가중치).  담당: 최홍묵

책임 지표: 후보 재현율 (진열대 식품 장면은 따로)

"저거"의 후보가 될 물건(컵·병·가방 등)은 COCO 에 이미 있어서 사전학습 YOLO 에서 시작한다.
가중치는 ``config.DETECTOR_WEIGHTS``(``JARVISEO_DETECTOR_WEIGHTS``) 하나로 정해서,
실행 루프 · 웹 · 평가 도구가 같은 설정을 따른다.
손은 여기서 보지 않는다. 손가락 방향(v2)은 MediaPipe ``HandLandmarker`` 관절로 구하고,
손끝 YOLO 는 학습하지 않는다(10/14 결정). 손끝 박스 하나로는 방향을 구할 수 없다.

파인튜닝은 측정으로 필요가 확인될 때만 한다(10/14). 안경 시점에서 후보 재현율이 50% 라
COCO 10종 + ``product``(포장 식품) 11종(``FINETUNE_CLASSES``)을 안경 카메라 영상으로 학습한다(#167).
11종 밖 물건은 학습한 가중치로는 안 잡힌다. 지킬 것:

1. **회차 비교는 같은 코드로 다시 잰다.** 사전학습 가중치(yolov8n.pt)는 파일로 남겨 두고,
   R1 · R2 가 나올 때마다 R0 까지 같은 코드로 다시 잰다. 날마다 적어 둔 숫자끼리 비교하면
   그사이 바뀐 채점 코드와 가중치 효과가 섞인다.
2. **test 셋은 최종 측정 전까지 열지 않는다.** 보면서 튜닝하면 나온 숫자가 실제 성능이
   아니게 된다.
3. 배포 포맷은 ONNX 로 통일한다. CoreML 은 맥 전용이라 팀원이 못 쓴다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jarviseo import config
from jarviseo.types import BBox, Detection

__all__ = [
    "EXCLUDED_LABELS",
    "FINETUNE_CLASSES",
    "MIN_AREA_RATIO",
    "ObjectDetector",
    "filter_detections",
]

# 안경 시점에서는 사용자 손·팔이 person 으로 잡힌다. 후보에 두면 가운데 단서에서 1위가 되기 쉽다.
EXCLUDED_LABELS = frozenset({"person"})
# 화면 넓이의 0.2% (1080p 에서 약 64×64). 이보다 작으면 크롭해도 VLM·OCR 이 못 읽고,
# 후보만 늘어 margin 을 깎는다. 안경 카메라 사진으로 재기 전까지는 출발점이다.
MIN_AREA_RATIO = 0.002
# 파인튜닝으로 학습할 종류. 순서가 곧 클래스 번호(0~10)다. YOLO 라벨 파일에는 이름이 아니라
# 번호가 적혀서, 순서를 바꾸면 이미 만든 라벨이 전부 다른 물건이 된다. 새 종류는 맨 뒤에만 붙인다.
# product = COCO 에 없는 포장 식품(봉지 · 컵라면 · 캔 · 햇반 · 병조림). 병 모양이면 bottle 이다.
FINETUNE_CLASSES = (
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


def filter_detections(detections: list[Detection], frame_size: tuple[int, int]) -> list[Detection]:
    """후보가 될 수 없는 검출(사람·아주 작은 박스)을 뺀다. 순서는 그대로 둔다."""
    width, height = frame_size
    min_area = width * height * MIN_AREA_RATIO
    return [d for d in detections if d.label not in EXCLUDED_LABELS and d.bbox.area >= min_area]


class ObjectDetector:
    """장면 안의 후보 물체들을 찾는다. 앱에서 하나만 만들어 돌려 쓴다(모델을 한 번만 읽으려고)."""

    def __init__(self, weights: Path | str | None = None, conf_threshold: float = 0.25) -> None:
        # CI 에는 ultralytics 가 없다. 모듈 위에서 import 하면 pointing 테스트가 전부 깨진다.
        from ultralytics import YOLO

        # 기본값 자리에 config 값을 바로 쓰면 함수를 정의할 때 한 번만 읽혀서,
        # 바꾼 설정이 안 따라온다.
        if weights is None:
            weights = config.DETECTOR_WEIGHTS
        self.model = YOLO(str(weights))
        # 실제로 읽은 파일. "yolov8n.pt" 처럼 이름만 줘도 ultralytics 가 찾은(받은) 위치가 들어
        # 있다. 평가 기록에 이 파일의 내용 지문을 남긴다(scripts/run_ablation.py).
        self.weights_path = Path(self.model.ckpt_path or weights)
        self.conf_threshold = conf_threshold

    def detect(self, image: np.ndarray) -> list[Detection]:
        """BGR 이미지에서 후보를 찾는다. 없으면 빈 리스트.

        ultralytics 는 numpy 배열을 OpenCV(BGR) 이미지로 읽으므로 RGB 로 바꾸지 않는다.
        """
        result = self.model.predict(image, conf=self.conf_threshold, verbose=False)[0]
        boxes = result.boxes
        # 텐서를 그대로 BBox 에 넣으면 DB 에 저장할 때 터진다. tolist() 로 float 로 바꾼다.
        detections = [
            Detection(label=result.names[int(cls)], confidence=conf, bbox=BBox(*xyxy))
            for xyxy, conf, cls in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
            )
        ]
        height, width = image.shape[:2]
        return filter_detections(detections, (width, height))
