"""물체 후보 검출 (기본은 사전학습 COCO yolov8n, ``.env`` 로 학습한 가중치).  담당: 최홍묵

책임 지표: 후보 재현율 (진열대 식품 장면은 따로)

"저거"의 후보가 될 물건(컵·병·가방 등)은 COCO 에 이미 있어서 사전학습 YOLO 에서 시작한다.
가중치는 ``config.DETECTOR_WEIGHTS``(``JARVISEO_DETECTOR_WEIGHTS``) 하나로 정해서,
실행 루프 · 웹 · 평가 도구가 같은 설정을 따른다.
손은 여기서 보지 않는다. 손가락 방향(v2)은 MediaPipe ``HandLandmarker`` 관절로 구하고,
손끝 YOLO 는 학습하지 않는다(10/14 결정). 손끝 박스 하나로는 방향을 구할 수 없다.

파인튜닝은 측정으로 필요가 확인될 때만 한다(10/14). 안경 시점에서 후보 재현율이 50% 라
COCO 9종 + ``product``(포장 식품) 10종(``FINETUNE_CLASSES``)을 안경 카메라 영상으로 학습한다(#167).
R1 은 ``remote`` 를 넣은 11종으로 학습했다. 10종 밖 물건은 학습한 가중치로는 안 잡힌다. 지킬 것:

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
    "MAX_AREA_RATIO",
    "MIN_AREA_RATIO",
    "ObjectDetector",
    "filter_detections",
]

# 안경 시점에서는 사용자 손·팔이 person 으로 잡힌다. 후보에 두면 가운데 단서에서 1위가 되기 쉽다.
EXCLUDED_LABELS = frozenset({"person"})
# 화면 넓이의 0.2% (1080p 에서 약 64×64). 이보다 작으면 크롭해도 VLM·OCR 이 못 읽고,
# 후보만 늘어 margin 을 깎는다. 안경 카메라 사진으로 재기 전까지는 출발점이다.
MIN_AREA_RATIO = 0.002
# 파인튜닝으로 학습할 종류. 순서가 곧 클래스 번호(0~9)다. YOLO 라벨 파일에는 이름이 아니라
# 번호가 적혀서, 순서를 바꾸면 이미 만든 라벨이 전부 다른 물건이 된다. 새 종류는 맨 뒤에만 붙인다.
# R2 에서 이 규칙을 한 번 어기고 remote 를 뺐다(찍은 적 없는 종류). R2 라벨은 이름에서 새로 만들고,
# 옛 번호를 쓰는 R1 묶음(yolo_r1)은 보관만 한다.
# product = COCO 에 없는 포장 식품(봉지 · 컵라면 · 캔 · 병조림 · 파우치). 병 모양이면 bottle,
# 햇반은 bowl 이다(이름은 내용물이 아니라 모양).
FINETUNE_CLASSES = (
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
# 사진 넓이의 절반을 넘게 덮는 박스는 가리킨 물건이 아니라 배경(냉장고 · 노트북 헛 박스)으로 본다.
# 가운데 단서에서 늘 1위라 정답이 후보에 없을 때 그 박스를 확신했다(valid 확신 오답 10장 중 4장).
# valid 정답 박스는 최대 0.353, 헛 박스는 0.59~0.99. 채점의 "사진 절반" 기준과 같은 숫자다(#215).
MAX_AREA_RATIO = 0.5


def filter_detections(detections: list[Detection], frame_size: tuple[int, int]) -> list[Detection]:
    """후보가 될 수 없는 검출(사람 · 아주 작은 박스 · 사진 절반 넘는 박스)을 뺀다. 순서는 그대로."""
    width, height = frame_size
    min_area, max_area = width * height * MIN_AREA_RATIO, width * height * MAX_AREA_RATIO
    return [
        d
        for d in detections
        if d.label not in EXCLUDED_LABELS and min_area <= d.bbox.area <= max_area
    ]


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
        # agnostic_nms: 기본 NMS 는 이름이 같은 박스끼리만 지운다. 그래서 컵 하나에
        # bottle · product 가 같이 남고, 같은 자리라 점수 차가 0 이 돼 같은 물건을 둘로
        # 되물었다(웹 턴 58 · 63 · 66). 이름과 상관없이 IoU 0.7(기본값) 넘게 겹치면 확신 높은
        # 하나만 남긴다. 노트북 위 휴대폰처럼 큰 박스 안의 작은 물건은 IoU 가 낮아 남는다(#266).
        # tools/auto_label.py 의 초안 라벨도 같은 옵션을 쓴다.
        result = self.model.predict(
            image, conf=self.conf_threshold, agnostic_nms=True, verbose=False
        )[0]
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
