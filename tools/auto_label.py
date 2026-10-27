"""자동 라벨 — 선생 모델로 학습 사진에 박스 초안을 붙이고, 회차별 학습 묶음을 만든다.  담당: 최홍묵

YOLO 파인튜닝(#167)의 둘째 단계. tools/extract_frames.py 가 뽑은 사진에 10종
(pointing/detector.py 의 FINETUNE_CLASSES) 박스를 붙여 두 가지를 만든다.
- YOLO 라벨: R1 은 이것을 그대로 학습했다(그때는 remote 를 넣은 11종)
- Label Studio 검수 파일: R2 는 사람이 고친 라벨로 학습한다

명령:
    compare  어느 선생이 우리 valid 사진에서 박스를 더 잘 치나 (YOLO-World vs yolo11x)
    label    고른 선생으로 라벨 초안을 만든다
    review   R1 초안에 YOLOE 식품 박스를 더해 사람 검수 작업을 만든다(R2).
             --weights 를 주면 새 영상의 사진에 우리가 학습한 가중치로 초안을 그린다(R3)
    import   검수 결과(Label Studio JSON)나 R1 초안을 사진별 YOLO 라벨로 바꾼다(R2 · 대조군)
    build    사진 + 라벨로 회차 폴더(yolo_r1 …)를 만든다. 학습 / 검증은 영상 단위로 나눈다.
             라벨 폴더를 여러 개 받는다(R3 = R2 검수본 + R3 검수본)

compare · label · review(R2) 는 CLIP 이 있는 곳(Colab, 또는 CLIP 을 깐 별도 가상환경)에서만 돈다.
ultralytics 는 YOLO-World · YOLOE 의 글귀를 처음 읽을 때 CLIP 을 git 에서 자동으로 설치해서,
팀 파이썬 환경에서 돌리면 승인 없이 패키지가 깔린다. 그래서 CLIP 이 없으면 미리 멈춘다.
review --weights(R3) 는 글귀를 안 읽어서 팀 환경에서 돈다. import · build 는 모델을 안 쓴다.

실행 (Colab, 레포 맨 위 폴더에서. 먼저 pip install git+https://github.com/ultralytics/CLIP.git):
    python tools/auto_label.py compare --pointing /content/drive/MyDrive/JARVISEO/pointing
    python tools/auto_label.py label --root /content/drive/MyDrive/JARVISEO/detector
    python tools/auto_label.py build --root /content/drive/MyDrive/JARVISEO/detector \\
        --labels /content/drive/MyDrive/JARVISEO/detector/auto/labels --val d05 \\
        --out /content/drive/MyDrive/JARVISEO/detector/yolo_r1

실행 (맥, R2 검수 작업. 가중치가 지금 폴더로 받아지므로 스크래치 폴더에서 · tools/README):
    <CLIP 가상환경>/bin/python <레포>/tools/auto_label.py review \\
        --root <레포>/data/datasets/detector

실행 (맥, R3 검수 작업. 레포 맨 위 폴더에서):
    python tools/auto_label.py review --weights data/models/detector_r2.pt \\
        --videos d07 d08 d09 --out data/datasets/detector/review_r3

저장 (--root 아래):
    auto/labels/d01_0001.txt   YOLO 라벨 (번호 중심x 중심y 너비 높이, 0~1)
    auto/tasks.json            Label Studio 로 가져올 검수 작업
    auto/label_config.xml      Label Studio 프로젝트 설정
    review/tasks.json          R2 검수 작업 (검수 순서 order · 메모 note 포함)
    review/label_config.xml    R2 Label Studio 프로젝트 설정
    review/labels/             검수 결과 YOLO 라벨 (import --export)
    review/auto_labels/        같은 사진의 R1 자동 라벨, 대조군 (import --predictions)
    review_r3/                 R3 검수 작업 · 결과(새 영상 d07~d09). 구조는 review/ 와 같다
    yolo_r1/images/{train,val}/ · yolo_r1/labels/{train,val}/ · yolo_r1/data.yaml   회차 하나
"""

import argparse
import csv
import importlib.util
import json
import re
import shutil
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from functools import partial
from itertools import zip_longest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from jarviseo import config
from jarviseo.pointing.detector import FINETUNE_CLASSES
from jarviseo.types import BBox, Detection

WORLD_WEIGHTS = "yolov8x-worldv2.pt"
COCO_WEIGHTS = "yolo11x.pt"
# 검출기 실행 때(ObjectDetector)와 같은 기준에서 출발한다. compare 로 0.1 과 같이 본다.
CONF = 0.25
# 두 선생의 결과를 합칠 때, 이만큼 겹치면 같은 물건으로 보고 신뢰도가 높은 쪽만 남긴다.
MERGE_IOU = 0.7
# run_ablation.py 의 IOU_MATCH 와 같은 기준. 정답 박스와 절반 넘게 겹치면 잡은 것이다.
MATCH_IOU = 0.5
# 영상 번호(extract_frames.py 의 이름 규칙). --val 에 이 모양만 받는다.
VIDEO_ID = re.compile(r"d\d{2}")
# 가장 많은 종류의 이만큼에 못 미치는 종류를 경고한다. 그 종류의 AP 만 바닥나는 원인이다.
RARE_RATIO = 0.1
# R2 검수 초안에 식품(product) 박스를 더하는 모델. R1 의 YOLO-World 는 product 를 7개밖에 못 붙였다.
YOLOE_WEIGHTS = "yoloe-26x-seg.pt"
# 같은 캔도 "can" 은 0.03, "tin can" 은 0.62 처럼 글귀에 따라 확신이 크게 달랐다(10/23 진단).
# 진단에서 정답 식품을 잡은 글귀들이다.
YOLOE_PRODUCT_PROMPTS = (
    "canned food",
    "box",
    "aluminum can",
    "packaged snack",
    "snack bag",
    "carton",
)
# 검수 초안에 식품 박스를 더할 신뢰도 기준.
# 헛 박스를 지우는 쪽이 놓친 물건을 찾아 그리는 쪽보다 빠르다.
REVIEW_CONF = 0.1
# 짧은 변이 이보다 작은 박스는 라벨하지 않는다(검수 규칙, 1280×720 사진의 px).
# 40×40 은 가리킴 후보의 최소 넓이(MIN_AREA_RATIO, 1280×720 에서 약 43×43)와 비슷하다.
MIN_SIDE = 40

# 우리 종류 → YOLO-World 에 줄 영어 글귀. "product" 한 단어로는 무엇을 찾을지 몰라서
# 구체적인 물건 이름으로 찾고 product 로 묶는다. 햇반은 bowl 이라 product 글귀에 넣지 않는다.
PROMPTS: dict[str, tuple[str, ...]] = {
    "bottle": ("bottle",),
    "cup": ("cup",),
    "bowl": ("bowl",),
    "laptop": ("laptop",),
    "mouse": ("mouse",),
    "keyboard": ("keyboard",),
    "cell phone": ("cell phone", "smartphone"),
    "book": ("book",),
    "tv": ("tv", "computer monitor"),
    "product": (
        "snack bag",
        "instant noodle cup",
        "can",
        "glass jar",
        "food package",
    ),
}
# 가리킴 정답표(labels.csv)의 이름 → 우리 종류. 없는 이름(glasses · watch 등 8장)은 10종 밖이다.
GT_TO_CLASS: dict[str, str] = {name: name for name in FINETUNE_CLASSES if name != "product"} | {
    "can": "product",
    "snack bag": "product",
    "jar": "product",
    "cup noodle": "product",
    "pouch": "product",
}

Teacher = Callable[[np.ndarray], list[Detection]]


def prompt_table(
    prompts: dict[str, tuple[str, ...]] | None = None,
) -> tuple[list[str], list[int]]:
    """(YOLO-World 에 줄 글귀 목록, 글귀 번호 → 우리 종류 번호).

    표에 없는 종류가 있거나 글귀가 없는 종류가 있으면 멈춘다. 그 종류는 라벨이 하나도 안 붙어
    학습에서 조용히 사라진다.
    """
    prompts = PROMPTS if prompts is None else prompts
    unknown = sorted(set(prompts) - set(FINETUNE_CLASSES))
    missing = [name for name in FINETUNE_CLASSES if not prompts.get(name)]
    if unknown or missing:
        raise ValueError(
            f"글귀 표가 종류 표와 다릅니다. 모르는 종류 {unknown}, 글귀 없음 {missing}"
        )
    phrases, owners = [], []
    for class_id, name in enumerate(FINETUNE_CLASSES):
        for phrase in prompts[name]:
            phrases.append(phrase)
            owners.append(class_id)
    return phrases, owners


def coco_to_class(names: dict[int, str]) -> dict[int, str]:
    """COCO 모델의 번호 → 우리 종류 이름. 10종 밖(person · chair 등)은 뺀다."""
    return {cls: name for cls, name in names.items() if name in FINETUNE_CLASSES}


def to_yolo_line(class_id: int, box: BBox, width: int, height: int) -> str:
    """YOLO 라벨 한 줄: "번호 중심x 중심y 너비 높이", 사진 크기로 나눠 0~1.

    사진 밖으로 나간 박스는 먼저 사진 안으로 자른다. 좌표를 0~1 로 나눈 뒤에 자르면
    중심과 크기가 어긋난다.
    """
    x1, y1 = max(0.0, box.x1), max(0.0, box.y1)
    x2, y2 = min(float(width), box.x2), min(float(height), box.y2)
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    w, h = (x2 - x1) / width, (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def merge_boxes(detections: list[Detection], iou: float = MERGE_IOU) -> list[Detection]:
    """신뢰도가 높은 순으로 남기고, 남긴 박스와 iou 이상 겹치면 버린다. 종류는 안 본다.

    같은 캔에 COCO 선생은 cup, World 는 product 를 붙이는 식의 겹침을 하나로 줄인다.
    """
    kept: list[Detection] = []
    for d in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(d.bbox.iou(k.bbox) < iou for k in kept):
            kept.append(d)
    return kept


def to_label_studio(
    file: str, detections: list[Detection], width: int, height: int, model_version: str
) -> dict:
    """Label Studio 작업 하나. 박스는 predictions 로 넣어 검수자가 고치기만 하게 한다.

    좌표는 왼쪽 위 기준 % 단위. from_name · to_name 은 label_config() 의 이름과 같아야 한다.
    이미지 주소는 Label Studio 의 로컬 파일 주소(--root 를 문서 루트로 연다).
    """
    result = [
        {
            "id": f"b{i}",
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "value": {
                "x": d.bbox.x1 / width * 100,
                "y": d.bbox.y1 / height * 100,
                "width": d.bbox.width / width * 100,
                "height": d.bbox.height / height * 100,
                "rotation": 0,
                "rectanglelabels": [d.label],
            },
            "score": d.confidence,
        }
        for i, d in enumerate(detections)
    ]
    return {
        "data": {"image": f"/data/local-files/?d={file}"},
        "predictions": [{"model_version": model_version, "result": result}],
    }


def from_label_studio(result: list[dict], width: int, height: int) -> list[Detection]:
    """to_label_studio 의 반대: Label Studio 박스(왼쪽 위 기준 %) → 픽셀 박스.

    사진 밖으로 나간 부분은 자른다(검수 화면에서 박스를 사진 끝까지 끌면 100% 를 조금 넘는다).
    사람이 그린 박스에는 신뢰도가 없어서 1.0 으로 둔다. 박스가 아닌 결과는 건너뛴다.
    """
    found = []
    for item in result:
        if item.get("type") != "rectanglelabels":
            continue
        v = item["value"]
        x1, y1 = max(0.0, v["x"] / 100 * width), max(0.0, v["y"] / 100 * height)
        x2 = min(float(width), (v["x"] + v["width"]) / 100 * width)
        y2 = min(float(height), (v["y"] + v["height"]) / 100 * height)
        if x2 <= x1 or y2 <= y1:
            continue
        score = item.get("score")
        found.append(
            Detection(
                label=v["rectanglelabels"][0],
                confidence=1.0 if score is None else score,
                bbox=BBox(x1, y1, x2, y2),
            )
        )
    return found


def task_file(task: dict) -> str:
    """작업의 사진 주소(/data/local-files/?d=frames/d01/d01_0001.jpg) → frames/d01/d01_0001.jpg."""
    (file,) = parse_qs(urlparse(task["data"]["image"]).query)["d"]
    return file


def prediction_result(task: dict) -> list[dict]:
    """작업에 딸린 초안(predictions) 박스. 초안이 없으면 빈 목록."""
    predictions = task.get("predictions") or []
    return predictions[0]["result"] if predictions else []


def label_config(with_note: bool = False) -> str:
    """Label Studio 프로젝트 설정. 종류는 FINETUNE_CLASSES 순서 그대로.

    밝기 · 대비 조절을 켠다. 밤 공원 영상(d04)은 밝기 중앙값이 9.9/255 라 그대로는 물건이 안 보이고,
    못 보고 지나친 물건은 라벨 없는 "배경"으로 학습된다. 화면에서만 바뀌고 사진 원본은 그대로다.
    확대 버튼을 켜고(편의점 d06 은 한 장에 상품이 수십 개다) 박스 회전은 끈다. YOLO 박스는
    돌릴 수 없어서, 돌린 박스는 변환할 때 틀어진다.
    with_note: 작업의 메모(note, 예: 뺀 remote 초안 수)를 사진 위에 보여 준다. review 작업만 있다.
    """
    labels = "\n".join(f'    <Label value="{name}"/>' for name in FINETUNE_CLASSES)
    note = '  <Header value="$note"/>\n' if with_note else ""
    return (
        "<View>\n"
        f"{note}"
        '  <Image name="image" value="$image" brightnessControl="true" contrastControl="true"'
        ' zoomControl="true"/>\n'
        '  <RectangleLabels name="label" toName="image" canRotate="false">\n'
        f"{labels}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )


def load_samples(pointing_dir: Path, sessions: list[int]) -> list[dict]:
    """가리킴 정답표에서 그 세션의 valid 사진만. test 줄은 읽지 않는다.

    run_ablation.py 의 load_samples 와 같은 거르기다.
    tools/ 는 scripts/ 를 불러오지 않아서 따로 둔다.
    """
    path = pointing_dir / "labels.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "file": row["file"],
            "gt": BBox(*(float(row[k]) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2"))),
            "gt_label": row["gt_label"],
        }
        for row in rows
        if row["split"] == "valid" and row["excluded"] != "1" and int(row["session"]) in sessions
    ]


def recall(samples: list[dict], detections: dict[str, list[Detection]]) -> dict:
    """선생 하나의 성적.

    - box_recall: 정답 박스와 겹치는 박스가 있는 사진 비율(종류 무관).
      가리킴 후보 재현율과 같은 정의
    - class_recall: 겹치는 박스의 종류까지 맞은 비율, 10종 안 정답만. 학습 라벨은 종류가 틀리면
      틀린 것을 배우므로 이것도 본다
    - boxes_per_photo: 사진당 박스 수. 재현율이 오른 만큼 헛 박스도 느는지 본다
    """
    box_hits = class_hits = class_total = product_hits = product_total = boxes = 0
    for s in samples:
        found = detections[s["file"]]
        boxes += len(found)
        hits = [d for d in found if d.bbox.iou(s["gt"]) >= MATCH_IOU]
        box_hits += bool(hits)
        target = GT_TO_CLASS.get(s["gt_label"])
        if target is None:
            continue
        correct = any(d.label == target for d in hits)
        class_total += 1
        class_hits += correct
        if target == "product":
            product_total += 1
            product_hits += correct
    n = len(samples)
    return {
        "photos": n,
        "box_recall": box_hits / n if n else None,
        "class_total": class_total,
        "class_recall": class_hits / class_total if class_total else None,
        "product_total": product_total,
        "product_recall": product_hits / product_total if product_total else None,
        "boxes_per_photo": boxes / n if n else None,
    }


def format_recall(teacher: str, r: dict) -> str:
    """compare 표의 한 줄. experiments.md 에 그대로 옮겨 적을 수 있게 표 모양으로."""

    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value * 100:.1f}%"

    per_photo = "-" if r["boxes_per_photo"] is None else f"{r['boxes_per_photo']:.1f}"
    return (
        f"| {teacher} | {pct(r['box_recall'])} ({r['photos']}장) | "
        f"{pct(r['class_recall'])} ({r['class_total']}장) | "
        f"{pct(r['product_recall'])} ({r['product_total']}장) | {per_photo} |"
    )


def read_image(path: Path) -> np.ndarray:
    """jpg 를 읽는다. cv2.imread 는 윈도우에서 경로에 한글이 있으면 None 을 돌려준다."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"사진을 읽지 못했습니다: {path}")
    return image


def run_model(model, image: np.ndarray, conf: float, to_class: dict[int, str]) -> list[Detection]:
    """모델 하나로 찾고, 모델 번호를 우리 종류 이름으로 바꾼다. to_class 에 없는 번호는 버린다.

    agnostic_nms: 같은 캔에 "can" 과 "glass jar" 가 같이 붙는 겹침을 종류와 상관없이 하나로 줄인다.
    """
    result = model.predict(image, conf=conf, agnostic_nms=True, verbose=False)[0]
    boxes = result.boxes
    found = []
    for xyxy, score, cls in zip(
        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
    ):
        name = to_class.get(int(cls))
        if name is not None:
            found.append(Detection(label=name, confidence=score, bbox=BBox(*xyxy)))
    return found


def make_teacher(name: str, conf: float) -> Teacher:
    """선생 모델. world = YOLO-World 10종, coco = yolo11x 9종(product 없음),
    mix = COCO 9종은 yolo11x, product 만 YOLO-World 로 찾아 합친다.

    가중치는 처음 부를 때 ultralytics 가 지금 폴더로 내려받는다(Colab).
    """
    from ultralytics import YOLO  # CI 에는 ultralytics 가 없다. 모듈 위에서 부르면 테스트가 깨진다

    if name == "world":
        phrases, owners = prompt_table()
        world = YOLO(WORLD_WEIGHTS)
        world.set_classes(phrases)
        to_class = {i: FINETUNE_CLASSES[c] for i, c in enumerate(owners)}
        return partial(run_model, world, conf=conf, to_class=to_class)

    coco = YOLO(COCO_WEIGHTS)
    coco_detect = partial(run_model, coco, conf=conf, to_class=coco_to_class(coco.names))
    if name == "coco":
        return coco_detect

    products = list(PROMPTS["product"])
    world = YOLO(WORLD_WEIGHTS)
    world.set_classes(products)
    product_detect = partial(
        run_model, world, conf=conf, to_class=dict.fromkeys(range(len(products)), "product")
    )
    return lambda image: merge_boxes(coco_detect(image) + product_detect(image))


def require_clip() -> None:
    """CLIP 이 없으면 모델을 부르기 전에 멈춘다.

    ultralytics 는 YOLO-World · YOLOE 의 글귀를 읽을 때 `import clip` 이 실패하면 git 에서 CLIP 을
    설치한다(ultralytics/nn/text_model.py 맨 위). 팀 conda 환경에서 그대로 돌리면 승인 없이 깔린다.
    """
    if importlib.util.find_spec("clip") is None:
        raise SystemExit(
            "CLIP 이 없어 멈춥니다. 팀 환경에 CLIP 이 자동 설치되지 않게 하려는 것입니다. "
            "Colab 은 먼저 pip install git+https://github.com/ultralytics/CLIP.git, "
            "맥은 CLIP 을 깐 별도 가상환경에서 돌립니다(tools/README)"
        )


def make_product_finder(conf: float) -> Teacher:
    """YOLOE 로 식품(product) 박스만 찾는다. 글귀가 여러 개라도 결과는 모두 product 다.

    가중치와 글귀 모델(MobileCLIP2)은 처음 부를 때 지금 폴더로 받아진다(합쳐 400MB 남짓).
    """
    from ultralytics import YOLOE  # CI 에는 ultralytics 가 없다

    model = YOLOE(YOLOE_WEIGHTS)
    model.set_classes(list(YOLOE_PRODUCT_PROMPTS))
    to_class = dict.fromkeys(range(len(YOLOE_PRODUCT_PROMPTS)), "product")
    return partial(run_model, model, conf=conf, to_class=to_class)


def weight_class_table(names: dict[int, str]) -> dict[int, str]:
    """가중치에 적힌 종류(번호 → 이름)가 10종 표와 똑같은지 보고 그대로 돌려준다.

    R1 가중치(remote 를 넣은 11종)를 넣으면 remote 초안이 섞이고, 사전학습 COCO 가중치를 넣으면
    product 가 빠진다. 어느 쪽이든 초안 기준이 회차마다 흔들려서 조용히 넘기지 않고 멈춘다.
    """
    if sorted(names.values()) != sorted(FINETUNE_CLASSES):
        extra = sorted(set(names.values()) - set(FINETUNE_CLASSES))
        lacking = sorted(set(FINETUNE_CLASSES) - set(names.values()))
        raise SystemExit(
            f"가중치의 종류가 10종 표와 다릅니다(더 있음 {extra[:5]} · 없음 {lacking}). "
            "R2 처럼 10종으로 학습한 가중치를 줍니다"
        )
    return dict(names)


def make_weight_teacher(weights: Path, conf: float) -> Teacher:
    """우리가 학습한 10종 가중치(R2 …)로 초안을 그린다(R3 검수 초안).

    지금 가장 나은 우리 모델이라 초안을 고칠 곳이 적고, 글귀를 안 읽어서 CLIP 이 필요 없다.
    """
    from ultralytics import YOLO  # CI 에는 ultralytics 가 없다

    model = YOLO(str(weights))
    return partial(run_model, model, conf=conf, to_class=weight_class_table(model.names))


def too_small(box: BBox, min_side: float = MIN_SIDE) -> bool:
    """짧은 변이 min_side 보다 작으면 True. 딱 min_side 인 박스는 남긴다."""
    return min(box.width, box.height) < min_side


def fill_gaps(
    base: list[Detection], extra: list[Detection], iou: float = MATCH_IOU
) -> list[Detection]:
    """extra 중 base 의 어느 박스와도 iou 미만으로 겹치는 것만 base 뒤에 붙인다.

    신뢰도로 겨루게 하지 않는다. YOLOE 의 "box" 글귀는 상자처럼 생긴 것을 다 잡아서,
    겨루면 키보드 · 노트북 초안이 product 로 덮인다. 빈 자리에만 더하면 R1 초안은 그대로 남아,
    대조군(R1 초안 그대로)과의 차이가 사람이 고친 것으로만 남는다.
    """
    added = [e for e in extra if all(e.bbox.iou(b.bbox) < iou for b in base)]
    return base + added


def _frame_number(stem: str) -> int:
    """d05_0007 → 7."""
    return int(stem.rsplit("_", 1)[1])


def review_order(stems: list[str], val_videos: set[str]) -> dict[str, int]:
    """검수 순서(1부터). ① 검증 영상의 짝수 번 → ② 학습 영상을 번갈아 → ③ 검증 영상의 홀수 번.

    검수는 3시간이라 학습 사진의 절반쯤만 본다. 영상 순서대로 보면 앞 영상 장면만 검수하게 되고,
    3초 간격이라 이웃 사진이 거의 같아서 영상 안에서도 짝수 번을 먼저 본다(d01 짝수 → d02 짝수 …).
    검증 사진(best.pt 를 고르는 데 쓴다)은 R2 와 대조군이 같이 쓰므로 절반을 맨 앞에서 끝낸다.
    """

    def even_first(names: list[str]) -> list[str]:
        names = sorted(names)
        return [n for n in names if _frame_number(n) % 2 == 0] + [
            n for n in names if _frame_number(n) % 2
        ]

    by_video: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        by_video[stem.split("_", 1)[0]].append(stem)
    val = sorted(stem for video in val_videos for stem in by_video.get(video, []))
    train = [even_first(by_video[v]) for v in sorted(by_video) if v not in val_videos]
    interleaved = [stem for group in zip_longest(*train) for stem in group if stem is not None]
    ordered = (
        [s for s in val if _frame_number(s) % 2 == 0]
        + interleaved
        + [s for s in val if _frame_number(s) % 2]
    )
    return {stem: i for i, stem in enumerate(ordered, start=1)}


def cmd_compare(args: argparse.Namespace) -> None:
    require_clip()
    samples = load_samples(args.pointing, args.sessions)
    if not samples:
        raise SystemExit(f"valid 사진이 없습니다: {args.pointing / 'labels.csv'}")
    sessions = " ".join(str(s) for s in args.sessions)
    print(f"valid {len(samples)}장 (세션 {sessions}) · conf {args.conf}")
    print("| 선생 | 박스 재현율 | 종류까지 맞음 | product | 사진당 박스 |")
    print("|---|---|---|---|---|")
    images = {s["file"]: read_image(args.pointing / "raw" / s["file"]) for s in samples}
    for name in ("world", "coco"):
        teacher = make_teacher(name, args.conf)
        detections = {file: teacher(image) for file, image in images.items()}
        print(format_recall(name, recall(samples, detections)))


def cmd_label(args: argparse.Namespace) -> None:
    require_clip()
    frames = sorted((args.root / "frames").glob("*/*.jpg"))
    if not frames:
        raise SystemExit(
            f"사진이 없습니다. extract_frames.py 를 먼저 돌립니다: {args.root / 'frames'}"
        )
    out = args.root / "auto"
    # R1 이 학습한 라벨을 덮어쓰면 R1 을 다시 만들 수 없다(회차 기록이 깨진다).
    if out.exists():
        raise SystemExit(f"이미 있습니다: {out} — 이름을 바꿔 보관한 뒤 다시 돌립니다")
    # 가중치를 내려받다 실패해도 빈 auto/ 가 남아 다음 실행을 막지 않게, 선생을 먼저 만든다.
    teacher = make_teacher(args.teacher, args.conf)
    (out / "labels").mkdir(parents=True)

    tasks, counts = [], Counter()
    for path in frames:
        image = read_image(path)
        height, width = image.shape[:2]
        found = teacher(image)
        lines = [
            to_yolo_line(FINETUNE_CLASSES.index(d.label), d.bbox, width, height) for d in found
        ]
        # 박스가 없는 사진도 빈 파일을 둔다. "물건 없는 사진"이라는 뜻이다.
        label_path = out / "labels" / f"{path.stem}.txt"
        label_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        file = path.relative_to(args.root).as_posix()
        tasks.append(to_label_studio(file, found, width, height, args.teacher))
        counts.update(d.label for d in found)

    (out / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    (out / "label_config.xml").write_text(label_config(), encoding="utf-8")
    print(
        f"사진 {len(frames)}장 · 박스 {sum(counts.values())}개 ({args.teacher}, conf {args.conf})"
    )
    for name in FINETUNE_CLASSES:
        print(f"  {name}: {counts[name]}")


def write_review(out: Path, tasks: list[dict]) -> None:
    """검수 작업과 Label Studio 설정을 쓴다. 가져온 순서가 곧 작업 번호라 검수 순서대로 정렬한다."""
    tasks.sort(key=lambda t: t["data"]["order"])
    out.mkdir(parents=True)
    (out / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    (out / "label_config.xml").write_text(label_config(with_note=True), encoding="utf-8")


def cmd_review(args: argparse.Namespace) -> None:
    out = args.out or args.root / "review"
    # 검수 도중 덮어쓰면 Label Studio 로 이미 가져간 작업과 어긋난다.
    if out.exists():
        raise SystemExit(
            f"이미 있습니다: {out} — 검수 중인 작업과 어긋나지 않게 보관한 뒤 다시 돌립니다"
        )
    if args.weights is not None:
        review_with_weights(args, out)
        return
    require_clip()
    drafts_path = args.drafts or args.root / "auto" / "tasks.json"
    if not drafts_path.exists():
        raise SystemExit(f"R1 초안이 없습니다: {drafts_path} — Drive 의 detector/auto/tasks.json")
    drafts = json.loads(drafts_path.read_text(encoding="utf-8"))
    files = [task_file(task) for task in drafts]
    missing = [file for file in files if not (args.root / file).exists()]
    if missing:
        raise SystemExit(f"초안의 사진 {len(missing)}장이 없습니다(예: {missing[0]}). --root 확인")
    find_products = make_product_finder(args.conf)
    order = review_order([Path(file).stem for file in files], set(args.val))

    # 먼저 다 만든 뒤에 쓴다. 중간에 멈추면 반쯤 만든 review/ 가 다음 실행을 막는다.
    tasks, counts = [], Counter()
    added = small_draft = small_product = 0
    outside: dict[str, Counter] = {}
    for task, file in zip(drafts, files, strict=True):
        image = read_image(args.root / file)
        height, width = image.shape[:2]
        draft = from_label_studio(prediction_result(task), width, height)
        # R1 은 remote 를 넣은 11종이었다. 10종 밖 초안은 빼고, 그 자리를 다시 보라고 메모를 남긴다.
        dropped = Counter(d.label for d in draft if d.label not in FINETUNE_CLASSES)
        kept = [d for d in draft if d.label in FINETUNE_CLASSES]
        products = find_products(image)
        small_draft += sum(too_small(d.bbox) for d in kept)
        small_product += sum(too_small(d.bbox) for d in products)
        kept = [d for d in kept if not too_small(d.bbox)]
        products = [d for d in products if not too_small(d.bbox)]
        boxes = fill_gaps(kept, products)
        added += len(boxes) - len(kept)
        counts.update(d.label for d in boxes)

        stem = Path(file).stem
        note = ""
        if dropped:
            outside[stem] = dropped
            names = ", ".join(f"{name} {n}개" for name, n in sorted(dropped.items()))
            note = f"R1 초안에서 10종 밖 {names}를 뺐습니다 — 그 자리 물건을 다시 보세요"
        new = to_label_studio(file, boxes, width, height, "r1+yoloe")
        new["data"] |= {"video": stem.split("_", 1)[0], "order": order[stem], "note": note}
        tasks.append(new)

    write_review(out, tasks)

    val_names = " ".join(sorted(args.val))
    val_count = sum(t["data"]["video"] in args.val for t in tasks)
    print(
        f"검수 작업 {len(tasks)}장 (검증 {val_names} {val_count}장) · "
        f"초안 박스 {sum(counts.values())}개 → {out}"
    )
    for name in FINETUNE_CLASSES:
        print(f"  {name}: {counts[name]}")
    print(f"YOLOE 식품 박스를 빈 자리에 더함: {added}개 (conf {args.conf})")
    removed = sum(outside.values(), Counter())
    names = ", ".join(f"{name} {n}개" for name, n in sorted(removed.items())) or "없음"
    print(f"10종 밖이라 뺀 R1 초안: {names} · 사진 {len(outside)}장 {' '.join(sorted(outside))}")
    print(
        f"짧은 변 {MIN_SIDE}px 미만이라 뺀 박스: R1 초안 {small_draft}개 · YOLOE {small_product}개"
    )


def review_with_weights(args: argparse.Namespace, out: Path) -> None:
    """R3: 새 영상의 사진에 우리가 학습한 가중치로 초안을 그려 검수 작업을 만든다.

    새 사진에는 R1 초안이 없어서, 사진 목록은 frames/<영상>/ 에서 --videos 에 든 영상만 읽는다.
    YOLOE 는 부르지 않는다. 초안 방법이 R2 와 달라도 최종 라벨은 사람이 고치므로,
    회차 비교에서 바뀌는 것은 "새 사진을 더한 것" 하나로 남는다.
    """
    frames = args.root / "frames"
    absent = [video for video in args.videos if not (frames / video).is_dir()]
    if absent:
        raise SystemExit(f"사진 폴더가 없습니다: {absent} — extract_frames.py 를 먼저 돌립니다")
    if not args.weights.exists():
        raise SystemExit(f"가중치가 없습니다: {args.weights}")
    files = sorted(
        path.relative_to(args.root).as_posix()
        for video in args.videos
        for path in (frames / video).glob("*.jpg")
    )
    if not files:
        raise SystemExit(f"사진이 없습니다: {frames} 의 {' '.join(args.videos)}")
    teacher = make_weight_teacher(args.weights, args.conf)
    order = review_order([Path(file).stem for file in files], set(args.val))

    tasks, counts, small = [], Counter(), 0
    for file in files:
        image = read_image(args.root / file)
        height, width = image.shape[:2]
        found = teacher(image)
        small += sum(too_small(d.bbox) for d in found)
        boxes = [d for d in found if not too_small(d.bbox)]
        counts.update(d.label for d in boxes)
        stem = Path(file).stem
        # 모델 이름에 가중치 파일 이름을 남겨, 어느 가중치로 그린 초안인지 작업에서 보이게 한다.
        new = to_label_studio(file, boxes, width, height, args.weights.stem)
        new["data"] |= {"video": stem.split("_", 1)[0], "order": order[stem], "note": ""}
        tasks.append(new)
    write_review(out, tasks)

    per_video = Counter(t["data"]["video"] for t in tasks)
    videos = " · ".join(f"{video} {n}장" for video, n in sorted(per_video.items()))
    print(f"검수 작업 {len(tasks)}장 ({videos}) · 초안 박스 {sum(counts.values())}개 → {out}")
    for name in FINETUNE_CLASSES:
        print(f"  {name}: {counts[name]}")
    print(
        f"초안: {args.weights.name} (conf {args.conf}) · "
        f"짧은 변 {MIN_SIDE}px 미만이라 뺀 박스 {small}개"
    )


def chosen_annotation(task: dict) -> dict | None:
    """검수 결과로 쓸 annotation. 취소(Skip)하지 않은 것 중 마지막(번호가 가장 큰) 것.

    없으면 None — 검수하지 않은 사진이다. 라벨 파일을 만들지 않아야 build 가 그 사진을 뺀다.
    빈 라벨로 두면 "물건 없음"으로 학습된다.
    """
    done = [a for a in task.get("annotations") or [] if not a.get("was_cancelled")]
    return max(done, key=lambda a: a.get("id", 0)) if done else None


def image_size(result: list[dict]) -> tuple[int, int] | None:
    """Label Studio 박스에 적힌 사진 크기(너비, 높이). 박스가 없으면 None."""
    for item in result:
        if "original_width" in item:
            return item["original_width"], item["original_height"]
    return None


def cmd_import(args: argparse.Namespace) -> None:
    # 회차 라벨을 덮어쓰면 그 회차를 같은 라벨로 다시 학습할 수 없다.
    if args.out.exists():
        raise SystemExit(f"이미 있습니다: {args.out} — 회차마다 새 폴더를 씁니다")
    source = args.export or args.predictions
    tasks = json.loads(source.read_text(encoding="utf-8"))

    # 먼저 다 읽고 검사한 뒤에 쓴다. 중간에 멈추면 반쯤 만든 라벨 폴더가 남는다.
    labels: dict[str, list[str]] = {}
    lead_times: dict[str, float] = {}
    outside, counts = Counter(), Counter()
    unreviewed = small = 0
    for task in tasks:
        stem = Path(task_file(task)).stem
        # 대조군은 사람이 검수한 사진과 같은 사진만 쓴다. 사진 수가 다르면 라벨 말고 양도 달라진다.
        if args.like is not None and not (args.like / f"{stem}.txt").exists():
            continue
        if args.export is not None:
            annotation = chosen_annotation(task)
            if annotation is None:
                unreviewed += 1
                continue
            result = annotation.get("result") or []
            lead_times[stem] = annotation.get("lead_time") or 0.0
        else:
            result = prediction_result(task)
        if stem in labels:
            raise SystemExit(
                f"같은 사진의 작업이 둘입니다: {stem} — 가져오기를 두 번 했는지 봅니다"
            )
        size = image_size(result)
        boxes = from_label_studio(result, *size) if size else []
        unknown = Counter(d.label for d in boxes if d.label not in FINETUNE_CLASSES)
        # 사람이 고른 이름이 10종 밖이면 Label Studio 에 다른 설정이 올라간 것이다. R1 초안의
        # remote 는 원래 있던 것이라 빼고 센다.
        if unknown and args.export is not None:
            raise SystemExit(
                f"10종 밖 이름이 있습니다: {dict(unknown)} ({stem}) — "
                "Label Studio 라벨 설정이 review/label_config.xml 과 같은지 봅니다"
            )
        outside += unknown
        kept = [d for d in boxes if d.label in FINETUNE_CLASSES]
        small += sum(too_small(d.bbox, args.min_side) for d in kept)
        kept = [d for d in kept if not too_small(d.bbox, args.min_side)]
        counts.update(d.label for d in kept)
        labels[stem] = [to_yolo_line(FINETUNE_CLASSES.index(d.label), d.bbox, *size) for d in kept]
    if not labels:
        raise SystemExit(
            "쓸 라벨이 없습니다 — 제출(Submit)한 사진이 없거나 --like 폴더가 비었습니다"
        )

    args.out.mkdir(parents=True)
    for stem, lines in labels.items():
        (args.out / f"{stem}.txt").write_text("".join(f"{x}\n" for x in lines), encoding="utf-8")

    videos = Counter(stem.split("_", 1)[0] for stem in labels)
    print(f"라벨 {len(labels)}장 → {args.out} ({source.name})")
    print("  영상별: " + " · ".join(f"{v} {n}장" for v, n in sorted(videos.items())))
    for name in FINETUNE_CLASSES:
        print(f"  {name}: {counts[name]}")
    names = ", ".join(f"{name} {n}개" for name, n in sorted(outside.items())) or "없음"
    print(f"뺀 박스: 10종 밖 {names} · 짧은 변 {args.min_side}px 미만 {small}개")
    if args.export is not None:
        print(f"검수 안 한 작업 {unreviewed}장 — 라벨 파일을 만들지 않아 학습에서 빠진다")
        total = sum(lead_times.values())
        median = statistics.median(lead_times.values())
        print(f"검수 시간: 합계 {total / 60:.0f}분 · 사진당 중앙값 {median:.0f}초")
        by_video: dict[str, float] = defaultdict(float)
        for stem, seconds in lead_times.items():
            by_video[stem.split("_", 1)[0]] += seconds
        print("  영상별: " + " · ".join(f"{v} {s / 60:.0f}분" for v, s in sorted(by_video.items())))


def split_of(stem: str, val_videos: set[str]) -> str:
    """사진 이름(d05_0007)의 영상 번호가 검증 영상이면 val, 아니면 train.

    사진 단위로 나누지 않는다. 같은 영상에서 1초 차이로 뽑은 사진은 거의 같아서,
    학습과 검증에 나뉘어 들어가면 검증 점수가 부풀려진다.
    """
    return "val" if stem.split("_", 1)[0] in val_videos else "train"


def class_names_file(labels_dir: Path) -> Path | None:
    """Label Studio 가 YOLO 로 내보낸 classes.txt. 라벨 폴더나 그 위 폴더에 있다.

    auto_label.py label 이 만든 라벨에는 없다(이미 FINETUNE_CLASSES 번호다).
    """
    for path in (labels_dir / "classes.txt", labels_dir.parent / "classes.txt"):
        if path.exists():
            return path
    return None


def remap_table(names: list[str]) -> dict[int, int]:
    """classes.txt 의 번호 → 우리 번호를 이름으로 맞춘다.

    Label Studio 가 내보낸 번호 순서가 FINETUNE_CLASSES 순서와 같다는 보장이 없다.
    맞추지 않으면 bottle 이 cup 으로 학습될 수 있다. 우리 표에 없는 이름이면 멈춘다.
    """
    unknown = [name for name in names if name not in FINETUNE_CLASSES]
    if unknown:
        raise ValueError(f"classes.txt 에 우리 종류 표에 없는 이름이 있습니다: {unknown}")
    return {i: FINETUNE_CLASSES.index(name) for i, name in enumerate(names)}


def remap_line(line: str, table: dict[int, int] | None) -> str:
    """라벨 한 줄의 종류 번호를 우리 번호로. table 이 없으면 번호 그대로(범위만 확인)."""
    cls, rest = line.split(maxsplit=1)
    new = int(cls) if table is None else table.get(int(cls), -1)
    if not 0 <= new < len(FINETUNE_CLASSES):
        raise ValueError(f"종류 번호가 0~{len(FINETUNE_CLASSES) - 1} 밖입니다: {line}")
    return f"{new} {rest}"


def data_yaml() -> str:
    """ultralytics 데이터 설정.

    path 키는 넣지 않는다. ultralytics 8.4(data/utils.py)는 path 가 없으면 이 파일이 있는 폴더를
    기준으로 train · val 을 푼다. path: . 같은 상대 경로는 지금 작업 폴더 기준이라 맥과 Colab 에서
    서로 다른 곳을 가리킨다.
    """
    names = "\n".join(f'  {i}: "{name}"' for i, name in enumerate(FINETUNE_CLASSES))
    return f"train: images/train\nval: images/val\nnames:\n{names}\n"


def rare_classes(counts: Counter) -> list[str]:
    """박스가 가장 많은 종류의 RARE_RATIO 에 못 미치는 종류(0개 포함).

    "전체의 10%"로 재면 10종의 평균이 10% 라 거의 모든 종류가 걸린다.
    그래서 가장 많은 종류와 견준다.
    """
    top = max(counts.values(), default=0)
    return [name for name in FINETUNE_CLASSES if counts[name] < top * RARE_RATIO]


def cmd_build(args: argparse.Namespace) -> None:
    # 회차 폴더를 덮어쓰면 그 회차를 다시 학습할 수 없다.
    if args.out.exists():
        raise SystemExit(
            f"이미 있습니다: {args.out} — 회차마다 새 폴더(yolo_r1, yolo_r2 …)를 씁니다"
        )
    frames = sorted((args.root / "frames").glob("*/*.jpg"))
    if not frames:
        raise SystemExit(
            f"사진이 없습니다. extract_frames.py 를 먼저 돌립니다: {args.root / 'frames'}"
        )
    val_videos = set(args.val)
    # 대조군은 학습 라벨만 다르고 검증 라벨(best.pt 를 고르는 기준)은 R2 와 같은 사람 라벨을 쓴다.
    # R3 는 라벨 폴더가 둘이다(R2 검수본 + R3 검수본). 사진마다 라벨이 있는 폴더 하나에서 읽는다.
    folders = {"train": args.labels, "val": [args.val_labels] if args.val_labels else args.labels}
    names_files = {
        folder: class_names_file(folder) for split in folders.values() for folder in split
    }

    # 먼저 다 읽고 검사한 뒤에 쓴다. 중간에 멈추면 반쯤 만든 회차 폴더가 다음 실행을 막는다.
    # 라벨 파일이 있는 사진만 넣는다. 검수하지 않은 사진을 "물건 없음"으로 가르치지 않으려고.
    # 빈 라벨 파일은 물건이 없는 사진이라 넣는다.
    items, missing, used = [], 0, Counter()
    try:
        # 이름표 다시 맞추기는 폴더마다 따로 한다(R1 처럼 옛 11종 번호로 쓴 폴더가 섞일 수 있다).
        tables: dict[Path, dict[int, int] | None] = {}
        for folder, names_file in names_files.items():
            tables[folder] = None
            if names_file is not None:
                names = names_file.read_text(encoding="utf-8").splitlines()
                tables[folder] = remap_table([name.strip() for name in names if name.strip()])
        for path in frames:
            split = split_of(path.stem, val_videos)
            found = [f for f in folders[split] if (f / f"{path.stem}.txt").exists()]
            if not found:
                missing += 1
                continue
            # 두 폴더에 다 있으면 어느 라벨이 맞는지 모른다. 한쪽을 조용히 고르면 R2 라벨이 바뀌어
            # "새 사진을 더한 것 하나만 다르다"는 회차 비교가 깨진다.
            if len(found) > 1:
                raise ValueError(f"같은 사진의 라벨이 두 폴더에 있습니다: {path.stem} — {found}")
            (folder,) = found
            raw = (folder / f"{path.stem}.txt").read_text(encoding="utf-8").splitlines()
            lines = [remap_line(line, tables[folder]) for line in raw if line.strip()]
            items.append((path, split, lines))
            used[folder] += 1
    except ValueError as e:
        raise SystemExit(str(e)) from None
    splits = {split for _, split, _ in items}
    if "val" not in splits:
        raise SystemExit(f"검증 영상 {sorted(val_videos)} 의 라벨 붙은 사진이 없습니다")
    if "train" not in splits:
        raise SystemExit("학습 사진이 없습니다. 모든 영상을 검증으로 뺐습니다")

    photos, counts = Counter(), {"train": Counter(), "val": Counter()}
    for path, split, lines in items:
        image_dir, label_dir = args.out / "images" / split, args.out / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        # 링크가 아니라 복사한다. 심볼릭 링크는 윈도우와 Drive 에서 깨진다.
        shutil.copy2(path, image_dir / path.name)
        (label_dir / f"{path.stem}.txt").write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8"
        )
        photos[split] += 1
        counts[split].update(FINETUNE_CLASSES[int(line.split()[0])] for line in lines)
    (args.out / "data.yaml").write_text(data_yaml(), encoding="utf-8")

    val_names = " ".join(sorted(val_videos))
    print(
        f"학습 {photos['train']}장 · 검증 {photos['val']}장 (검증 영상 {val_names}) · "
        f"라벨 없어 뺀 사진 {missing}장"
    )
    if args.val_labels is not None:
        print(f"검증 라벨은 {args.val_labels} 에서 읽음")
    if len(used) > 1:
        print("라벨을 가져온 폴더: " + " · ".join(f"{f} {n}장" for f, n in sorted(used.items())))
    for names_file in sorted({f for f in names_files.values() if f is not None}):
        print(f"종류 번호를 이름으로 다시 맞춤: {names_file}")
    print("종류 | 학습 박스 | 검증 박스")
    for name in FINETUNE_CLASSES:
        print(f"  {name} | {counts['train'][name]} | {counts['val'][name]}")
    rare = rare_classes(counts["train"] + counts["val"])
    if rare:
        print(f"경고: 박스가 가장 많은 종류의 10% 에 못 미치는 종류 — {', '.join(rare)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="자동 라벨 — 선생 모델로 박스 초안")
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="선생 비교 (valid 사진 재현율)")
    compare.add_argument(
        "--pointing",
        type=Path,
        default=config.DATASETS_DIR / "pointing",
        help="가리킴 평가셋 폴더 (labels.csv · raw/). Colab 에는 s01 · s02 만 올린다",
    )
    compare.add_argument("--sessions", type=int, nargs="+", default=[1, 2], help="valid 세션")
    compare.add_argument("--conf", type=float, default=CONF, help="신뢰도 기준")

    label = sub.add_parser("label", help="라벨 초안 만들기")
    label.add_argument(
        "--root", type=Path, default=config.DATASETS_DIR / "detector", help="검출기 데이터 폴더"
    )
    label.add_argument("--teacher", choices=("world", "mix"), default="world", help="선생 모델")
    label.add_argument("--conf", type=float, default=CONF, help="신뢰도 기준")

    review = sub.add_parser(
        "review", help="검수 작업 — R2: R1 초안 + YOLOE 식품 박스 / R3: --weights 가중치 초안"
    )
    review.add_argument(
        "--root", type=Path, default=config.DATASETS_DIR / "detector", help="검출기 데이터 폴더"
    )
    review.add_argument(
        "--drafts", type=Path, default=None, help="R1 초안. 기본 <root>/auto/tasks.json"
    )
    review.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="이 10종 가중치로 초안을 그린다(R3, 예: data/models/detector_r2.pt). "
        "CLIP 이 필요 없다",
    )
    review.add_argument(
        "--videos", nargs="+", default=None, help="--weights 로 초안을 그릴 영상 번호, 예: d07 d08"
    )
    review.add_argument(
        "--out", type=Path, default=None, help="작업을 쓸 폴더. 기본 <root>/review(R2 검수 원본)"
    )
    review.add_argument(
        "--conf",
        type=float,
        default=None,
        help=f"초안 신뢰도 기준. 기본 --weights 는 {CONF}(비서 검출과 같게), "
        f"없으면 YOLOE 식품 박스 {REVIEW_CONF}",
    )
    review.add_argument(
        "--val", nargs="+", default=["d05"], help="검증 영상 번호 (검수 순서의 맨 앞 · 맨 뒤)"
    )

    build = sub.add_parser("build", help="회차별 학습 묶음 (영상 단위 분할 · data.yaml)")
    build.add_argument(
        "--root", type=Path, default=config.DATASETS_DIR / "detector", help="검출기 데이터 폴더"
    )
    build.add_argument(
        "--labels",
        type=Path,
        nargs="+",
        required=True,
        help="라벨 폴더(여럿 가능). R1 은 <root>/auto/labels, R2 는 <root>/review/labels(import), "
        "R3 는 <root>/review/labels <root>/review_r3/labels",
    )
    build.add_argument(
        "--val-labels",
        type=Path,
        default=None,
        help="검증 사진의 라벨 폴더(없으면 --labels). 대조군은 R2 와 같은 사람 라벨을 준다",
    )
    build.add_argument("--val", nargs="+", required=True, help="검증으로 뺄 영상 번호, 예: d05")
    build.add_argument("--out", type=Path, required=True, help="회차 폴더, 예: <root>/yolo_r1")

    imp = sub.add_parser("import", help="검수 결과(Label Studio JSON) 또는 R1 초안 → YOLO 라벨")
    source = imp.add_mutually_exclusive_group(required=True)
    source.add_argument("--export", type=Path, help="Label Studio 에서 JSON 으로 내보낸 파일(R2)")
    source.add_argument("--predictions", type=Path, help="R1 초안 auto/tasks.json(대조군)")
    imp.add_argument(
        "--like",
        type=Path,
        default=None,
        help="이 폴더에 라벨이 있는 사진만(대조군 = R2 와 같은 사진)",
    )
    imp.add_argument(
        "--out", type=Path, required=True, help="라벨을 쓸 폴더, 예: <root>/review/labels"
    )
    imp.add_argument(
        "--min-side", type=int, default=MIN_SIDE, help="짧은 변이 이보다 작은 박스는 뺀다(px)"
    )

    args = parser.parse_args()
    if args.command in ("build", "review"):
        bad = [v for v in args.val if not VIDEO_ID.fullmatch(v)]
        if bad:
            parser.error(f"--val 은 d05 처럼 영상 번호로 씁니다: {bad}")
    if args.command == "review":
        if args.weights is not None and not args.videos:
            parser.error(
                "--weights 로 초안을 그릴 때는 --videos 로 영상을 고릅니다(예: d07 d08 d09)"
            )
        if args.videos and args.weights is None:
            parser.error("--videos 는 --weights 와 같이 씁니다(R2 갈래는 R1 초안의 사진을 쓴다)")
        bad = [v for v in args.videos or [] if not VIDEO_ID.fullmatch(v)]
        if bad:
            parser.error(f"--videos 는 d07 처럼 영상 번호로 씁니다: {bad}")
        if args.conf is None:
            args.conf = CONF if args.weights is not None else REVIEW_CONF
    if args.command in ("compare", "label", "review") and not 0 < args.conf < 1:
        parser.error(f"--conf 는 0 과 1 사이입니다: {args.conf}")
    return args


def main() -> None:
    args = parse_args()
    commands = {
        "compare": cmd_compare,
        "label": cmd_label,
        "review": cmd_review,
        "import": cmd_import,
        "build": cmd_build,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
