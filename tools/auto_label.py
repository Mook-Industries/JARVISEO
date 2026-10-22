"""자동 라벨 — 선생 모델로 학습 사진에 박스 초안을 붙인다.  담당: 최홍묵

YOLO 파인튜닝(#167)의 둘째 단계. tools/extract_frames.py 가 뽑은 사진에 11종
(pointing/detector.py 의 FINETUNE_CLASSES) 박스를 붙여 두 가지를 만든다.
- YOLO 라벨: R1 은 이것을 그대로 학습한다
- Label Studio 검수 파일: R2 는 사람이 고친 라벨로 학습한다

명령:
    compare  어느 선생이 우리 valid 사진에서 박스를 더 잘 치나 (YOLO-World 11종 vs yolo11x 10종)
    label    고른 선생으로 라벨 초안을 만든다

Colab 에서 돌린다. ultralytics 는 YOLO-World 를 처음 쓸 때 CLIP 을 git 에서 자동으로 설치해서,
팀 파이썬 환경에서 돌리면 승인 없이 패키지가 깔린다. 순수 함수는 CI 에서 테스트한다.

실행 (Colab, 레포 맨 위 폴더에서):
    python tools/auto_label.py compare --pointing /content/drive/MyDrive/JARVISEO/pointing
    python tools/auto_label.py label --root /content/drive/MyDrive/JARVISEO/detector

저장 (--root 아래):
    auto/labels/d01_0001.txt   YOLO 라벨 (번호 중심x 중심y 너비 높이, 0~1)
    auto/tasks.json            Label Studio 로 가져올 검수 작업
    auto/label_config.xml      Label Studio 프로젝트 설정
"""

import argparse
import csv
import json
from collections import Counter
from collections.abc import Callable
from functools import partial
from pathlib import Path

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

# 우리 종류 → YOLO-World 에 줄 영어 글귀. "product" 한 단어로는 무엇을 찾을지 몰라서
# 구체적인 물건 이름으로 찾고 product 로 묶는다.
PROMPTS: dict[str, tuple[str, ...]] = {
    "bottle": ("bottle",),
    "cup": ("cup",),
    "bowl": ("bowl",),
    "laptop": ("laptop",),
    "mouse": ("mouse",),
    "keyboard": ("keyboard",),
    "cell phone": ("cell phone", "smartphone"),
    "remote": ("remote",),
    "book": ("book",),
    "tv": ("tv", "computer monitor"),
    "product": (
        "snack bag",
        "instant noodle cup",
        "can",
        "glass jar",
        "food package",
        "instant rice container",
    ),
}
# 가리킴 정답표(labels.csv)의 이름 → 우리 종류. 없는 이름(glasses · watch 등 8장)은 11종 밖이다.
GT_TO_CLASS: dict[str, str] = {name: name for name in FINETUNE_CLASSES if name != "product"} | {
    "can": "product",
    "snack bag": "product",
    "jar": "product",
    "cup noodle": "product",
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
    """COCO 모델의 번호 → 우리 종류 이름. 11종 밖(person · chair 등)은 뺀다."""
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


def label_config() -> str:
    """Label Studio 프로젝트 설정. 종류는 FINETUNE_CLASSES 순서 그대로."""
    labels = "\n".join(f'    <Label value="{name}"/>' for name in FINETUNE_CLASSES)
    return (
        "<View>\n"
        '  <Image name="image" value="$image"/>\n'
        '  <RectangleLabels name="label" toName="image">\n'
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
    - class_recall: 겹치는 박스의 종류까지 맞은 비율, 11종 안 정답만. 학습 라벨은 종류가 틀리면
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
    """선생 모델. world = YOLO-World 11종, coco = yolo11x 10종(product 없음),
    mix = COCO 10종은 yolo11x, product 만 YOLO-World 로 찾아 합친다.

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


def cmd_compare(args: argparse.Namespace) -> None:
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

    args = parser.parse_args()
    if not 0 < args.conf < 1:
        parser.error(f"--conf 는 0 과 1 사이입니다: {args.conf}")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "compare":
        cmd_compare(args)
    else:
        cmd_label(args)


if __name__ == "__main__":
    main()
