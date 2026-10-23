"""자동 라벨 — 선생 모델로 학습 사진에 박스 초안을 붙이고, 회차별 학습 묶음을 만든다.  담당: 최홍묵

YOLO 파인튜닝(#167)의 둘째 단계. tools/extract_frames.py 가 뽑은 사진에 11종
(pointing/detector.py 의 FINETUNE_CLASSES) 박스를 붙여 두 가지를 만든다.
- YOLO 라벨: R1 은 이것을 그대로 학습한다
- Label Studio 검수 파일: R2 는 사람이 고친 라벨로 학습한다

명령:
    compare  어느 선생이 우리 valid 사진에서 박스를 더 잘 치나 (YOLO-World 11종 vs yolo11x 10종)
    label    고른 선생으로 라벨 초안을 만든다
    build    사진 + 라벨로 회차 폴더(yolo_r1 …)를 만든다. 학습 / 검증은 영상 단위로 나눈다

compare · label 은 Colab 에서 돌린다. ultralytics 는 YOLO-World 를 처음 쓸 때 CLIP 을 git 에서
자동으로 설치해서, 팀 파이썬 환경에서 돌리면 승인 없이 패키지가 깔린다.
build 는 모델을 안 써서 어디서든 돈다.

실행 (Colab, 레포 맨 위 폴더에서):
    python tools/auto_label.py compare --pointing /content/drive/MyDrive/JARVISEO/pointing
    python tools/auto_label.py label --root /content/drive/MyDrive/JARVISEO/detector
    python tools/auto_label.py build --root /content/drive/MyDrive/JARVISEO/detector \\
        --labels /content/drive/MyDrive/JARVISEO/detector/auto/labels --val d05 \\
        --out /content/drive/MyDrive/JARVISEO/detector/yolo_r1

저장 (--root 아래):
    auto/labels/d01_0001.txt   YOLO 라벨 (번호 중심x 중심y 너비 높이, 0~1)
    auto/tasks.json            Label Studio 로 가져올 검수 작업
    auto/label_config.xml      Label Studio 프로젝트 설정
    yolo_r1/images/{train,val}/ · yolo_r1/labels/{train,val}/ · yolo_r1/data.yaml   회차 하나
"""

import argparse
import csv
import json
import re
import shutil
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
# 영상 번호(extract_frames.py 의 이름 규칙). --val 에 이 모양만 받는다.
VIDEO_ID = re.compile(r"d\d{2}")
# 가장 많은 종류의 이만큼에 못 미치는 종류를 경고한다. 그 종류의 AP 만 바닥나는 원인이다.
RARE_RATIO = 0.1

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
    """Label Studio 프로젝트 설정. 종류는 FINETUNE_CLASSES 순서 그대로.

    밝기 · 대비 조절을 켠다. 밤 공원 영상(d04)은 밝기 중앙값이 9.9/255 라 그대로는 물건이 안 보이고,
    못 보고 지나친 물건은 라벨 없는 "배경"으로 학습된다. 화면에서만 바뀌고 사진 원본은 그대로다.
    """
    labels = "\n".join(f'    <Label value="{name}"/>' for name in FINETUNE_CLASSES)
    return (
        "<View>\n"
        '  <Image name="image" value="$image" brightnessControl="true" contrastControl="true"/>\n'
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

    "전체의 10%"로 재면 11종의 평균이 9% 라 거의 모든 종류가 걸린다. 그래서 가장 많은 종류와 견준다.
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
    names_file = class_names_file(args.labels)
    val_videos = set(args.val)

    # 먼저 다 읽고 검사한 뒤에 쓴다. 중간에 멈추면 반쯤 만든 회차 폴더가 다음 실행을 막는다.
    # 라벨 파일이 있는 사진만 넣는다. 검수하지 않은 사진을 "물건 없음"으로 가르치지 않으려고.
    # 빈 라벨 파일은 물건이 없는 사진이라 넣는다.
    items, missing = [], 0
    try:
        table = None
        if names_file is not None:
            names = names_file.read_text(encoding="utf-8").splitlines()
            table = remap_table([name.strip() for name in names if name.strip()])
        for path in frames:
            label_path = args.labels / f"{path.stem}.txt"
            if not label_path.exists():
                missing += 1
                continue
            raw = label_path.read_text(encoding="utf-8").splitlines()
            lines = [remap_line(line, table) for line in raw if line.strip()]
            items.append((path, split_of(path.stem, val_videos), lines))
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
    if names_file is not None:
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

    build = sub.add_parser("build", help="회차별 학습 묶음 (영상 단위 분할 · data.yaml)")
    build.add_argument(
        "--root", type=Path, default=config.DATASETS_DIR / "detector", help="검출기 데이터 폴더"
    )
    build.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="라벨 폴더. R1 은 <root>/auto/labels, R2 는 Label Studio 에서 내보낸 labels",
    )
    build.add_argument("--val", nargs="+", required=True, help="검증으로 뺄 영상 번호, 예: d05")
    build.add_argument("--out", type=Path, required=True, help="회차 폴더, 예: <root>/yolo_r1")

    args = parser.parse_args()
    if args.command == "build":
        bad = [v for v in args.val if not VIDEO_ID.fullmatch(v)]
        if bad:
            parser.error(f"--val 은 d05 처럼 영상 번호로 씁니다: {bad}")
    elif not 0 < args.conf < 1:
        parser.error(f"--conf 는 0 과 1 사이입니다: {args.conf}")
    return args


def main() -> None:
    args = parse_args()
    commands = {"compare": cmd_compare, "label": cmd_label, "build": cmd_build}
    commands[args.command](args)


if __name__ == "__main__":
    main()
