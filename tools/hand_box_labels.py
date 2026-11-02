"""손 박스 자동 정답 — 손 영상으로 손 박스 검출기의 학습 묶음을 만든다.  담당: 최홍묵

손 검출 개선 ⑤-b(#256). MediaPipe 는 손바닥을 찾을 때 사진 전체를 192px 로 줄여서
팔을 뻗은 작은 손 · 흐린 손을 놓친다. 그 단계를 우리 안경 영상으로 학습한
손 박스 검출기(YOLOv8n, 1종 hand)로 보강한다. 이 도구는 학습 정답을 사람 검수 없이 만든다.

정답 박스 = MediaPipe 가 찾은 손의 관절 상자 + 여백. 사진 전체에서 못 찾으면
4조각 → 3조각에서 다시 찾고, 그래도 없으면 바로 앞뒤 프레임의 박스로
한 장 빈 칸을 채운다(이어 읽기). 사진 전체가 놓친 손(조각 · 이어 읽기 박스)이
검출기가 MediaPipe 보다 더 배울 몫이다.

손 영상에서 손을 못 찾은 프레임은 배경(손 없음)으로 넣지 않고 건너뛴다.
손 영상은 거의 모든 프레임에 손이 있어서, 넣으면 MediaPipe 가 놓친 손을
"손이 없다"고 가르친다. 배경은 검출기 영상 프레임과 s05 의 손 없음 표시만 쓴다.
평가에 쓰는 사진(valid · test · 실물 루프)은 읽지 않는다. s05 는 학습용 세션이다.

명령:
    label  손 영상 · 검출기 프레임 · s05 에서 자동 정답을 만든다(10~15분)
    sheet  찾은 단계(사진 전체 · 4조각 · 3조각 · 이어 읽기 · 배경)별 확인 모음 사진
    build  장소 단위로 학습 / 검증을 나눈 YOLO 묶음과 Drive 에 올릴 zip

실행 (레포 맨 위 폴더에서):
    python tools/hand_box_labels.py label
    python tools/hand_box_labels.py sheet
    python tools/hand_box_labels.py build

저장 (data/datasets/hand/):
    boxes/images · boxes/labels   긴 변 640 사진 · YOLO 라벨(0 중심x 중심y 너비 높이, 0~1)
    boxes/boxes.csv               박스 한 줄(배경 사진은 좌표 없이 한 줄)
    boxes/label_report.txt        출처 × 단계 장수 · 채운 장수 · 건너뛴 장수
    boxes/sheets/<단계>.jpg        확인 모음 사진
    yolo_h1/                      학습 묶음(images · labels 의 train · val, data.yaml)
    drive_upload/                 Drive 에 올릴 zip 과 노트북 사본
"""

import argparse
import csv
import random
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# 손 영상은 hand_clips 와 같은 함수로 읽어야 같은 구간 · 같은 시각의 프레임을 본다.
# tools 는 패키지가 아니다. python tools/<도구>.py 로 돌리면 이 폴더가 sys.path 맨 앞이다.
from hand_clips import (
    clip_info,
    find_videos,
    read_csv,
    sampled_frames,
    save_jpg,
    segment_at,
    segments_by_video,
)

from jarviseo import config
from jarviseo.pointing.hands import (
    HandDetector,
    HandPose,
    landmark_box,
    merge_hands,
    search_tiles,
    shift_hand,
)
from jarviseo.types import BBox

STEP_SEC = 0.2  # hand_clips 의 뽑기 간격과 같다 — 이어 읽기는 이 간격의 앞뒤 프레임을 본다
# 묶음에는 0.4초 칸만 넣는다. 0.2초 옆 프레임은 거의 같은 사진이다.
KEEP_EVERY = 2
# 관절은 마디 중심이라 손끝 살 · 손바닥 가장자리가 관절 상자 밖으로 나온다.
# 출발점이고 sheet 모음 사진으로 손이 잘리지 않는지 본다.
BOX_MARGIN = 0.15
# 학습 입력 크기. ultralytics 가 어차피 긴 변을 이 크기로 줄여 읽어서, 미리 줄여 zip 을 줄인다.
IMAGE_LONG_SIDE = 640
# 처음 보는 장소에서 잘 찾는 epoch 를 고르려고 장소로 나눈다(검출기 학습도 d05 가 검증).
VAL_PLACES = ("loby",)
VAL_VIDEOS = ("d05",)
STAGES = ("full", "g2x2", "sq3", "fill")
BACKGROUND = "background"
SHEET_COUNT = 40
SHEET_TILE = (320, 180)
SHEET_COLUMNS = 8
ROUND = "h1"
ZIP_NAME = f"JARVISEO_HAND_BOX_{ROUND.upper()}_data.zip"
NOTEBOOK = config.PROJECT_ROOT / "notebooks" / "hand_box_finetune.ipynb"
CSV_COLUMNS = [
    "image",
    "source",
    "video",
    "place",
    "person",
    "cls",
    "t_sec",
    "stage",
    "x1",
    "y1",
    "x2",
    "y2",
]

Finder = Callable[[np.ndarray], list[HandPose]]


@dataclass
class Shot:
    """정답을 붙일 사진 한 장. boxes 는 원본 픽셀 좌표의 관절 상자(여백 전)다.

    scale · size 는 저장한 사진(긴 변 640)의 배율과 (너비, 높이).
    """

    image_id: str
    source: str  # hand · detector · s05
    video: str
    stage: str  # full · g2x2 · sq3 · fill · none · background
    boxes: list[BBox] = field(default_factory=list)
    place: str = ""
    person: str = ""
    cls: str = ""
    t_sec: float = 0.0
    scale: float = 1.0
    size: tuple[int, int] = (0, 0)


def padded_box(box: BBox, image_size: tuple[int, int], margin: float = BOX_MARGIN) -> BBox:
    """관절 상자에 사방으로 너비 · 높이의 margin 만큼 더한다. 사진 밖은 자른다."""
    width, height = image_size
    dx, dy = (box.x2 - box.x1) * margin, (box.y2 - box.y1) * margin
    return BBox(
        max(0.0, box.x1 - dx),
        max(0.0, box.y1 - dy),
        min(float(width), box.x2 + dx),
        min(float(height), box.y2 + dy),
    )


def square_tiles(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """짧은 변 길이의 정사각 3개(양 끝 · 가운데). 4조각이 못 찾은 손을 찾을 때만 쓴다.

    사전 측정에서 4조각 뒤에 돌려 손 영상 프레임 923장을 더 찾았다. 실행 쪽에는 안 넣는다.
    """
    side = min(width, height)
    if width >= height:
        return [(x, 0, x + side, side) for x in sorted({0, (width - side) // 2, width - side})]
    return [(0, y, side, y + side) for y in sorted({0, (height - side) // 2, height - side})]


def search_squares(image: np.ndarray, find: Finder) -> list[HandPose]:
    """정사각 3개에서 찾은 손. 원본 좌표로 옮기고 같은 손은 하나만 남긴다."""
    height, width = image.shape[:2]
    found: list[HandPose] = []
    for x1, y1, x2, y2 in square_tiles(width, height):
        found += [shift_hand(h, x1, y1) for h in find(image[y1:y2, x1:x2])]
    return merge_hands([], found)


def staged_hands(image: np.ndarray, find: Finder) -> tuple[str, list[HandPose]]:
    """사진 전체 → 4조각 → 3조각 순서로 찾아, 손이 처음 나온 단계와 그 손들.

    셋 다 못 찾으면 ("none", []). find 는 MediaPipe 를 한 번 돌리는 함수다.
    """
    hands = find(image)
    if hands:
        return "full", hands
    hands = merge_hands([], search_tiles(image, find))
    if hands:
        return "g2x2", hands
    hands = search_squares(image, find)
    if hands:
        return "sq3", hands
    return "none", []


def fill_holes(shots: Sequence[Shot], step: float = STEP_SEC) -> int:
    """한 장만 빈 프레임을 바로 앞뒤 프레임 박스의 평균으로 채운다(이어 읽기). 채운 장 수.

    shots 는 영상 하나의 프레임을 시간순으로. 앞뒤가 같은 손 모양 구간이고 step 간격으로
    붙어 있고 손 수가 같을 때만 채운다. 손이 둘이면 왼쪽부터 짝짓는다.
    사전 측정에서 이렇게 채운 프레임이 262장이었다. 0.8초까지 늘리면 529장이지만
    그사이 손이 움직여 박스가 어긋난다.
    """
    filled = 0
    for prev, shot, nxt in zip(shots, shots[1:], shots[2:], strict=False):
        if shot.boxes or not prev.boxes or len(prev.boxes) != len(nxt.boxes):
            continue
        if not prev.cls == shot.cls == nxt.cls:
            continue
        if shot.t_sec - prev.t_sec > step * 1.5 or nxt.t_sec - shot.t_sec > step * 1.5:
            continue
        before = sorted(prev.boxes, key=lambda b: b.center[0])
        after = sorted(nxt.boxes, key=lambda b: b.center[0])
        shot.boxes = [
            BBox((a.x1 + b.x1) / 2, (a.y1 + b.y1) / 2, (a.x2 + b.x2) / 2, (a.y2 + b.y2) / 2)
            for a, b in zip(before, after, strict=True)
        ]
        shot.stage = "fill"
        filled += 1
    return filled


def s05_role(has_hand: bool, found: int) -> str:
    """s05 사진의 쓰임. 사람이 표시한 손 있음 / 없음이 MediaPipe 보다 우선한다.

    손 없음 표시 → 배경(MediaPipe 가 잡은 손은 버린다 — 사전 측정 12장 중 1장이 헛손).
    손 있음인데 못 찾음 → 건너뜀(손이 어디 있는지 몰라 배경으로도 못 쓴다).
    """
    if not has_hand:
        return BACKGROUND
    return "positive" if found else "skip"


def yolo_line(box: BBox, image_size: tuple[int, int]) -> str:
    """YOLO 라벨 한 줄: 종류 번호(손 = 0) · 중심 x · 중심 y · 너비 · 높이(사진 크기로 나눈 0~1)."""
    width, height = image_size
    return (
        f"0 {(box.x1 + box.x2) / 2 / width:.6f} {(box.y1 + box.y2) / 2 / height:.6f} "
        f"{(box.x2 - box.x1) / width:.6f} {(box.y2 - box.y1) / height:.6f}"
    )


def split_of(source: str, place: str, video: str) -> str:
    """학습 / 검증. 손 영상은 장소, 검출기 프레임은 영상으로 나누고 s05 는 늘 학습이다.

    프레임을 무작위로 나누면 0.2초 옆 프레임이 검증에 섞여 점수가 부풀려진다.
    """
    if source == "hand" and place in VAL_PLACES:
        return "val"
    if source == "detector" and video in VAL_VIDEOS:
        return "val"
    return "train"


def is_kept(t_sec: float) -> bool:
    """묶음에 넣는 0.4초 칸인가(0.2초 칸 두 장에 한 장)."""
    return round(t_sec / STEP_SEC) % KEEP_EVERY == 0


def shrink(image: np.ndarray, long_side: int = IMAGE_LONG_SIDE) -> tuple[np.ndarray, float]:
    """긴 변을 long_side 로 줄인 사진과 배율. 이미 작으면 그대로(배율 1)."""
    height, width = image.shape[:2]
    scale = min(1.0, long_side / max(height, width))
    if scale == 1.0:
        return image, 1.0
    size = (round(width * scale), round(height * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def label_rows(shot: Shot) -> tuple[list[str], list[dict]]:
    """YOLO 라벨 줄과 boxes.csv 줄. 박스는 저장한 사진 좌표로 줄인 뒤 여백을 더한다.

    배경 사진은 라벨이 빈 파일이고, boxes.csv 에는 좌표 없이 한 줄을 남긴다(장수를 세려고).
    """
    base = {
        "image": shot.image_id,
        "source": shot.source,
        "video": shot.video,
        "place": shot.place,
        "person": shot.person,
        "cls": shot.cls,
        "t_sec": round(shot.t_sec, 2),
        "stage": shot.stage,
    }
    if not shot.boxes:
        return [], [{**base, "x1": "", "y1": "", "x2": "", "y2": ""}]
    lines, rows = [], []
    for b in shot.boxes:
        s = shot.scale
        box = padded_box(BBox(b.x1 * s, b.y1 * s, b.x2 * s, b.y2 * s), shot.size)
        lines.append(yolo_line(box, shot.size))
        rows.append(
            {
                **base,
                "x1": round(box.x1, 1),
                "y1": round(box.y1, 1),
                "x2": round(box.x2, 1),
                "y2": round(box.y2, 1),
            }
        )
    return lines, rows


def read_image(path: Path) -> np.ndarray:
    """cv2.imread 는 윈도우에서 경로에 한글이 있으면 조용히 실패해서, 바이트로 읽어 푼다."""
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"사진을 못 읽었습니다: {path}")
    return image


def save_small(out: Path, shot: Shot, image: np.ndarray) -> None:
    """긴 변 640 으로 줄여 저장하고 배율 · 크기를 shot 에 적는다."""
    small, shot.scale = shrink(image)
    shot.size = (small.shape[1], small.shape[0])
    save_jpg(out / "images" / f"{shot.image_id}.jpg", small)


def write_label(out: Path, shot: Shot) -> list[dict]:
    lines, rows = label_rows(shot)
    text = "".join(f"{line}\n" for line in lines)
    (out / "labels" / f"{shot.image_id}.txt").write_text(text, encoding="utf-8")
    return rows


def label_hand_videos(root: Path, out: Path, find: Finder, counts: Counter) -> list[dict]:
    """손 영상 9개. 0.2초마다 찾고 이어 읽기로 채운 뒤, 0.4초 칸의 손 있는 프레임만 남긴다."""
    by_video = segments_by_video(read_csv(root / "segments.csv"))
    paths = {p.stem: p for p in find_videos(root / "videos")}
    rows: list[dict] = []
    for video, segments in sorted(by_video.items()):
        place, person = clip_info(paths[video])
        shots: list[Shot] = []
        sampled = sampled_frames(
            paths[video],
            STEP_SEC,
            retrieve=lambda t, segs=segments: segment_at(segs, t) is not None,
        )
        for t_sec, frame in sampled:
            cls = segment_at(segments, t_sec)
            if frame is None or cls is None:
                continue
            stage, hands = staged_hands(frame, find)
            counts[("hand", stage)] += 1
            shot = Shot(
                image_id=f"{video}_{round(t_sec * 1000):06d}",
                source="hand",
                video=video,
                stage=stage,
                boxes=[landmark_box(h) for h in hands],
                place=place,
                person=person,
                cls=cls,
                t_sec=t_sec,
            )
            # 이어 읽기는 영상이 끝나야 정해진다. 프레임을 들고 있지 않게 남길 칸은 먼저 저장한다.
            if is_kept(t_sec):
                save_small(out, shot, frame)
            shots.append(shot)
        filled = fill_holes(shots)
        counts[("hand", "fill")] += filled
        kept = skipped = 0
        for shot in shots:
            if not is_kept(shot.t_sec):
                continue
            if not shot.boxes:
                (out / "images" / f"{shot.image_id}.jpg").unlink()
                skipped += 1
                continue
            rows += write_label(out, shot)
            kept += 1
        counts[("hand", "skipped")] += skipped
        print(f"{video}: 프레임 {len(shots)} · 채움 {filled} · 남김 {kept} · 건너뜀 {skipped}")
    return rows


def label_detector_frames(frames_dir: Path, out: Path, find: Finder, counts: Counter) -> list[dict]:
    """검출기 영상 프레임(3초 간격). 손이 있으면 손 박스, 없으면 배경."""
    if not frames_dir.exists():
        raise SystemExit(f"검출기 프레임 폴더가 없습니다: {frames_dir} (--detector-frames)")
    rows: list[dict] = []
    for path in sorted(frames_dir.rglob("*.jpg")):
        image = read_image(path)
        stage, hands = staged_hands(image, find)
        shot = Shot(
            image_id=path.stem,
            source="detector",
            video=path.parent.name,
            stage=stage if hands else BACKGROUND,
            boxes=[landmark_box(h) for h in hands],
        )
        counts[("detector", shot.stage)] += 1
        save_small(out, shot, image)
        rows += write_label(out, shot)
    print(f"검출기 프레임: {sum(v for (s, _), v in counts.items() if s == 'detector')}장")
    return rows


def label_s05(pointing: Path, out: Path, find: Finder, counts: Counter) -> list[dict]:
    """가리킴 학습용 세션 s05. labels.csv 에서 train 줄만 읽어 평가 사진은 건드리지 않는다."""
    rows: list[dict] = []
    for row in read_csv(pointing / "labels.csv"):
        if row["split"] != "train" or row["session"] != "5" or row["excluded"] == "1":
            continue
        image = read_image(pointing / "raw" / row["file"])
        stage, hands = staged_hands(image, find)
        role = s05_role(row["has_hand"] == "1", len(hands))
        counts[("s05", role)] += 1
        if role == "skip":
            continue
        background = role == BACKGROUND
        shot = Shot(
            image_id=Path(row["file"]).stem,
            source="s05",
            video="s05",
            stage=BACKGROUND if background else stage,
            boxes=[] if background else [landmark_box(h) for h in hands],
        )
        save_small(out, shot, image)
        rows += write_label(out, shot)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    """encoding 을 빼면 윈도우에서 cp949 로 저장되고, newline="" 을 빼면 빈 줄이 끼인다."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def box_widths(rows: Sequence[dict]) -> list[float]:
    """박스 너비(px, 긴 변 640 사진 기준). 배경 줄은 뺀다."""
    return [float(r["x2"]) - float(r["x1"]) for r in rows if r["x1"] != ""]


def images_by_kind(rows: Sequence[dict]) -> Counter:
    """사진 장수를 (출처, 단계, 학습/검증)으로. 박스가 여럿인 사진도 한 장으로 센다."""
    kinds = {
        r["image"]: (r["source"], r["stage"], split_of(r["source"], r["place"], r["video"]))
        for r in rows
    }
    return Counter(kinds.values())


def format_report(rows: Sequence[dict], counts: Counter) -> str:
    hand_stages = ("full", "g2x2", "sq3", "none")
    frames = sum(counts[("hand", st)] for st in hand_stages)
    found = sum(counts[("hand", st)] for st in hand_stages[:3]) + counts[("hand", "fill")]
    stages = " · ".join(f"{st} {counts[('hand', st)]}" for st in hand_stages)
    widths = box_widths(rows)
    lines = [
        f"# 손 박스 자동 정답 ({ROUND})",
        f"손 영상 프레임(0.2초) {frames}: {stages} · 이어 읽기 {counts[('hand', 'fill')]}",
        f"정답 붙은 프레임 {found}/{frames} ({found / max(frames, 1):.1%})",
        f"손 영상 0.4초 칸에서 손 못 찾아 건너뜀 {counts[('hand', 'skipped')]}",
        f"s05: 손 {counts[('s05', 'positive')]} · 배경 {counts[('s05', BACKGROUND)]} · "
        f"건너뜀 {counts[('s05', 'skip')]}",
        "",
        "묶음에 들어갈 사진 (출처 · 단계 · 학습/검증: 장수)",
    ]
    lines += [
        f"- {s} · {st} · {sp}: {n}" for (s, st, sp), n in sorted(images_by_kind(rows).items())
    ]
    if widths:
        p10 = statistics.quantiles(widths, n=10)[0] if len(widths) >= 10 else min(widths)
        lines += [
            "",
            f"손 박스 {len(widths)}개 · 640 사진에서 너비 중앙값 {statistics.median(widths):.0f}px "
            f"· 하위 10% {p10:.0f}px",
        ]
    return "\n".join(lines) + "\n"


def run_label(hand_root: Path, frames_dir: Path, pointing: Path, out: Path) -> None:
    csv_path = out / "boxes.csv"
    if csv_path.exists():
        raise SystemExit(f"이미 있습니다: {csv_path} — 다시 만들려면 {out} 를 지우세요")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    # 실제 비서와 같은 손 검출(기준 0.1 · 손 2개)을 한 번 돌리는 함수. 판정 · 조각 · 사진 특징 ·
    # 손 박스는 안 쓴다 — 조각은 staged_hands 가 직접 돌고, 손 박스는 이 정답으로 학습할 모델이다.
    find = HandDetector(classifier=None, tiles=False, encoder=None, boxes=False).detect
    counts: Counter = Counter()
    rows = label_hand_videos(hand_root, out, find, counts)
    rows += label_detector_frames(frames_dir, out, find, counts)
    rows += label_s05(pointing, out, find, counts)
    write_rows(csv_path, rows)
    report = format_report(rows, counts)
    (out / "label_report.txt").write_text(report, encoding="utf-8")
    print(report)


def draw_tile(path: Path, rows: Sequence[dict]) -> np.ndarray:
    """사진에 박스(초록)와 이름을 그려 모음 칸 크기로 줄인다."""
    image = read_image(path)
    for r in rows:
        if r["x1"] == "":
            continue
        p1 = (round(float(r["x1"])), round(float(r["y1"])))
        p2 = (round(float(r["x2"])), round(float(r["y2"])))
        cv2.rectangle(image, p1, p2, (0, 220, 0), 2)
    tile = cv2.resize(image, SHEET_TILE, interpolation=cv2.INTER_AREA)
    cv2.putText(tile, path.stem, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(tile, path.stem, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return tile


def contact(tiles: Sequence[np.ndarray]) -> np.ndarray:
    tile_w, tile_h = SHEET_TILE
    rows = -(-len(tiles) // SHEET_COLUMNS)
    sheet = np.zeros((rows * tile_h, SHEET_COLUMNS * tile_w, 3), np.uint8)
    for i, tile in enumerate(tiles):
        y, x = (i // SHEET_COLUMNS) * tile_h, (i % SHEET_COLUMNS) * tile_w
        sheet[y : y + tile_h, x : x + tile_w] = tile
    return sheet


def run_sheet(out: Path) -> None:
    """단계마다 무작위 40장(시드 고정)에 박스를 그린 모음 사진. 사람 검수가 아니라 보기만 한다."""
    by_image: dict[str, list[dict]] = defaultdict(list)
    for r in read_csv(out / "boxes.csv"):
        by_image[r["image"]].append(r)
    rng = random.Random(0)
    (out / "sheets").mkdir(exist_ok=True)
    for stage in (*STAGES, BACKGROUND):
        ids = sorted(i for i, rs in by_image.items() if rs[0]["stage"] == stage)
        if not ids:
            continue
        pick = rng.sample(ids, min(SHEET_COUNT, len(ids)))
        tiles = [draw_tile(out / "images" / f"{i}.jpg", by_image[i]) for i in pick]
        save_jpg(out / "sheets" / f"{stage}.jpg", contact(tiles))
        print(f"{stage}: {len(ids)}장 중 {len(pick)}장 → {out / 'sheets' / f'{stage}.jpg'}")


def data_yaml() -> str:
    """ultralytics 데이터 설정. path 키는 넣지 않는다(auto_label.data_yaml 과 같은 이유).

    path 가 없으면 이 파일이 있는 폴더를 기준으로 train · val 을 푼다. path: . 는 작업 폴더
    기준이라 맥과 Colab 에서 서로 다른 곳을 가리킨다.
    """
    return 'train: images/train\nval: images/val\nnames:\n  0: "hand"\n'


def prep_report(rows: Sequence[dict]) -> str:
    """학습 묶음 기록. 노트북이 Colab 보고서에 그대로 옮긴다."""
    kinds = images_by_kind(rows)
    lines = [f"# 손 박스 학습 묶음 {ROUND}"]
    for split in ("train", "val"):
        part = [r for r in rows if split_of(r["source"], r["place"], r["video"]) == split]
        images = len({r["image"] for r in part})
        background = sum(n for (_, st, sp), n in kinds.items() if sp == split and st == BACKGROUND)
        lines.append(
            f"- {split}: 사진 {images} (배경 {background}) · 손 박스 {len(box_widths(part))}"
        )
    lines.append(
        "- 검증 = 손 영상 " + " · ".join(VAL_PLACES) + " + 검출기 " + " · ".join(VAL_VIDEOS)
    )
    lines += [f"  - {s} · {st} · {sp}: {n}" for (s, st, sp), n in sorted(kinds.items())]
    return "\n".join(lines) + "\n"


def run_build(boxes: Path, out: Path, upload: Path, notebook: Path) -> None:
    """장소로 나눈 YOLO 묶음 → zip 하나와 노트북 사본을 Drive 에 올릴 폴더에. 덮어쓰지 않는다."""
    zip_path = upload / ZIP_NAME
    for path in (out, zip_path):
        if path.exists():
            raise SystemExit(f"이미 있습니다: {path} — 다시 만들려면 지우세요(덮어쓰지 않음)")
    if not notebook.exists():
        raise SystemExit(f"노트북이 없습니다: {notebook}")
    rows = read_csv(boxes / "boxes.csv")
    splits = {r["image"]: split_of(r["source"], r["place"], r["video"]) for r in rows}
    for image_id, split in sorted(splits.items()):
        for kind, suffix in (("images", ".jpg"), ("labels", ".txt")):
            target = out / kind / split / f"{image_id}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(boxes / kind / f"{image_id}{suffix}", target)
    (out / "data.yaml").write_text(data_yaml(), encoding="utf-8")
    report = prep_report(rows)
    (out / "prep_report.txt").write_text(report, encoding="utf-8")
    upload.mkdir(parents=True, exist_ok=True)
    # 사진은 이미 jpg 라 더 줄지 않는다. 압축하지 않고 담아 빨리 만든다.
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                z.write(path, Path(out.name) / path.relative_to(out))
    shutil.copy2(notebook, upload / notebook.name)
    print(report)
    size_mb = zip_path.stat().st_size / 1e6
    print(f"zip: {zip_path} ({size_mb:.0f}MB) · 노트북: {upload / notebook.name}")


def parse_args() -> argparse.Namespace:
    hand = config.DATASETS_DIR / "hand"
    parser = argparse.ArgumentParser(description="손 박스 자동 정답 · 학습 묶음 (#256 ⑤-b)")
    parser.add_argument("command", choices=("label", "sheet", "build"))
    parser.add_argument(
        "--hand-root", type=Path, default=hand, help="손 영상 폴더(videos/ · segments.csv)"
    )
    parser.add_argument(
        "--detector-frames",
        type=Path,
        default=config.DATASETS_DIR / "detector" / "frames",
        help="검출기 영상 프레임 폴더(영상별 하위 폴더)",
    )
    parser.add_argument("--pointing", type=Path, default=config.DATASETS_DIR / "pointing")
    parser.add_argument("--boxes", type=Path, default=hand / "boxes", help="자동 정답 폴더")
    parser.add_argument("--out", type=Path, default=hand / f"yolo_{ROUND}", help="학습 묶음 폴더")
    parser.add_argument(
        "--upload", type=Path, default=hand / "drive_upload", help="Drive 에 올릴 폴더"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "label":
            run_label(args.hand_root, args.detector_frames, args.pointing, args.boxes)
        elif args.command == "sheet":
            run_sheet(args.boxes)
        else:
            run_build(args.boxes, args.out, args.upload, NOTEBOOK)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
