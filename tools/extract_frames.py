"""영상 → 학습용 사진 — 1초에 1장 뽑고, 거의 같은 장면은 버린다.  담당: 최홍묵

YOLO 파인튜닝(#167)의 첫 단계. 안경 카메라로 찍은 영상(QuickTime .mov)을 사진으로 바꿔
tools/auto_label.py 가 라벨 초안을 붙일 수 있게 한다.

파일 이름에 영상 번호(d01)를 남긴다. 학습 / 검증을 영상 단위로 나눌 때 이 번호만 본다.
같은 영상에서 1초 차이로 뽑은 사진은 거의 같아서, 사진 단위로 섞으면 검증 점수가 부풀려진다.

실행 (레포 맨 위 폴더에서):
    python tools/extract_frames.py                    # data/datasets/detector/videos/ 의 영상 전부
    python tools/extract_frames.py --max-distance 8   # 비슷한 장면을 더 많이 버린다

저장:
    data/datasets/detector/frames/d01/d01_0001.jpg
    data/datasets/detector/frames.csv     file · video · t_sec (어느 영상 몇 초에서 뽑았나)

이미 사진 폴더가 있는 영상은 건너뛴다. 다시 뽑으려면 그 영상의 폴더를 지운다.
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

from jarviseo import config

VIDEO_SUFFIXES = (".mov", ".mp4")
# d01_kitchen — 영상 번호(d + 두 자리) + 장소(영어 소문자, capture_scenes.py --place 와 같은 규칙)
VIDEO_NAME = re.compile(r"(d\d{2})_[a-z]+")
FRAMES_COLUMNS = ["file", "video", "t_sec"]
JPEG_QUALITY = 95
# dHash 64비트 중 이만큼 이하로 다르면 같은 장면으로 본다. 출발점이다.
# 3분 영상 하나에서 60~100장이 남는지 보고 조정한다.
MAX_DISTANCE = 6


def video_id(path: Path) -> str:
    """영상 파일 이름에서 번호를 꺼낸다. 예: d01_kitchen.mov → d01

    분할이 이 번호 하나에 기대므로, 규칙에 안 맞는 이름은 조용히 넘기지 않고 멈춘다.
    """
    match = VIDEO_NAME.fullmatch(path.stem)
    if match is None:
        raise ValueError(
            f"영상 이름은 d01_kitchen 처럼 짓습니다(번호 두 자리 · 장소 영어 소문자): {path.name}"
        )
    return match.group(1)


class FrameSampler:
    """영상 시각을 보고 every_sec 마다 한 장씩 고른다.

    프레임 번호로 세지 않는다. QuickTime 녹화는 초당 장 수가 일정하지 않을 수 있어서
    "30장마다 1장"으로 세면 간격이 밀린다.
    """

    def __init__(self, every_sec: float) -> None:
        self.every_sec = every_sec
        self.target = 0.0

    def take(self, t_sec: float) -> bool:
        if t_sec < self.target:
            return False
        # 다음 목표는 지금 칸의 끝(2.4초에 골랐으면 3.0초). 끊겨서 여러 칸을 건너뛰어도
        # 한 장만 고르고, 같은 사진을 여러 장 저장하지 않는다.
        self.target = (t_sec // self.every_sec + 1) * self.every_sec
        return True


def dhash(image: np.ndarray) -> int:
    """사진의 지문(64비트, 차이 해시). 9×8 로 줄여 이웃한 두 칸 중 왼쪽이 더 밝으면 1.

    밝기가 조금 바뀌어도 지문이 거의 그대로라, 조명만 다른 같은 장면을 중복으로 잡는다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, :-1] > small[:, 1:]).flatten()
    return sum(1 << i for i, bit in enumerate(bits) if bit)


def hamming(a: int, b: int) -> int:
    """두 지문에서 다른 비트 수. 0 이면 같은 그림, 64 에 가까울수록 다른 그림."""
    return (a ^ b).bit_count()


class Deduper:
    """마지막으로 남긴 장과 비교해 거의 같은 장면을 버린다.

    바로 앞에 뽑은 장과 비교하지 않는다. 고개를 천천히 돌리면 1초마다 조금씩만 달라져서,
    앞 장과 비교하면 장면이 크게 바뀌어도 계속 "비슷하다"로 버려진다.
    남긴 장과 비교해야 차이가 쌓였을 때 다음 장이 남는다.
    """

    def __init__(self, max_distance: int) -> None:
        self.max_distance = max_distance
        self.last_kept: int | None = None

    def is_new(self, fingerprint: int) -> bool:
        if self.last_kept is not None and hamming(fingerprint, self.last_kept) <= self.max_distance:
            return False
        self.last_kept = fingerprint
        return True


def frame_name(video: str, number: int) -> str:
    """영상 번호 · 장 번호로 파일 이름을 만든다. 예: ("d01", 7) → d01_0007.jpg"""
    return f"{video}_{number:04d}.jpg"


def save_jpg(path: Path, image: np.ndarray) -> None:
    """jpg 로 저장한다. capture_scenes.py 와 같은 방식이다.

    cv2.imwrite 는 윈도우에서 경로에 한글이 있으면 에러 없이 실패한다.
    그래서 메모리에서 인코딩한 뒤 파이썬이 파일을 쓰게 한다.
    """
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"jpg 인코딩에 실패했습니다: {path}")
    buffer.tofile(path)


def append_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    """frames.csv 에 줄을 덧붙인다. 파일이 처음 생길 때만 헤더를 쓴다.

    encoding 을 빼면 윈도우에서 cp949 로 저장되고, newline="" 을 빼면 줄 사이에 빈 줄이 생긴다.
    """
    is_new = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FRAMES_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def extract(
    video_path: Path, root: Path, every_sec: float, max_distance: int
) -> tuple[list[dict[str, object]], int, float] | None:
    """영상 하나에서 사진을 뽑아 root/frames/<번호>/ 에 저장한다.

    (frames.csv 에 쓸 줄, 시각으로 고른 장 수, 영상 길이 초). 영상을 못 열면 None.
    """
    video = video_id(video_path)
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    if not cap.isOpened() or not ok:
        cap.release()
        return None
    out_dir = root / "frames" / video
    out_dir.mkdir(parents=True)  # 첫 장을 읽은 뒤에 만든다. 못 여는 영상은 빈 폴더를 남기지 않는다
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sampler, deduper = FrameSampler(every_sec), Deduper(max_distance)
    rows: list[dict[str, object]] = []
    sampled, index, t_sec = 0, 0, 0.0
    while ok:
        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        # 백엔드에 따라 시각이 0 으로만 나온다. 그때는 프레임 번호 ÷ FPS 로 대신한다.
        if t_sec <= 0 and index > 0:
            t_sec = index / fps
        if sampler.take(t_sec):
            sampled += 1
            if deduper.is_new(dhash(frame)):
                name = frame_name(video, len(rows) + 1)
                save_jpg(out_dir / name, frame)
                # 다른 컴퓨터에서도 읽히게 root 기준 상대 경로를 / 로 적는다(labels.csv 와 같다)
                rel = (out_dir / name).relative_to(root).as_posix()
                rows.append({"file": rel, "video": video, "t_sec": round(t_sec, 2)})
        index += 1
        ok, frame = cap.read()
    cap.release()
    return rows, sampled, t_sec


def find_videos(videos_dir: Path) -> list[Path]:
    """영상 목록. 이름 규칙이 틀리거나 번호가 겹치면 하나도 처리하기 전에 멈춘다.

    d01_kitchen.mov 와 d01_desk.mov 가 같이 있으면 둘째가 "이미 있음"으로 건너뛰어져
    한 영상이 통째로 빠진다. 그래서 먼저 다 확인한다.
    """
    videos = sorted(p for p in videos_dir.glob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    seen: dict[str, Path] = {}
    for path in videos:
        video = video_id(path)
        if video in seen:
            raise ValueError(f"영상 번호 {video} 가 겹칩니다: {seen[video].name}, {path.name}")
        seen[video] = path
    return videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="영상 → 학습용 사진 (1초에 1장, 중복 거르기)")
    parser.add_argument(
        "--root",
        type=Path,
        default=config.DATASETS_DIR / "detector",
        help="검출기 데이터 폴더 (videos/ 를 읽고 frames/ 에 쓴다)",
    )
    parser.add_argument("--every", type=float, default=1.0, help="몇 초에 한 장")
    parser.add_argument(
        "--max-distance", type=int, default=MAX_DISTANCE, help="이 이하로 다르면 같은 장면(0~64)"
    )
    args = parser.parse_args()
    if args.every <= 0:
        parser.error(f"--every 는 0 보다 커야 합니다: {args.every}")
    if not 0 <= args.max_distance <= 64:
        parser.error(f"--max-distance 는 0~64 입니다: {args.max_distance}")
    return args


def main() -> None:
    args = parse_args()
    videos_dir = args.root / "videos"
    try:
        videos = find_videos(videos_dir)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if not videos:
        print(f"영상이 없습니다: {videos_dir}")
        return

    for path in videos:
        video = video_id(path)
        if (args.root / "frames" / video).exists():
            print(f"{path.name}: 이미 뽑았습니다. 건너뜀 (다시 하려면 frames/{video} 폴더를 지움)")
            continue
        result = extract(path, args.root, args.every, args.max_distance)
        if result is None:
            print(
                f"{path.name}: 영상을 열지 못했습니다. "
                "QuickTime 파일 → 내보내기 → 1080p 로 다시 저장해 보세요."
            )
            continue
        rows, sampled, seconds = result
        append_rows(args.root / "frames.csv", rows)
        print(
            f"{path.name}: {seconds:.0f}초 · 뽑은 {sampled}장 · "
            f"버린 {sampled - len(rows)}장 · 남은 {len(rows)}장"
        )


if __name__ == "__main__":
    main()
