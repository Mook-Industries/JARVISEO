"""손 영상 → 손 모양 구간 · 손 관절 표 · 손 사진 — 가리키는 손 분류기 학습 데이터.  담당: 최홍묵

가리키는 손 분류기(#223)의 첫 단계. 지금 "가리키는 손" 판정은 검지만 펴고 나머지를
접었는지 보는 15° 굽힘 규칙(R0, src/jarviseo/pointing/hands.py)이다. 손 영상 5개로 미리 재 보니
주먹 60% · 다른 손 모양 51% · 물건 든 손 29% 를 가리킴으로 읽었다.
그래서 손 모양 이름이 붙은 영상으로 분류기를 학습한다.

영상 하나 = 한 사람이 한 장소에서 손 모양 6가지를 정해진 순서로 20초씩 보여 준 것이다.
순서: point(가리킴) → hold(물건 쥐기) → open(편 손) → fist(주먹) → rest(쉬는 손)
→ other(브이 · 엄지척 등). 모양 사이마다 손바닥으로 렌즈를 2초 가렸다.
그 까만 구간을 찾아 영상을 6구간으로 자른다.

파일 이름에 장소 · 사람을 남긴다(h_hall_p2.mov). 학습 / 시험은 사람 단위로 나눈다.
같은 사람 손은 장소가 달라도 모양 버릇이 같아서, 섞으면 처음 보는 사람 손에서의
실력보다 점수가 부풀려진다.

실행 (레포 맨 위 폴더에서):
    python tools/hand_clips.py segments   # 1. 렌즈 가림으로 6구간 → segments.csv
    python tools/hand_clips.py extract    # 2. 구간마다 0.2초 간격으로 손 관절 표 + 손 사진
    python tools/hand_clips.py summary    # 3. 사람 · 장소 · 손 모양별 손 찾은 비율 · R0 점수

저장 (data/datasets/hand/):
    segments.csv     video · place · person · cls · start_sec · end_sec · source(auto / manual)
    review/<영상>.jpg 자동으로 못 자른 영상의 2초 간격 모음 사진.
                     보고 segments.csv 에 source=manual 로 6줄을 적는다
    landmarks.csv    한 줄 = 프레임 하나의 손 하나 (손이 없으면 n_hands 0 인 한 줄).
                     열은 tools/README.md "가리키는 손 분류기 학습 데이터"
    crops/<영상>/     손 사진 224×224, <시각 ms>_<손 번호>.jpg
"""

import argparse
import csv
import math
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from jarviseo import config
from jarviseo.pointing.hands import (
    NUM_LANDMARKS,
    HandDetector,
    HandPose,
    bend_gap_deg,
    hand_patch,
    is_pointing,
)

# 촬영 순서와 같아야 한다. 구간 이름을 이 순서로 붙인다.
CLASSES = ("point", "hold", "open", "fist", "rest", "other")
POSITIVE = "point"
VIDEO_SUFFIXES = (".mov", ".mp4")
# h_hall_p2 — 장소(영어 소문자) + 사람(p + 숫자)
VIDEO_NAME = re.compile(r"h_([a-z]+)_(p\d+)")

# 렌즈 가림 찾기. 회색 평균 밝기가 이 아래로 이어지는 구간을 모은다.
# 사전 측정에서 진짜 가림은 최저 3~29, 쉬는 손 장면의 어두운 바닥은 33~40 이었다.
# loby p1 은 열로 가림이 29 까지 밝아져서 고정 기준(30 미만)으로는 놓친다.
# 그래서 기준은 넉넉히 두고, 그 아래 구간 중 가장 어두운 5개를 고른다.
DARK_LEVEL = 40.0
# 이보다 짧으면 깜빡임. hall p3 에 0.6초짜리 가림이 있다.
MIN_COVER_SEC = 0.4
NUM_COVERS = len(CLASSES) - 1
STEP_SEC = 0.2
# 구간 양 끝은 손을 렌즈에서 떼고 자세를 잡는 중이라 버린다.
TRIM_SEC = 0.5
# 한 모양을 20초씩 찍었다. 이 밖이면 가림을 잘못 고른 것이다(hall p1 은 고르면 51초 구간이 생긴다).
SEGMENT_MIN_SEC, SEGMENT_MAX_SEC = 8.0, 40.0
# 모음 사진 간격. 가림이 2초라 4초 간격이면 가림을 통째로 건너뛸 수 있다.
SHEET_EVERY_SEC = 2.0
SHEET_TILE = (192, 108)
SHEET_COLUMNS = 10
JPEG_QUALITY = 95

SEGMENTS_COLUMNS = ["video", "place", "person", "cls", "start_sec", "end_sec", "source"]
WORLD_COLUMNS = [f"w{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in "xyz"]
PIXEL_COLUMNS = [f"p{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in "xy"]
LANDMARK_COLUMNS = [
    "video",
    "place",
    "person",
    "cls",
    "t_sec",
    "hand_idx",
    "n_hands",
    "handedness",
    "hand_score",
    "label",
    "rule",
    "bend_gap",
    *WORLD_COLUMNS,
    *PIXEL_COLUMNS,
    "crop",
]

Cover = tuple[float, float]
Segment = tuple[str, float, float]


def clip_info(path: Path) -> tuple[str, str]:
    """영상 파일 이름에서 (장소, 사람)을 꺼낸다. 예: h_hall_p2.mov → ("hall", "p2")

    사람 단위 나누기가 이 두 값에 기대므로, 규칙에 안 맞는 이름은 조용히 넘기지 않고 멈춘다.
    """
    match = VIDEO_NAME.fullmatch(path.stem)
    if match is None:
        raise ValueError(
            f"손 영상 이름은 h_hall_p2 처럼 짓습니다(장소 영어 소문자 · 사람 p + 숫자): {path.name}"
        )
    return match.group(1), match.group(2)


def dark_runs(
    times: Sequence[float], brightness: Sequence[float], dark: float = DARK_LEVEL
) -> list[tuple[float, float, float]]:
    """밝기가 dark 아래로 이어진 구간들. (첫 까만 시각, 다시 밝아진 시각, 구간 최저 밝기)

    끝을 "다시 밝아진 시각"으로 둬서 0.2초 간격 2장이 까마면 0.4초가 된다.
    영상 끝까지 까마면 끝은 마지막 시각이다.
    """
    runs: list[tuple[float, float, float]] = []
    start: float | None = None
    lowest = math.inf
    for t, value in zip(times, brightness, strict=True):
        if value < dark:
            if start is None:
                start, lowest = t, value
            lowest = min(lowest, value)
        elif start is not None:
            runs.append((start, t, lowest))
            start = None
    if start is not None:
        runs.append((start, times[-1], lowest))
    return runs


def find_covers(
    times: Sequence[float],
    brightness: Sequence[float],
    count: int = NUM_COVERS,
    min_sec: float = MIN_COVER_SEC,
) -> list[Cover] | None:
    """렌즈를 가린 구간 count 개를 시간순으로. 모자라면 None.

    min_sec 보다 짧은 구간(깜빡임)은 버리고, 남은 것 중 가장 어두운 count 개를 고른다.
    쉬는 손 장면의 어두운 바닥도 기준 아래로 내려가지만 진짜 가림보다는 밝다.
    """
    runs = [run for run in dark_runs(times, brightness) if run[1] - run[0] >= min_sec]
    if len(runs) < count:
        return None
    darkest = sorted(runs, key=lambda run: run[2])[:count]
    return sorted((start, end) for start, end, _ in darkest)


def segments_from_covers(covers: Sequence[Cover], end_sec: float) -> list[Segment]:
    """가림 사이 구간에 촬영 순서대로 이름을 붙인다. 첫 구간은 0초부터, 마지막은 영상 끝까지."""
    bounds = [0.0, *[t for cover in covers for t in cover], end_sec]
    return [(cls, bounds[2 * i], bounds[2 * i + 1]) for i, cls in enumerate(CLASSES)]


def segments_look_right(segments: Sequence[Segment]) -> bool:
    """구간이 6개이고 모두 8~40초면 True.

    가림을 잘못 고르면 두 모양이 한 구간에 섞인 채 조용히 넘어가서, 길이로 한 번 거른다.
    """
    return len(segments) == len(CLASSES) and all(
        SEGMENT_MIN_SEC <= end - start <= SEGMENT_MAX_SEC for _, start, end in segments
    )


def segment_at(segments: Sequence[Segment], t_sec: float, trim: float = TRIM_SEC) -> str | None:
    """이 시각이 어느 구간 안쪽(양 끝 trim 초를 뺀)인지. 가림 · 구간 끝이면 None."""
    for cls, start, end in segments:
        if start + trim <= t_sec <= end - trim:
            return cls
    return None


def label_for(cls: str, n_hands: int) -> int | None:
    """학습 이름표. 가리킴 구간의 손 하나 = 1, 다른 구간의 손 = 0, 뺄 손 = None.

    가리킴 구간에 손이 둘 이상이면 어느 쪽이 가리킨 손인지 몰라서 뺀다.
    둘 다 1로 두면 화면에 걸린 쉬는 손을 "가리킴"으로 가르친다.
    """
    if n_hands == 0:
        return None
    if cls == POSITIVE:
        return 1 if n_hands == 1 else None
    return 0


def hand_values(hand: HandPose) -> dict[str, object]:
    """손 하나의 표 칸: 왼손/오른손 · 15° 규칙 판정 · 굽힘 차이 · 관절 (3D 63칸 + 화면 42칸)."""
    values: dict[str, object] = {
        "handedness": hand.handedness,
        "hand_score": round(hand.score, 4),
        "rule": int(is_pointing(hand)),
        "bend_gap": round(bend_gap_deg(hand), 2),
    }
    world = [round(v, 5) for point in hand.world for v in point]
    pixel = [round(v, 1) for point in hand.landmarks for v in point]
    values.update(zip(WORLD_COLUMNS, world, strict=True))
    values.update(zip(PIXEL_COLUMNS, pixel, strict=True))
    return values


def crop_name(t_sec: float, hand_idx: int) -> str:
    """손 사진 파일 이름. 예: (12.4, 0) → 012400_0.jpg (시각 ms 여섯 자리라 이름순 = 시간순)"""
    return f"{round(t_sec * 1000):06d}_{hand_idx}.jpg"


def sampled_frames(
    video_path: Path, every_sec: float, retrieve: Callable[[float], bool] = lambda t: True
) -> Iterator[tuple[float, np.ndarray | None]]:
    """영상을 처음부터 읽으며 every_sec 칸마다 한 장씩 (시각, 프레임)을 낸다.

    retrieve(시각)가 False 인 칸은 프레임을 풀지 않고 None 을 낸다(가림 구간 등, 빨리 넘기려고).
    프레임 번호가 아니라 영상 시각으로 센다. QuickTime 녹화는 초당 장 수가 일정하지 않을 수 있다
    (extract_frames.FrameSampler 와 같은 방식).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"영상을 열지 못했습니다: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target, index = 0.0, 0
    try:
        while cap.grab():
            t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            # 백엔드에 따라 시각이 0 으로만 나온다. 그때는 프레임 번호 ÷ FPS 로 대신한다.
            if t_sec <= 0 and index > 0:
                t_sec = index / fps
            index += 1
            if t_sec < target:
                continue
            target = (t_sec // every_sec + 1) * every_sec
            if not retrieve(t_sec):
                yield t_sec, None
                continue
            ok, frame = cap.retrieve()
            if ok:
                yield t_sec, frame
    finally:
        cap.release()


def brightness(frame: np.ndarray) -> float:
    """회색 평균 밝기 0~255. 가림은 화면 전체가 어두워지는 것이라 320×180 으로 줄여 잰다."""
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def contact_sheet(tiles: Sequence[tuple[float, float, np.ndarray]]) -> np.ndarray:
    """(시각, 밝기, 프레임)들을 한 장에. 칸마다 왼쪽 위에 '시각s 밝기'를 적는다."""
    tile_w, tile_h = SHEET_TILE
    rows = math.ceil(len(tiles) / SHEET_COLUMNS)
    sheet = np.zeros((rows * tile_h, SHEET_COLUMNS * tile_w, 3), np.uint8)
    for i, (t_sec, value, frame) in enumerate(tiles):
        tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        text = f"{t_sec:.1f}s {value:.0f}"
        cv2.putText(tile, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(tile, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y, x = (i // SHEET_COLUMNS) * tile_h, (i % SHEET_COLUMNS) * tile_w
        sheet[y : y + tile_h, x : x + tile_w] = tile
    return sheet


def save_jpg(path: Path, image: np.ndarray) -> None:
    """jpg 로 저장한다. cv2.imwrite 는 윈도우에서 경로에 한글이 있으면 에러 없이 실패해서
    메모리에서 인코딩한 뒤 파이썬이 쓴다(extract_frames.py 와 같다)."""
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"jpg 인코딩에 실패했습니다: {path}")
    buffer.tofile(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path, columns: list[str], rows: Iterable[dict[str, object]], append: bool = False
) -> None:
    """encoding 을 빼면 윈도우에서 cp949 로 저장되고, newline="" 을 빼면 빈 줄이 끼인다."""
    is_new = not append or not path.exists()
    with path.open("a" if append else "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def find_videos(videos_dir: Path) -> list[Path]:
    """영상 목록. 이름 규칙이 틀린 영상이 하나라도 있으면 아무것도 하기 전에 멈춘다."""
    videos = sorted(p for p in videos_dir.glob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    for path in videos:
        clip_info(path)
    return videos


def segment_rows(video: str, segments: Sequence[Segment], source: str) -> list[dict[str, object]]:
    place, person = clip_info(Path(video))
    return [
        {
            "video": video,
            "place": place,
            "person": person,
            "cls": cls,
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "source": source,
        }
        for cls, start, end in segments
    ]


def manual_videos(rows: Iterable[dict[str, str]]) -> set[str]:
    """손으로 적은 줄이 있는 영상. segments 를 다시 돌려도 이 영상은 덮어쓰지 않는다."""
    return {row["video"] for row in rows if row["source"] == "manual"}


def segments_by_video(rows: Iterable[dict[str, str]]) -> dict[str, list[Segment]]:
    """segments.csv 줄 → 영상별 (손 모양, 시작, 끝). 손으로 적은 영상이 6줄이 아니면 멈춘다."""
    by_video: dict[str, list[Segment]] = {}
    for row in rows:
        by_video.setdefault(row["video"], []).append(
            (row["cls"], float(row["start_sec"]), float(row["end_sec"]))
        )
    for video, segments in by_video.items():
        if sorted(cls for cls, _, _ in segments) != sorted(CLASSES):
            raise ValueError(f"{video}: segments.csv 에 손 모양 6가지가 한 줄씩 있어야 합니다")
    return by_video


def run_segments(root: Path) -> None:
    videos = find_videos(root / "videos")
    if not videos:
        print(f"영상이 없습니다: {root / 'videos'}")
        return
    csv_path = root / "segments.csv"
    old = read_csv(csv_path) if csv_path.exists() else []
    keep = manual_videos(old)
    rows: list[dict[str, object]] = [row for row in old if row["video"] in keep]
    for path in videos:
        if path.stem in keep:
            print(f"{path.name}: 손으로 적은 구간이 있어 그대로 둡니다")
            continue
        times: list[float] = []
        values: list[float] = []
        tiles: list[tuple[float, float, np.ndarray]] = []
        next_tile = 0.0
        for t_sec, frame in sampled_frames(path, STEP_SEC):
            value = brightness(frame)
            times.append(t_sec)
            values.append(value)
            if t_sec >= next_tile:
                next_tile += SHEET_EVERY_SEC
                tiles.append((t_sec, value, cv2.resize(frame, SHEET_TILE)))
        covers = find_covers(times, values)
        segments = segments_from_covers(covers, times[-1]) if covers else []
        if segments_look_right(segments):
            rows += segment_rows(path.stem, segments, "auto")
            lengths = " · ".join(f"{cls} {end - start:.1f}" for cls, start, end in segments)
            print(f"{path.name}: 자동 ({lengths}초)")
            continue
        review = root / "review"
        review.mkdir(exist_ok=True)
        save_jpg(review / f"{path.stem}.jpg", contact_sheet(tiles))
        runs = [run for run in dark_runs(times, values) if run[1] - run[0] >= MIN_COVER_SEC]
        found = " · ".join(f"{s:.1f}~{e:.1f}(최저 {low:.0f})" for s, e, low in runs)
        print(
            f"{path.name}: 자동으로 못 잘랐습니다 — 까만 구간 {len(runs)}개 {found}\n"
            f"  review/{path.stem}.jpg 를 보고 segments.csv 에 source=manual 로 6줄을 적으세요"
        )
    order = {cls: i for i, cls in enumerate(CLASSES)}
    rows.sort(key=lambda row: (row["video"], order[str(row["cls"])]))
    write_csv(csv_path, SEGMENTS_COLUMNS, rows)
    print(f"저장: {csv_path} ({len({row['video'] for row in rows})}/{len(videos)}개 영상)")


def run_extract(root: Path, detector: HandDetector) -> None:
    csv_path = root / "landmarks.csv"
    if csv_path.exists():
        raise SystemExit(f"이미 있습니다: {csv_path} — 다시 뽑으려면 이 파일과 crops/ 를 지우세요")
    by_video = segments_by_video(read_csv(root / "segments.csv"))
    paths = {p.stem: p for p in find_videos(root / "videos")}
    for video, segments in sorted(by_video.items()):
        path = paths[video]
        place, person = clip_info(path)
        crops_dir = root / "crops" / video
        crops_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        frames = 0
        found = 0
        sampled = sampled_frames(
            path, STEP_SEC, retrieve=lambda t, segs=segments: segment_at(segs, t) is not None
        )
        for t_sec, frame in sampled:
            cls = segment_at(segments, t_sec)
            if frame is None or cls is None:
                continue
            frames += 1
            hands = detector.detect(frame)
            found += bool(hands)
            base = {"video": video, "place": place, "person": person, "cls": cls}
            base.update({"t_sec": round(t_sec, 2), "n_hands": len(hands)})
            if not hands:
                rows.append(base)  # 손을 못 찾은 프레임도 남겨 손 검출률을 같은 표에서 센다
                continue
            for i, hand in enumerate(hands):
                name = crop_name(t_sec, i)
                # 실행 때 R2 분류기가 자르는 손 사진과 같은 함수(pointing/hands.py)로 자른다
                save_jpg(crops_dir / name, hand_patch(frame, hand))
                row = {
                    **base,
                    "hand_idx": i,
                    "label": label_for(cls, len(hands)),
                    **hand_values(hand),
                }
                rows.append({**row, "crop": f"{video}/{name}"})
        write_csv(csv_path, LANDMARK_COLUMNS, rows, append=True)
        rate = found / max(frames, 1)
        print(f"{video}: 프레임 {frames} · 손 찾음 {found} ({rate:.0%}) · 줄 {len(rows)}")


def summarize(rows: Iterable[dict[str, str]], key: str | None) -> dict[str, dict[str, float]]:
    """key 열(None 이면 전체)별 프레임 · 손 찾은 프레임 · 이름표 개수 · 15° 규칙 점수.

    프레임은 (영상, 시각)으로 센다. 손이 둘인 프레임은 줄이 둘이라 줄 수로 세면 안 된다.
    15° 규칙 점수는 이름표가 있는 손(뺀 손 제외)에서 잰다.
    가리킴 재현율 = 가리킨 손 중 규칙이 맞힌 비율,
    오탐률 = 가리킴이 아닌 손 중 규칙이 가리킴으로 읽은 비율.
    """
    groups: dict[str, dict[str, float]] = {}
    seen: dict[str, set[tuple[str, str]]] = {}
    hands_seen: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        name = "전체" if key is None else row[key]
        stat = groups.setdefault(name, dict.fromkeys(("tp", "fn", "fp", "tn", "dropped"), 0))
        frame = (row["video"], row["t_sec"])
        seen.setdefault(name, set()).add(frame)
        if int(row["n_hands"]) == 0:
            continue
        hands_seen.setdefault(name, set()).add(frame)
        if row["label"] == "":
            stat["dropped"] += 1
            continue
        label, rule = int(row["label"]), int(row["rule"])
        stat[("tp" if rule else "fn") if label else ("fp" if rule else "tn")] += 1
    for name, stat in groups.items():
        positives, negatives = stat["tp"] + stat["fn"], stat["fp"] + stat["tn"]
        stat["frames"] = len(seen[name])
        stat["found"] = len(hands_seen.get(name, ()))
        stat["accuracy"] = (stat["tp"] + stat["tn"]) / max(positives + negatives, 1)
        stat["recall"] = stat["tp"] / max(positives, 1)
        stat["false_alarm"] = stat["fp"] / max(negatives, 1)
    return groups


def run_summary(root: Path) -> None:
    rows = read_csv(root / "landmarks.csv")
    for key, title in ((None, "전체"), ("person", "사람"), ("place", "장소"), ("cls", "손 모양")):
        print(f"\n[{title}]  프레임 · 손 찾음 · 이름표 1 / 0 / 뺌 · R0 정확도 · 재현율 · 오탐률")
        for name, s in sorted(summarize(rows, key).items()):
            print(
                f"  {name:10} {s['frames']:5.0f} · {s['found'] / max(s['frames'], 1):4.0%} · "
                f"{s['tp'] + s['fn']:4.0f} / {s['fp'] + s['tn']:4.0f} / {s['dropped']:3.0f} · "
                f"{s['accuracy']:5.1%} · {s['recall']:5.1%} · {s['false_alarm']:5.1%}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="손 영상 → 손 모양 구간 · 손 관절 표 · 손 사진")
    parser.add_argument("command", choices=("segments", "extract", "summary"))
    parser.add_argument(
        "--root",
        type=Path,
        default=config.DATASETS_DIR / "hand",
        help="손 데이터 폴더 (videos/ 를 읽고 segments.csv · landmarks.csv · crops/ 를 쓴다)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "segments":
            run_segments(args.root)
        elif args.command == "extract":
            run_extract(args.root, HandDetector(encoder=None))
        else:
            run_summary(args.root)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
