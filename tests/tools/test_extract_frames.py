"""tools/extract_frames.py — 영상 없이 검사할 수 있는 부분과, 작은 합성 영상으로 한 바퀴.

틀리면 학습 데이터가 조용히 망가지는 곳을 잡는다 — 영상 번호(분할의 근거) · 1초 간격 ·
중복 거르기(마지막으로 남긴 장과 비교) · frames.csv · 이름이 겹치는 영상.

tools/ 는 패키지가 아니라서 파일 경로로 직접 불러온다.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "extract_frames.py"
_spec = importlib.util.spec_from_file_location("extract_frames", _TOOL_PATH)
extract_frames = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_frames)


def _set_argv(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["extract_frames.py", *args])


def _gradient(width: int = 90, height: int = 80, reverse: bool = False) -> np.ndarray:
    """가로로 밝아지는(reverse 면 어두워지는) 회색 BGR 사진."""
    row = np.linspace(0, 255, width).astype(np.uint8)
    if reverse:
        row = row[::-1]
    return np.repeat(np.tile(row, (height, 1))[:, :, None], 3, axis=2)


# --- 영상 번호 -------------------------------------------------------------------


def test_video_id_takes_number_from_name():
    """d01_kitchen.mov → d01. 분할은 이 번호만 본다."""
    assert extract_frames.video_id(Path("d01_kitchen.mov")) == "d01"
    assert extract_frames.video_id(Path("videos/d12_study.mp4")) == "d12"


@pytest.mark.parametrize(
    "name", ["kitchen.mov", "d1_kitchen.mov", "d01-kitchen.mov", "d01_desk2.mov", "D01_desk.mov"]
)
def test_video_id_rejects_names_off_the_rule(name):
    """규칙에 안 맞는 이름은 조용히 넘기지 않고 멈춘다."""
    with pytest.raises(ValueError, match="d01_kitchen"):
        extract_frames.video_id(Path(name))


def test_find_videos_rejects_duplicate_numbers(tmp_path):
    """같은 번호가 둘이면 하나가 통째로 빠지므로, 처리하기 전에 멈춘다."""
    (tmp_path / "d01_kitchen.mov").touch()
    (tmp_path / "d01_desk.mov").touch()
    with pytest.raises(ValueError, match="d01"):
        extract_frames.find_videos(tmp_path)


def test_find_videos_lists_only_videos_in_order(tmp_path):
    """영상만 번호 순서로. 메모 파일 등은 무시한다."""
    for name in ("d02_desk.mov", "d01_kitchen.MP4", "memo.txt"):
        (tmp_path / name).touch()
    assert [p.name for p in extract_frames.find_videos(tmp_path)] == [
        "d01_kitchen.MP4",
        "d02_desk.mov",
    ]


# --- 1초 간격 ---------------------------------------------------------------------


def test_sampler_takes_one_frame_per_second():
    """30fps 3초 영상에서 0 · 1 · 2 · 3초 근처의 4장만 고른다."""
    sampler = extract_frames.FrameSampler(1.0)
    times = [i / 30 for i in range(91)]
    picked = [t for t in times if sampler.take(t)]
    assert len(picked) == 4
    assert picked[0] == 0.0


def test_sampler_follows_time_not_frame_count():
    """초당 장 수가 들쭉날쭉해도 시각으로 고른다."""
    sampler = extract_frames.FrameSampler(1.0)
    times = [0.0, 0.2, 0.95, 1.01, 1.02, 1.9, 2.05]
    assert [t for t in times if sampler.take(t)] == [0.0, 1.01, 2.05]


def test_sampler_takes_one_frame_after_a_gap():
    """끊겨서 여러 칸을 건너뛰어도 한 장만 고르고, 다음 칸부터 다시 센다."""
    sampler = extract_frames.FrameSampler(1.0)
    times = [0.0, 0.5, 3.4, 3.6, 4.1]
    assert [t for t in times if sampler.take(t)] == [0.0, 3.4, 4.1]


# --- 중복 거르기 ------------------------------------------------------------------


def test_dhash_same_scene_with_brighter_light_is_close():
    """같은 장면이 조금 밝아져도 지문이 거의 그대로다."""
    image = _gradient()
    brighter = cv2.add(image, np.full_like(image, 10))
    assert extract_frames.hamming(extract_frames.dhash(image), extract_frames.dhash(brighter)) <= 2


def test_dhash_different_scenes_are_far():
    """밝아지는 방향이 반대인 그림은 64비트가 다 다르다."""
    a = extract_frames.dhash(_gradient())
    b = extract_frames.dhash(_gradient(reverse=True))
    assert extract_frames.hamming(a, b) == 64


def test_dhash_accepts_gray_image():
    """흑백 사진도 받는다."""
    gray = cv2.cvtColor(_gradient(), cv2.COLOR_BGR2GRAY)
    assert extract_frames.dhash(gray) == extract_frames.dhash(_gradient())


def test_deduper_compares_with_last_kept_not_previous():
    """천천히 바뀌는 장면: 앞 장과는 늘 2비트 차이지만, 남긴 장과 차이가 쌓이면 남긴다.

    바로 앞 장과 비교했다면 0번만 남고 장면이 크게 바뀐 뒤에도 계속 버렸을 것이다.
    """
    deduper = extract_frames.Deduper(max_distance=6)
    # k 번째 지문은 0번에서 2k 비트가 다르다
    hashes = [(1 << (2 * k)) - 1 for k in range(7)]
    kept = [i for i, h in enumerate(hashes) if deduper.is_new(h)]
    assert kept == [0, 4]


def test_deduper_keeps_first_frame():
    assert extract_frames.Deduper(max_distance=64).is_new(0)


# --- 파일 · 기록 ------------------------------------------------------------------


def test_frame_name_keeps_video_number():
    assert extract_frames.frame_name("d03", 7) == "d03_0007.jpg"


def test_append_rows_writes_header_once(tmp_path):
    """두 영상을 차례로 덧붙여도 헤더는 한 줄. 한글이 섞여도 utf-8 로 읽힌다."""
    path = tmp_path / "frames.csv"
    first = {"file": "frames/d01/d01_0001.jpg", "video": "d01", "t_sec": 0}
    second = {"file": "frames/d02/한글.jpg", "video": "d02", "t_sec": 1.5}
    extract_frames.append_rows(path, [first])
    extract_frames.append_rows(path, [second])
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["video"] for r in rows] == ["d01", "d02"]
    assert rows[1]["file"] == "frames/d02/한글.jpg"


# --- 인자 -------------------------------------------------------------------------


@pytest.mark.parametrize("args", [["--every", "0"], ["--max-distance", "65"]])
def test_parse_args_rejects_bad_values(monkeypatch, args):
    _set_argv(monkeypatch, *args)
    with pytest.raises(SystemExit):
        extract_frames.parse_args()


def test_main_without_videos_says_so(monkeypatch, tmp_path, capsys):
    _set_argv(monkeypatch, "--root", str(tmp_path))
    extract_frames.main()
    assert "영상이 없습니다" in capsys.readouterr().out


def test_main_stops_on_bad_name_before_any_work(monkeypatch, tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "kitchen.mov").touch()
    _set_argv(monkeypatch, "--root", str(tmp_path))
    with pytest.raises(SystemExit, match="d01_kitchen"):
        extract_frames.main()
    assert not (tmp_path / "frames").exists()


# --- 합성 영상으로 한 바퀴 ----------------------------------------------------------


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> bool:
    """작은 mp4 를 만든다. 이 OS 의 OpenCV 가 mp4 를 못 쓰면 False."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        return False
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path.exists() and path.stat().st_size > 0


def test_main_extracts_one_per_second_and_drops_same_scene(monkeypatch, tmp_path, capsys):
    """10fps 4초: 0~2초는 같은 장면, 2초부터 다른 장면 → 0초 · 2초 두 장만 남는다."""
    videos = tmp_path / "videos"
    videos.mkdir()
    still, other = _gradient(160, 120), _gradient(160, 120, reverse=True)
    if not _write_video(videos / "d01_kitchen.mp4", [still] * 20 + [other] * 20, fps=10):
        pytest.skip("이 OpenCV 는 mp4 를 쓰지 못한다")
    _set_argv(monkeypatch, "--root", str(tmp_path))
    extract_frames.main()

    assert sorted(p.name for p in (tmp_path / "frames" / "d01").iterdir()) == [
        "d01_0001.jpg",
        "d01_0002.jpg",
    ]
    with (tmp_path / "frames.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["file"] for r in rows] == ["frames/d01/d01_0001.jpg", "frames/d01/d01_0002.jpg"]
    assert float(rows[1]["t_sec"]) == pytest.approx(2.0, abs=0.15)
    assert "뽑은 4장 · 버린 2장 · 남은 2장" in capsys.readouterr().out

    # 다시 돌리면 이미 뽑은 영상은 건너뛰고 frames.csv 에 줄을 더하지 않는다
    extract_frames.main()
    assert "건너뜀" in capsys.readouterr().out
    with (tmp_path / "frames.csv").open(encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_extract_returns_none_for_unreadable_video(tmp_path):
    """영상이 아닌 파일은 None 이고 빈 폴더를 남기지 않는다."""
    path = tmp_path / "d01_kitchen.mov"
    path.write_bytes(b"not a video")
    assert extract_frames.extract(path, tmp_path, 1.0, 6) is None
    assert not (tmp_path / "frames" / "d01").exists()
