"""tools/hand_clips.py — 영상 · MediaPipe 없이 검사할 수 있는 부분과, 작은 합성 영상으로 한 바퀴.

틀리면 손 분류기 학습 데이터가 조용히 망가지는 곳을 잡는다 — 이름에서 사람 꺼내기(나누기의 근거) ·
렌즈 가림 고르기(어두운 바닥 · 깜빡임 거르기) · 구간 이름 순서 · 이상한 구간 길이 · 이름표 ·
손으로 적은 구간 지키기 · 프레임 수 세기.

tools/ 는 패키지가 아니라서 파일 경로로 직접 불러온다.
"""

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from jarviseo.pointing.hands import HandPose

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "hand_clips.py"
_spec = importlib.util.spec_from_file_location("hand_clips", _TOOL_PATH)
hand_clips = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hand_clips)


def _series(spans: list[tuple[float, float, float]], end: float, base: float = 150.0):
    """0.2초 간격 밝기 줄. spans 의 (시작, 끝, 밝기) 동안만 그 밝기, 나머지는 base."""
    times = [round(i * 0.2, 1) for i in range(int(end / 0.2) + 1)]
    values = [next((v for s, e, v in spans if s <= t < e), base) for t in times]
    return times, values


FIVE_COVERS = [(20, 22, 5), (42, 44, 8), (64, 66, 10), (86, 88, 12), (108, 110, 20)]


# --- 영상 이름 --------------------------------------------------------------------


def test_clip_info_takes_place_and_person():
    """h_hall_p2.mov → (hall, p2). 사람 단위 나누기는 이 값만 본다."""
    assert hand_clips.clip_info(Path("h_hall_p2.mov")) == ("hall", "p2")


@pytest.mark.parametrize(
    "name", ["hall_p2.mov", "h_Hall_p2.mov", "h_hall_2.mov", "h_hall_p2_x.mov"]
)
def test_clip_info_stops_on_bad_name(name):
    """규칙 밖 이름은 조용히 넘기지 않고 멈춘다."""
    with pytest.raises(ValueError, match="h_hall_p2"):
        hand_clips.clip_info(Path(name))


# --- 렌즈 가림 -------------------------------------------------------------------


def test_find_covers_returns_five_covers_in_time_order():
    """가림 5개면 그대로, 시간순으로. 끝은 다시 밝아진 시각이다."""
    times, values = _series(FIVE_COVERS, 130)
    covers = hand_clips.find_covers(times, values)
    assert covers == [(20.0, 22.0), (42.0, 44.0), (64.0, 66.0), (86.0, 88.0), (108.0, 110.0)]


def test_find_covers_skips_dim_floor_that_is_brighter_than_real_covers():
    """쉬는 손 장면의 어두운 바닥(36)은 기준(40) 아래여도 진짜 가림보다 밝아 빠진다."""
    times, values = _series([*FIVE_COVERS, (95, 99, 36)], 130)
    assert (95.0, 99.0) not in hand_clips.find_covers(times, values)
    assert len(hand_clips.find_covers(times, values)) == 5


def test_find_covers_drops_short_blink():
    """0.2초 깜빡임은 아무리 까매도 가림이 아니다."""
    times, values = _series([*FIVE_COVERS, (30, 30.2, 0)], 130)
    assert (30.0, 30.2) not in hand_clips.find_covers(times, values)


def test_find_covers_none_when_not_enough():
    """가림이 4개뿐이면(살색으로 가린 영상) None → 손으로 적는다."""
    times, values = _series(FIVE_COVERS[:4], 130)
    assert hand_clips.find_covers(times, values) is None


def test_dark_run_at_video_end_ends_at_last_time():
    """영상 끝까지 까마면 끝은 마지막 시각."""
    times, values = _series([(9, 11, 5)], 10)
    assert hand_clips.dark_runs(times, values) == [(9.0, 10.0, 5)]


# --- 구간 ------------------------------------------------------------------------


def test_segments_from_covers_names_in_shooting_order():
    """가림 사이 6구간에 point → hold → open → fist → rest → other."""
    covers = [(20, 22), (42, 44), (64, 66), (86, 88), (108, 110)]
    segments = hand_clips.segments_from_covers(covers, 130)
    assert segments[0] == ("point", 0.0, 20)
    assert segments[1] == ("hold", 22, 42)
    assert segments[-1] == ("other", 110, 130)
    assert [cls for cls, _, _ in segments] == list(hand_clips.CLASSES)


def test_segments_look_right_rejects_too_long_segment():
    """가림 하나를 놓쳐 두 모양이 한 구간(51초)에 섞이면 False."""
    good = hand_clips.segments_from_covers(
        [(20, 22), (42, 44), (64, 66), (86, 88), (108, 110)], 130
    )
    assert hand_clips.segments_look_right(good)
    bad = hand_clips.segments_from_covers(
        [(20, 22), (73, 75), (95, 97), (117, 119), (140, 142)], 160
    )
    assert not hand_clips.segments_look_right(bad)


def test_segment_at_trims_both_ends():
    """구간 양 끝 0.5초와 가림 안은 None."""
    segments = [("point", 0.0, 20.0), ("hold", 22.0, 42.0)]
    assert hand_clips.segment_at(segments, 0.3) is None
    assert hand_clips.segment_at(segments, 10.0) == "point"
    assert hand_clips.segment_at(segments, 19.8) is None
    assert hand_clips.segment_at(segments, 21.0) is None
    assert hand_clips.segment_at(segments, 22.5) == "hold"


# --- 이름표 -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "n_hands", "expected"),
    [("point", 1, 1), ("point", 2, None), ("fist", 2, 0), ("hold", 1, 0), ("point", 0, None)],
)
def test_label_for(cls, n_hands, expected):
    """가리킴 구간의 손 하나만 1. 손이 둘이면 어느 손이 가리켰는지 몰라 뺀다."""
    assert hand_clips.label_for(cls, n_hands) == expected


# --- 표 칸 ------------------------------------------------------------------------


def _hand() -> HandPose:
    landmarks = tuple((float(i), float(2 * i)) for i in range(21))
    world = tuple((0.01 * i, 0.0, 0.0) for i in range(21))
    return HandPose(landmarks=landmarks, world=world, handedness="Right", score=0.9)


def test_hand_values_fills_every_landmark_column():
    """3D 63칸 + 화면 42칸이 모두 차고, 규칙 판정이 0/1 이다."""
    values = hand_clips.hand_values(_hand())
    for column in [*hand_clips.WORLD_COLUMNS, *hand_clips.PIXEL_COLUMNS]:
        assert column in values
    assert values["p3_y"] == 6.0
    assert values["rule"] in (0, 1)
    assert values["handedness"] == "Right"


def test_crop_name_sorts_by_time():
    """시각 ms 여섯 자리라 이름순이 곧 시간순."""
    assert hand_clips.crop_name(12.4, 0) == "012400_0.jpg"
    assert hand_clips.crop_name(2.0, 1) < hand_clips.crop_name(12.0, 0)


# --- segments.csv -----------------------------------------------------------------


def test_segments_by_video_stops_when_a_class_is_missing():
    """손으로 적다 한 줄을 빠뜨리면 extract 전에 멈춘다."""
    rows = [
        {"video": "h_hall_p1", "cls": cls, "start_sec": "0", "end_sec": "10", "source": "manual"}
        for cls in hand_clips.CLASSES[:5]
    ]
    with pytest.raises(ValueError, match="h_hall_p1"):
        hand_clips.segments_by_video(rows)


def test_manual_videos_only_counts_manual_rows():
    rows = [{"video": "a", "source": "auto"}, {"video": "b", "source": "manual"}]
    assert hand_clips.manual_videos(rows) == {"b"}


# --- 요약 -------------------------------------------------------------------------


def _row(video, t, cls, n_hands, label="", rule=""):
    return {"video": video, "t_sec": t, "cls": cls, "person": "p1", "n_hands": str(n_hands),
            "label": label, "rule": rule}  # fmt: skip


def test_summarize_counts_frames_not_rows_and_scores_rule():
    """손 둘인 프레임은 한 프레임. 뺀 손은 점수에서 빠지고, 재현율 · 오탐률은 이름표 기준."""
    rows = [
        _row("v", "1.0", "point", 1, "1", "1"),  # 맞힘
        _row("v", "1.2", "point", 1, "1", "0"),  # 놓침
        _row("v", "1.4", "point", 2),  # 뺀 손 둘
        _row("v", "1.4", "point", 2),
        _row("v", "1.6", "point", 0),  # 손 못 찾음
        _row("v", "2.0", "fist", 1, "0", "1"),  # 오탐
        _row("v", "2.2", "fist", 1, "0", "0"),
        _row("v", "2.4", "fist", 1, "0", "0"),
    ]
    stat = hand_clips.summarize(rows, None)["전체"]
    assert stat["frames"] == 7 and stat["found"] == 6 and stat["dropped"] == 2
    assert stat["recall"] == 0.5
    assert stat["false_alarm"] == pytest.approx(1 / 3)
    assert stat["accuracy"] == 0.6
    by_cls = hand_clips.summarize(rows, "cls")
    assert by_cls["point"]["frames"] == 4 and by_cls["fist"]["frames"] == 3


# --- 합성 영상으로 segments 한 바퀴 ------------------------------------------------


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


def _clip(cover_value: int) -> list[np.ndarray]:
    """10fps: 손 모양 9초 + 가림 1초 × 5 + 마지막 9초 = 59초."""
    bright = np.full((36, 64, 3), 150, np.uint8)
    cover = np.full((36, 64, 3), cover_value, np.uint8)
    frames = []
    for i in range(6):
        frames += [bright] * 90
        if i < 5:
            frames += [cover] * 10
    return frames


def test_segments_command_writes_auto_rows_and_keeps_manual(monkeypatch, tmp_path, capsys):
    """까만 가림은 자동 6줄, 살색(밝은) 가림은 review 사진만, 손으로 적은 영상은 그대로."""
    videos = tmp_path / "videos"
    videos.mkdir()
    if not _write_video(videos / "h_hall_p1.mp4", _clip(5), fps=10):
        pytest.skip("이 OpenCV 는 mp4 를 쓰지 못한다")
    _write_video(videos / "h_hall_p2.mp4", _clip(120), fps=10)
    _write_video(videos / "h_hall_p3.mp4", _clip(120), fps=10)
    manual = [
        {"video": "h_hall_p3", "place": "hall", "person": "p3", "cls": cls,
         "start_sec": i * 10, "end_sec": i * 10 + 9, "source": "manual"}
        for i, cls in enumerate(hand_clips.CLASSES)
    ]  # fmt: skip
    hand_clips.write_csv(tmp_path / "segments.csv", hand_clips.SEGMENTS_COLUMNS, manual)
    monkeypatch.setattr(sys, "argv", ["hand_clips.py", "segments", "--root", str(tmp_path)])

    hand_clips.main()

    rows = hand_clips.read_csv(tmp_path / "segments.csv")
    auto = [r for r in rows if r["video"] == "h_hall_p1"]
    assert [r["cls"] for r in auto] == list(hand_clips.CLASSES)
    assert all(r["source"] == "auto" for r in auto)
    assert float(auto[1]["start_sec"]) == pytest.approx(10.0, abs=0.3)
    assert not [r for r in rows if r["video"] == "h_hall_p2"]
    assert (tmp_path / "review" / "h_hall_p2.jpg").exists()
    assert [r["source"] for r in rows if r["video"] == "h_hall_p3"] == ["manual"] * 6
    out = capsys.readouterr().out
    assert "손으로 적은 구간이 있어 그대로" in out and "2/3개 영상" in out


def test_extract_stops_when_landmarks_exist(tmp_path):
    """이미 뽑은 표를 덮어쓰지 않는다(손 검출기를 부르기 전에 멈춤)."""
    (tmp_path / "landmarks.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="지우세요"):
        hand_clips.run_extract(tmp_path, detector=None)
