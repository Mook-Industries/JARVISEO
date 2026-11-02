"""손 관절 측정 — 사진별 결과 · 표 한 줄 집계 · 출력 · CSV · 그림 · 조각 확대 줄 · main 한 바퀴.

MediaPipe 없이 HandPose 를 직접 만들어 넣는다. 검출은 tests/pointing/test_hands.py 가 본다.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

from jarviseo.pointing.hands import HandPose
from jarviseo.types import BBox

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
# check_hands 가 같은 폴더의 run_ablation 을 import 한다. 스크립트로 실행하면 그 폴더가 경로에 든다.
sys.path.append(str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location("check_hands", _SCRIPTS / "check_hands.py")
check_hands = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_hands)

GT = BBox(900, 0, 1100, 100)  # 정답 박스 중심 (1000, 50)


def _hand(pointing: bool, tip=(1000.0, 500.0), base=(1000.0, 600.0)) -> HandPose:
    """pointing 이면 검지만 펴고 나머지를 접은 손(차이 180°), 아니면 다 편 손.

    방향은 base → tip. 기본값은 위쪽, 정답 박스 중심을 똑바로 가리킨다.
    """
    folded = [(0.0, 0.0, 0.0), (0.0, 0.02, 0.0), (0.0, 0.02, 0.02), (0.0, 0.0, 0.02)]
    straight = [(0.0, 0.0, 0.0), (0.0, 0.02, 0.0), (0.0, 0.04, 0.0), (0.0, 0.06, 0.0)]
    world = [(0.0, 0.0, 0.0)] * 5 + straight + (folded if pointing else straight) * 3
    landmarks = [(0.0, 0.0)] * 21
    landmarks[5], landmarks[8] = base, tip
    return HandPose(tuple(landmarks), tuple(world), "Right", 0.9)


def _sample(file: str, has_hand: bool) -> dict:
    return {"file": file, "has_hand": has_hand, "gt": GT}


def test_hand_rows_records_pointing_hand_and_angle():
    """가리키는 손을 고르고 정답 박스 중심까지 각도를 적는다. 손이 없으면 None."""
    samples = [_sample("a.jpg", True), _sample("b.jpg", False)]
    hands = {"a.jpg": [_hand(False), _hand(True)], "b.jpg": []}
    rows = check_hands.hand_rows(samples, hands, {"a.jpg": 12.0, "b.jpg": 8.0}, 15.0)
    assert rows[0] == {
        "file": "a.jpg",
        "has_hand": True,
        "hands": 2,
        "pointing": True,
        "score": 1.0,
        "angle": 0.0,
        "ms": 12.0,
    }
    assert (rows[1]["hands"], rows[1]["pointing"], rows[1]["score"], rows[1]["angle"]) == (
        0,
        False,
        None,
        None,
    )


def test_hand_rows_open_hand_is_not_pointing():
    """손이 잡혀도 다 편 손뿐이면 가리킴이 아니다(손 단서 기권)."""
    rows = check_hands.hand_rows(
        [_sample("a.jpg", False)], {"a.jpg": [_hand(False)]}, {"a.jpg": 10.0}, 15.0
    )
    assert (rows[0]["hands"], rows[0]["pointing"]) == (1, False)


def _row(has_hand, hands=1, pointing=True, angle=10.0, ms=10.0) -> dict:
    return {"has_hand": has_hand, "hands": hands, "pointing": pointing, "angle": angle, "ms": ms}


def test_summarize_counts_by_hand_and_angle():
    """손 있음 · 없음을 나눠 세고, 30° 는 안으로, 30° 넘으면 밖으로 센다."""
    rows = [
        _row(True, angle=10.0),
        _row(True, angle=30.0),
        _row(True, angle=50.0),
        _row(True, hands=1, pointing=False, angle=None),
        _row(True, hands=0, pointing=False, angle=None),
        _row(False, pointing=True, angle=90.0),
        _row(False, hands=0, pointing=False, angle=None),
    ]
    m = check_hands.summarize(rows)
    assert (m["with_hand"], m["found"], m["pointing"], m["within"]) == (5, 4, 3, 2)
    assert m["median_angle"] == 30.0  # 손 있음 사진만, 손 없음 오탐(90°)은 안 섞는다
    assert (m["without_hand"], m["false_pointing"]) == (2, 1)


def test_summarize_without_any_pointing_hand():
    """가리키는 손이 하나도 없으면 각도 중앙값은 None, 빈 결과도 멈추지 않는다."""
    m = check_hands.summarize([_row(True, hands=0, pointing=False, angle=None)])
    assert (m["pointing"], m["within"], m["median_angle"]) == (0, 0, None)
    assert check_hands.summarize([])["with_hand"] == 0


def test_format_row():
    """표 한 줄: 기준 · 장수 · 각도 · 지연."""
    m = check_hands.summarize([_row(True, angle=12.4, ms=11.0), _row(False, pointing=False)])
    line = check_hands.format_row(0.1, 15.0, m)
    assert line == "0.10 | 15° | 1/1 | 1/1 | 1 | 12° | 0/1 | 11.0/11.0"


def test_write_csv(tmp_path):
    """사진별 결과를 헤더와 함께 쓴다. 한글 경로도 된다."""
    rows = check_hands.hand_rows(
        [_sample("s01/a.jpg", True)], {"s01/a.jpg": [_hand(True)]}, {"s01/a.jpg": 9.5}, 15.0
    )
    path = check_hands.write_csv(tmp_path / "손" / "hands_valid.csv", rows)
    with path.open(encoding="utf-8", newline="") as f:
        saved = list(csv.DictReader(f))
    assert saved[0]["file"] == "s01/a.jpg"
    assert (saved[0]["pointing"], saved[0]["angle"]) == ("True", "0.0")


def test_draw_does_not_touch_original():
    """그림은 사본에 그리고, 원본 사진은 그대로 둔다."""
    image = np.zeros((200, 2000, 3), np.uint8)
    hand = _hand(True, tip=(1000.0, 150.0), base=(1000.0, 190.0))
    out = check_hands.draw(image, [hand], hand, GT)
    assert out.shape == image.shape
    assert not image.any()
    assert out.any()


def test_save_image_korean_path(tmp_path):
    """한글이 든 경로에도 저장되고 다시 읽힌다."""
    path = tmp_path / "그림" / "s01" / "a.jpg"
    check_hands.save_image(path, np.full((10, 10, 3), 255, np.uint8))
    assert check_hands.read_image(path).shape == (10, 10, 3)


# --- 조각 확대 (#256) -----------------------------------------------------------------


def test_format_row_marks_tile_row():
    """조각 확대 줄은 검출 기준 뒤에 +조각 을 붙인다."""
    m = check_hands.summarize([_row(True, angle=12.4, ms=11.0), _row(False, pointing=False)])
    line = check_hands.format_row(0.1, 15.0, m, tiles=True)
    assert line == "0.10+조각 | 15° | 1/1 | 1/1 | 1 | 12° | 0/1 | 11.0/11.0"


def test_false_pointing_files_only_no_hand_photos():
    """손 없음 사진에서 가리키는 손을 고른 사진만 모은다."""
    rows = [
        {"file": "a.jpg", "has_hand": False, "pointing": True},
        {"file": "b.jpg", "has_hand": True, "pointing": True},
        {"file": "c.jpg", "has_hand": False, "pointing": False},
    ]
    assert check_hands.false_pointing_files(rows) == ["a.jpg"]


class _NoHands:
    """손을 하나도 못 찾는 가짜 검출기. 부른 횟수를 센다."""

    def __init__(self, **kwargs) -> None:
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return []


def test_find_hands_warms_up_once_then_times_each_photo():
    """첫 사진으로 한 번 데우고(버림), 사진마다 찾은 손과 시간을 남긴다."""
    images = {"a.jpg": np.zeros((4, 4, 3), np.uint8), "b.jpg": np.zeros((4, 4, 3), np.uint8)}
    detector = _NoHands()
    hands, detect_ms = check_hands.find_hands(detector, images)
    assert detector.calls == 3
    assert hands == {"a.jpg": [], "b.jpg": []}
    assert set(detect_ms) == {"a.jpg", "b.jpg"}


def test_main_reads_default_sessions_and_prints_tile_row(tmp_path, monkeypatch, capsys):
    """main 이 기본 세션(valid 1 · 2)만 읽고 조각 확대 줄까지 끝까지 돈다.

    세션 인자를 빠뜨려 load_samples 에서 바로 멈춘 적이 있다. 세션 4 사진은 안 섞인다.
    """
    labels = tmp_path / "labels.csv"
    columns = ["file", "session", "split", "utterance", "gt_x1", "gt_y1", "gt_x2", "gt_y2"]
    columns += ["gt_label", "has_hand", "ambiguous", "prev_label", "excluded"]
    with labels.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerow(["s01/a.jpg", "1", "valid", "이건 뭐야?", 1, 2, 3, 4, "cup", 1, 0, "", 0])
        writer.writerow(["s04/d.jpg", "4", "valid", "저건 뭐야?", 1, 2, 3, 4, "cup", 0, 0, "", 0])
    monkeypatch.setattr(check_hands, "HandDetector", _NoHands)
    monkeypatch.setattr(check_hands, "read_image", lambda path: np.zeros((10, 10, 3), np.uint8))
    out = tmp_path / "out"
    argv = ["check_hands.py", "--labels", str(labels), "--raw", str(tmp_path), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)

    check_hands.main()

    printed = capsys.readouterr().out
    assert "valid 1장 (손 있음 1 · 없음 0)" in printed
    assert "0.10+조각 | 15° | 0/1 | 0/1 |" in printed
    assert (out / "hands_valid.csv").exists()
