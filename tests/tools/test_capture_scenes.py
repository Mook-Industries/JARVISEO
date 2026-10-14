"""tools/capture_scenes.py 중 카메라 없이 검사할 수 있는 부분을 확인한다.

카메라 루프(main)는 실물이 있어야 해서 여기서 다루지 않는다.
대신 틀리면 촬영한 사진이 덮어써지거나, 다음 작업(#3 분할)이 manifest 를
못 읽게 되는 부분 — 파일명 · 번호 · manifest · 저장 · 인자 검사 — 을 잡는다.

tools/ 는 패키지가 아니라서 파일 경로로 직접 불러온다.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from jarviseo import config

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "capture_scenes.py"
_spec = importlib.util.spec_from_file_location("capture_scenes", _TOOL_PATH)
capture_scenes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture_scenes)


def _set_argv(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["capture_scenes.py", *args])


def _row(utterance: str) -> dict[str, object]:
    return {
        "file": "s03_kitchen_0001.jpg",
        "session": 3,
        "place": "kitchen",
        "utterance": utterance,
        "captured_at": "2026-10-07T14:00:00",
    }


# --- 다음 작업과의 약속 -----------------------------------------------------


def test_manifest_열_순서는_다음_작업과의_약속():
    # #3 미리 라벨·분할 스크립트가 이 순서로 읽는다. 바꾸면 그쪽도 같이 고쳐야 한다.
    expected = ["file", "session", "place", "utterance", "captured_at"]
    assert capture_scenes.MANIFEST_COLUMNS == expected


# --- 파일명 ----------------------------------------------------------------


def test_파일명은_세션_2자리_번호_4자리():
    assert capture_scenes.build_filename(3, "kitchen", 42) == "s03_kitchen_0042.jpg"
    assert capture_scenes.build_filename(12, "desk", 1) == "s12_desk_0001.jpg"


# --- 다음 번호 --------------------------------------------------------------


def test_폴더가_없으면_1번부터(tmp_path):
    assert capture_scenes.next_number(tmp_path / "s01") == 1


def test_빈_폴더면_1번부터(tmp_path):
    assert capture_scenes.next_number(tmp_path) == 1


def test_장소가_달라도_세션_안에서_번호가_이어진다(tmp_path):
    for name in ["s03_desk_0001.jpg", "s03_desk_0002.jpg", "s03_kitchen_0003.jpg"]:
        (tmp_path / name).touch()
    assert capture_scenes.next_number(tmp_path) == 4


def test_중간_사진을_지워도_기존_사진을_덮어쓰지_않는다(tmp_path):
    # 0002 를 지운 상황. "개수 + 1" 로 세면 3 이 나와서 기존 0003 을 덮어쓴다.
    for name in ["s03_desk_0001.jpg", "s03_desk_0003.jpg"]:
        (tmp_path / name).touch()
    assert capture_scenes.next_number(tmp_path) == 4


def test_규칙에_안_맞는_파일은_건너뛴다(tmp_path):
    for name in ["s03_desk_0002.jpg", "memo.jpg", "s03_desk_abc.jpg", "manifest.csv"]:
        (tmp_path / name).touch()
    assert capture_scenes.next_number(tmp_path) == 3


# --- manifest --------------------------------------------------------------


def test_manifest_헤더는_처음_한_번만_쓴다(tmp_path):
    path = tmp_path / "manifest.csv"
    capture_scenes.append_manifest(path, _row("저거 뭐야"))
    capture_scenes.append_manifest(path, _row("저 빨간 거"))

    lines = path.read_text(encoding="utf-8").splitlines()
    # BOM(utf-8-sig)이 붙으면 첫 줄 맨 앞에 보이지 않는 글자가 생겨 여기서 걸린다.
    assert lines[0] == "file,session,place,utterance,captured_at"
    # 헤더 1 + 데이터 2. 윈도우에서 newline="" 을 빼면 빈 줄이 끼어 여기서 걸린다.
    assert len(lines) == 3


def test_manifest_한글과_쉼표가_그대로_남는다(tmp_path):
    path = tmp_path / "manifest.csv"
    capture_scenes.append_manifest(path, _row("음, 왼쪽에 있는 컵 뭐야"))

    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["utterance"] == "음, 왼쪽에 있는 컵 뭐야"
    assert rows[0]["session"] == "3"


# --- jpg 저장 --------------------------------------------------------------


def test_한글_경로에도_jpg가_저장된다(tmp_path):
    # cv2.imwrite 는 윈도우에서 한글 경로면 조용히 실패한다. CI 의 windows 가 이걸 잡는다.
    folder = tmp_path / "한글폴더"
    folder.mkdir()
    path = folder / "s01_desk_0001.jpg"
    image = np.full((1080, 1920, 3), 128, dtype=np.uint8)

    capture_scenes.save_jpg(path, image)

    assert path.exists()
    restored = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert restored.shape == (1080, 1920, 3)


# --- 실행 인자 --------------------------------------------------------------


def test_인자_기본값(monkeypatch):
    _set_argv(monkeypatch, "--session", "3", "--place", "kitchen")
    args = capture_scenes.parse_args()

    assert args.session == 3
    assert args.place == "kitchen"
    assert args.camera == config.CAMERA_INDEX
    assert args.out == config.DATASETS_DIR / "pointing" / "raw"


@pytest.mark.parametrize("place", ["Kitchen", "주방", "my_desk", "desk1", "my desk"])
def test_장소_이름은_영어_소문자만(monkeypatch, place):
    # 밑줄이 들어가면 번호 파싱(rsplit)이 깨지고, 한글·공백은 윈도우 경로 문제를 만든다.
    _set_argv(monkeypatch, "--session", "3", "--place", place)
    with pytest.raises(SystemExit):
        capture_scenes.parse_args()


def test_세션은_필수(monkeypatch):
    _set_argv(monkeypatch, "--place", "desk")
    with pytest.raises(SystemExit):
        capture_scenes.parse_args()
