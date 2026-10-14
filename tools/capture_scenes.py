"""지시 장면 촬영 도구 — 안경 카메라로 사진을 찍고, 그때 한 말을 함께 기록한다.

가리킴 모델의 valid·test 사진은 직접 찍어야 하고, 사진마다 "그때 한 말"이
정답으로 붙어 있어야 평가할 수 있다. 나중에 사진을 보고 지어내지 않도록
찍는 순간에 터미널에서 발화를 받아 manifest.csv 에 적는다.

실행:
    python tools/capture_scenes.py --session 1 --place desk

    스페이스 : 촬영 → 터미널에 발화 입력 (빈칸 엔터 = 취소)
    q        : 종료

저장:
    data/datasets/pointing/raw/s01/s01_desk_0001.jpg
    data/datasets/pointing/raw/s01/manifest.csv

주의: 터미널에서 발화를 입력한 뒤 미리보기 창을 한 번 클릭해야 키가 다시 먹는다.
cv2.waitKey 는 미리보기 창에 포커스가 있을 때만 키를 받는다.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from jarviseo import config
from jarviseo.capture import USBCamSource

# 다음 작업(#3 미리 라벨·분할)이 이 순서로 읽는다. 바꾸면 그쪽도 같이 고친다.
MANIFEST_COLUMNS = ["file", "session", "place", "utterance", "captured_at"]
# 1080p 를 그대로 띄우면 맥북 화면을 넘친다. 저장은 원본, 미리보기만 줄인다.
PREVIEW_WIDTH = 960
JPEG_QUALITY = 95
WINDOW_NAME = "capture_scenes"


def build_filename(session: int, place: str, number: int) -> str:
    """세션·장소·번호로 파일 이름을 만든다. 예: (3, "kitchen", 42) → s03_kitchen_0042.jpg"""
    return f"s{session:02d}_{place}_{number:04d}.jpg"


def next_number(session_dir: Path) -> int:
    """세션 폴더에 이미 있는 사진을 보고, 다음에 쓸 번호를 돌려준다.

    "파일 개수 + 1" 이 아니라 "가장 큰 번호 + 1" 이다. 중간 사진을 지우면
    개수가 줄어서, 개수로 세면 이미 있는 사진을 덮어쓴다.
    """
    if not session_dir.exists():
        return 1
    numbers = []
    for path in session_dir.glob("*.jpg"):
        try:
            # "s03_desk_0007" → 오른쪽 밑줄에서 한 번만 잘라 "0007" → 7
            numbers.append(int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            # 규칙에 안 맞는 jpg 가 섞여 있어도 멈추지 않는다.
            continue
    return max(numbers) + 1 if numbers else 1


def append_manifest(manifest_path: Path, row: dict[str, object]) -> None:
    """manifest.csv 에 한 줄을 덧붙인다. 파일이 처음 생길 때만 헤더를 쓴다.

    encoding 을 빼면 윈도우에서 cp949 로 저장돼 한글이 깨지고,
    newline="" 을 빼면 윈도우에서 줄 사이에 빈 줄이 생긴다.
    utf-8-sig 는 쓰지 않는다. 이어 쓰기 모드에서는 BOM 이 파일 중간에 박힌다.
    """
    # 열기 전에 확인해야 한다. 열고 나면 파일이 항상 있어서 구분이 안 된다.
    is_new = not manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def save_jpg(path: Path, image: np.ndarray) -> None:
    """사진을 jpg 로 저장한다.

    cv2.imwrite 는 윈도우에서 경로에 한글이 있으면 에러 없이 실패한다.
    그래서 메모리에서 인코딩한 뒤 파이썬이 파일을 쓰게 한다.
    """
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"jpg 인코딩에 실패했습니다: {path}")
    buffer.tofile(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="지시 장면 촬영 — 사진과 발화를 함께 저장")
    parser.add_argument("--session", type=int, required=True, help="세션 번호, 예: 3")
    parser.add_argument("--place", required=True, help="영어 소문자, 예: desk kitchen shelf")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="카메라 번호")
    parser.add_argument(
        "--out",
        type=Path,
        default=config.DATASETS_DIR / "pointing" / "raw",
        help="저장할 최상위 폴더",
    )
    args = parser.parse_args()
    # 밑줄이 들어가면 next_number 의 번호 파싱이 깨지고, 한글·공백은 윈도우 경로 문제를 만든다.
    if not (args.place.isascii() and args.place.isalpha() and args.place.islower()):
        parser.error(f"--place 는 영어 소문자만 됩니다: {args.place!r}")
    return args


def main() -> None:
    args = parse_args()
    session_dir = args.out / f"s{args.session:02d}"
    session_dir.mkdir(parents=True, exist_ok=True)
    # 세션마다 manifest 하나. 여러 사람이 찍은 세션을 합칠 때 같은 파일을 고치지 않는다.
    manifest_path = session_dir / "manifest.csv"
    number = next_number(session_dir)
    saved = 0

    print(f"저장 위치: {session_dir}")
    print(f"시작 번호: {number:04d}")
    print("스페이스 = 촬영, q = 종료 (발화 입력 뒤 미리보기 창을 한 번 클릭)")

    with USBCamSource(index=args.camera) as src:
        first = src.read()
        if first is None:
            print("카메라에서 프레임을 읽지 못했습니다.")
            return
        # cap.set() 은 실패해도 조용하다. 640x480 으로 떨어지면 성분표 글자를 못 읽는다.
        if first.size != (1920, 1080):
            print(f"경고: 실제 해상도가 {first.size} 입니다. 1920x1080 이 아닙니다.")

        while True:
            frame = src.read()
            if frame is None:
                print("카메라 연결이 끊겼습니다.")
                break

            width, height = frame.size
            preview_height = round(height * PREVIEW_WIDTH / width)
            preview = cv2.resize(frame.image, (PREVIEW_WIDTH, preview_height))
            # putText 는 한글을 못 그린다. 영어·숫자만 쓴다.
            status = (
                f"s{args.session:02d} {args.place}  "
                f"saved {saved}  next {number:04d}  SPACE=shot Q=quit"
            )
            cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, preview)

            # 한 번만 부르고 담아 둔다. 두 번 부르면 키를 놓친다.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != ord(" "):
                continue

            # 스페이스를 누른 순간의 원본을 붙잡아 둔다. 발화를 입력하는 동안 미리보기는 멈춘다.
            shot = frame.image.copy()
            utterance = input("발화 (빈칸=취소): ").strip()
            if not utterance:
                print("취소했습니다.")
                continue

            filename = build_filename(args.session, args.place, number)
            # 사진 먼저, manifest 나중. 저장이 실패하면 없는 사진의 줄이 manifest 에 남지 않는다.
            save_jpg(session_dir / filename, shot)
            append_manifest(
                manifest_path,
                {
                    "file": filename,
                    "session": args.session,
                    "place": args.place,
                    "utterance": utterance,
                    "captured_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            number += 1
            saved += 1
            print(f"저장: {filename}")

    cv2.destroyAllWindows()
    print(f"이번 실행에서 {saved}장을 저장했습니다. manifest: {manifest_path}")


if __name__ == "__main__":
    main()
