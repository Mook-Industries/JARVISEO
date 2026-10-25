"""가리킴 평가셋 정답 표시 — 찍은 사진마다 가리킨 대상의 정답 박스를 남긴다.  담당: 최홍묵

capture_scenes.py 로 찍은 세션 폴더의 사진에 YOLO 후보를 왼쪽부터 번호로 그려 보여 주고,
정답 후보 번호를 고르거나(후보에 없으면 직접 박스를 그려)
data/datasets/pointing/labels.csv 에 남긴다.
scripts/run_ablation.py 가 이 파일로 v1~v5 를 채점한다. 정답은 이름이 아니라 박스라서,
같은 컵이 둘이어도 어느 쪽이 정답인지 구별된다.

실행 (레포 맨 위 폴더에서):
    python tools/label_targets.py --session 1               # valid 세션
    python tools/label_targets.py --session 3 --split test  # test 세션 (최종 측정 때만 채점)

키 (발화·라벨을 터미널에 입력한 뒤에는 그림 창을 한 번 클릭해야 키가 다시 먹는다):
    1~9  그 번호 후보가 정답 → 터미널에서 정답 이름 확인(Enter = YOLO 이름)
    d    후보에 없으면 직접 박스 그리기(끌고 Enter)
    h    손 보임 켜고 끄기            a  사람이 봐도 모호함 켜고 끄기(되묻는 게 정답)
    p    앞 질문의 대상 라벨 입력(v5)  x  쓸 수 없는 사진으로 빼기
    q    끝내기 (다음에 이어서 한다. 이미 표시한 사진은 다시 묻지 않는다)
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from jarviseo import config
from jarviseo.types import BBox, Detection

RAW_DIR = config.DATASETS_DIR / "pointing" / "raw"
LABELS_PATH = config.DATASETS_DIR / "pointing" / "labels.csv"
# run_ablation.py 가 이 이름으로 읽는다. 바꾸면 그쪽도 같이 고친다.
LABEL_COLUMNS = [
    "file",
    "session",
    "split",
    "utterance",
    "gt_x1",
    "gt_y1",
    "gt_x2",
    "gt_y2",
    "gt_label",
    "has_hand",
    "ambiguous",
    "prev_label",
    "excluded",
]
# 1080p 를 그대로 띄우면 맥북 화면을 넘친다. 보여 주기만 줄이고 좌표는 원본으로 저장한다.
PREVIEW_WIDTH = 960
WINDOW_NAME = "label_targets"
# 사진마다 터미널에 찍는 키 안내. 파일 머리말에만 두면 쓰는 중에 못 찾는다.
KEY_HELP = (
    "  키: 1~9 정답 · d 직접 박스 · h 손 · a 모호 · p 앞 대상 · x 제외 · q 끝\n"
    "  1~9: 터미널에 정답 이름 — YOLO 이름이 맞으면 Enter, 틀리면 입력 → 그림 창 다시 클릭\n"
    "  d: 그림 창에서 끌고 Enter(취소 c) → 터미널에 라벨 입력 → 그림 창 다시 클릭\n"
    "  라벨: 영어 소문자 · 띄어쓰기, COCO 이름 그대로 (cell phone, air conditioner)"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """CSV 를 줄마다 dict 로 읽는다. 파일이 없으면 빈 리스트.

    encoding 을 빼면 윈도우에서 한글이 깨지고, newline="" 을 빼면 줄 사이에 빈 줄이 생긴다.
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_label(path: Path, row: dict[str, object]) -> None:
    """labels.csv 에 한 줄을 덧붙인다. 파일이 처음 생길 때만 헤더를 쓴다.

    한 장마다 바로 써서, 중간에 꺼져도 앞의 표시가 남는다.
    """
    # 열기 전에 확인해야 한다. 열고 나면 파일이 항상 있어서 구분이 안 된다.
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def relative_file(session: int, filename: str) -> str:
    """raw 폴더 기준 상대 경로. 절대 경로를 넣으면 다른 컴퓨터에서 깨진다."""
    return f"s{session:02d}/{filename}"


def pending(manifest_rows: list[dict[str, str]], labeled: set[str], session: int) -> list[dict]:
    """manifest 에 있지만 아직 표시하지 않은 사진만 남긴다. 순서는 찍은 순서 그대로."""
    return [r for r in manifest_rows if relative_file(session, r["file"]) not in labeled]


def left_to_right(detections: list[Detection]) -> list[Detection]:
    """화면 왼쪽부터 번호를 매기려고 정렬한다. YOLO 순서는 신뢰도 순이라 위치와 상관없다."""
    return sorted(detections, key=lambda d: d.bbox.x1)


def chosen_label(yolo_label: str, typed: str) -> str:
    """후보 번호로 고른 정답의 이름. 입력이 비면 YOLO 이름, 아니면 입력한 이름(앞뒤 공백 제거).

    박스는 맞아도 YOLO 이름은 틀릴 수 있다. 안경 케이스를 remote, 텀블러를 parking meter,
    모니터를 laptop 으로 불렀고, 그 이름이 그대로 정답표에 들어가 학습 종류까지 잘못 정했다(10/23).
    """
    return typed.strip() or yolo_label


def roi_to_box(roi: tuple[float, float, float, float], scale: float) -> BBox | None:
    """미리보기에서 그린 (x, y, 폭, 높이) 를 원본 좌표 박스로 되돌린다. 그리다 만 경우 None.

    배율을 안 곱하면 박스가 왼쪽 위에 쪼그라져 저장되는데, 에러가 안 나서 채점 때 전부 틀림이 된다.
    """
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return None
    return BBox(x * scale, y * scale, (x + w) * scale, (y + h) * scale)


def make_row(
    file: str,
    session: int,
    split: str,
    utterance: str,
    box: BBox | None,
    label: str,
    *,
    has_hand: bool = False,
    ambiguous: bool = False,
    prev_label: str = "",
    excluded: bool = False,
) -> dict[str, object]:
    """labels.csv 한 줄. 참/거짓은 0/1 로 쓴다("False" 글자를 bool() 로 읽으면 참이 되는 함정)."""
    # 뺀 사진은 정답 박스가 없어서 좌표를 비운다
    x1 = y1 = x2 = y2 = ""
    if box is not None:
        x1, y1, x2, y2 = (round(v, 1) for v in (box.x1, box.y1, box.x2, box.y2))
    return {
        "file": file,
        "session": session,
        "split": split,
        "utterance": utterance,
        "gt_x1": x1,
        "gt_y1": y1,
        "gt_x2": x2,
        "gt_y2": y2,
        "gt_label": label,
        "has_hand": int(has_hand),
        "ambiguous": int(ambiguous),
        "prev_label": prev_label,
        "excluded": int(excluded),
    }


def read_image(path: Path) -> np.ndarray | None:
    """cv2.imread 는 윈도우에서 경로에 한글이 있으면 조용히 실패해서, 바이트로 읽어 푼다."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def draw(image: np.ndarray, candidates: list[Detection], status: str) -> tuple[np.ndarray, float]:
    """후보 박스에 1, 2, 3… 번호를 그린 미리보기와, 미리보기 → 원본 배율을 돌려준다."""
    height, width = image.shape[:2]
    scale = width / PREVIEW_WIDTH
    preview = cv2.resize(image, (PREVIEW_WIDTH, round(height / scale)))
    for number, d in enumerate(candidates, start=1):
        x1, y1, x2, y2 = (round(v / scale) for v in (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2))
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # putText 는 한글을 못 그린다. 번호와 영어 라벨만 쓴다.
        tag = f"{number} {d.label} {d.confidence:.2f}"
        cv2.putText(preview, tag, (x1 + 3, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(preview, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return preview, scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="가리킴 평가셋 정답 표시")
    parser.add_argument("--session", type=int, required=True, help="세션 번호, 예: 1")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="촬영 사진 최상위 폴더")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH, help="정답을 남길 CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_dir = args.raw / f"s{args.session:02d}"
    manifest = read_csv(session_dir / "manifest.csv")
    if not manifest:
        raise SystemExit(f"찍은 사진이 없다: {session_dir / 'manifest.csv'}")
    labeled = {row["file"] for row in read_csv(args.labels)}
    todo = pending(manifest, labeled, args.session)
    print(f"세션 {args.session:02d} ({args.split}): 전체 {len(manifest)}장, 남은 {len(todo)}장")
    if not todo:
        return

    # 실제 실행 · 채점과 같은 검출기라서, 화면의 후보가 곧 v1~v5 가 고를 후보다.
    from jarviseo.pointing.detector import ObjectDetector

    detector = ObjectDetector()
    for done, item in enumerate(todo, start=1):
        file = relative_file(args.session, item["file"])
        image = read_image(session_dir / item["file"])
        if image is None:
            print(f"사진을 못 읽어서 건너뛴다: {file}")
            continue
        candidates = left_to_right(detector.detect(image))
        print(f"\n[{done}/{len(todo)}] {file}  발화: {item['utterance']}  후보 {len(candidates)}개")
        print(KEY_HELP)
        has_hand, ambiguous, prev_label = False, False, ""
        row = None
        while row is None:
            status = f"hand={int(has_hand)} ambiguous={int(ambiguous)} prev={prev_label or '-'}"
            preview, scale = draw(image, candidates, status)
            cv2.imshow(WINDOW_NAME, preview)
            # 한 번만 부르고 담아 둔다. 두 번 부르면 키를 놓친다.
            key = cv2.waitKey(0) & 0xFF
            flags = {"has_hand": has_hand, "ambiguous": ambiguous, "prev_label": prev_label}
            if key == ord("q"):
                cv2.destroyAllWindows()
                print("끝냈다. 다음에 같은 명령으로 이어서 한다.")
                return
            if ord("1") <= key <= ord("9") and key - ord("0") <= len(candidates):
                chosen = candidates[key - ord("1")]
                typed = input(f"정답 라벨 (영어, Enter = {chosen.label}): ")
                row = make_row(
                    file,
                    args.session,
                    args.split,
                    item["utterance"],
                    chosen.bbox,
                    chosen_label(chosen.label, typed),
                    **flags,
                )
            elif key == ord("d"):
                roi = cv2.selectROI(WINDOW_NAME, preview, showCrosshair=True, fromCenter=False)
                box = roi_to_box(roi, scale)
                if box is None:
                    print("박스를 그리지 않아서 취소했다.")
                    continue
                label = input("정답 라벨 (영어, 빈칸=unknown): ").strip() or "unknown"
                row = make_row(
                    file, args.session, args.split, item["utterance"], box, label, **flags
                )
            elif key == ord("x"):
                row = make_row(
                    file, args.session, args.split, item["utterance"], None, "", excluded=True
                )
            elif key == ord("h"):
                has_hand = not has_hand
            elif key == ord("a"):
                ambiguous = not ambiguous
            elif key == ord("p"):
                prev_label = input("앞 질문의 대상 라벨 (영어, 빈칸=지우기): ").strip()
        append_label(args.labels, row)
        print(f"  저장: {row['gt_label'] or '제외'}")
    cv2.destroyAllWindows()
    print("이 세션은 다 표시했다.")


if __name__ == "__main__":
    main()
