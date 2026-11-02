"""손 관절 측정 — 손 검출률 · 가리킴 판정 · 방향 오차 · 지연.  담당: 최홍묵

tools/label_targets.py 가 만든 labels.csv 의 사진에서 MediaPipe 로 손을 찾고(pointing/hands.py),
검출 기준 × 가리킴 판정 기준마다 표 한 줄을 찍는다.

- 손 찾음: 손 있음(h) 사진 중 손이 하나라도 잡힌 장수
- 가리킴 통과: 손 있음 사진 중 가리키는 손을 고른 장수
- 30° 안: 그중 검지 끝에서 뻗은 선과 '검지 끝 → 정답 박스 중심'이 30° 안인 장수.
  정답표에 방향 열이 없어서, 정답 박스 쪽을 가리켰는지로 방향 오차를 대신 잰다
- 손 없음 오탐: 손 없음 사진(들고 있는 손 · 키보드에 얹은 손 포함)에서 가리키는 손을 고른 장수

표 끝의 "0.10+조각" 줄은 조각 확대(#256)를 켠 지금 기준(0.1 · 15°)이다. 가리키는 손을 못 고른
사진만 4조각에서 다시 찾는다. 조각을 돌릴지를 15° 로 정해서 다른 판정 기준으로는 재지 않는다.

run_ablation.py 와 따로 둔다. 손 검출기 채점을 단서 채점에 섞으면 7분류가 흐려진다.
DB 에는 남기지 않는다(eval_run 은 ablation 버전 비교용). 결과는 docs/experiments.md 에 옮긴다.

실행 (레포 맨 위 폴더에서):
    python scripts/check_hands.py
    python scripts/check_hands.py --draw          # 관절 · 선 · 정답 박스를 그린 사진 저장
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from run_ablation import DEFAULT_SESSIONS, POINTING_DIR, load_samples, percentiles, read_image

from jarviseo.pointing.hands import (
    MIN_HAND_CONFIDENCE,
    POINTING_MIN_GAP_DEG,
    HandDetector,
    HandPose,
    angle_to,
    pick_pointing_hand,
    pointing_ray,
    pointing_score,
)
from jarviseo.types import BBox

# 정답 박스 중심이 선에서 이 각도 안이면 그쪽을 가리켰다고 본다. 출발점이다.
ANGLE_OK_DEG = 30.0
CSV_COLUMNS = ["file", "has_hand", "hands", "pointing", "score", "angle", "ms"]


def hand_rows(
    samples: list[dict],
    hands: dict[str, list[HandPose]],
    detect_ms: dict[str, float],
    min_gap_deg: float,
) -> list[dict]:
    """사진마다 찾은 손 수 · 가리키는 손 · 정답 박스 중심까지 각도."""
    rows = []
    for s in samples:
        found = hands[s["file"]]
        # 15° 규칙의 기준별 측정이라 손 분류기(#223) 설정과 상관없이 규칙으로 판정한다
        hand = pick_pointing_hand(found, min_gap_deg, classifier=None)
        angle = None
        if hand is not None:
            angle = angle_to(pointing_ray(hand), s["gt"].center)
        rows.append(
            {
                "file": s["file"],
                "has_hand": s["has_hand"],
                "hands": len(found),
                "pointing": hand is not None,
                "score": None if hand is None else round(pointing_score(hand), 3),
                "angle": None if angle is None else round(angle, 1),
                "ms": round(detect_ms[s["file"]], 1),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    """표 한 줄에 들어갈 숫자. 표본이 작아 % 대신 장수로 둔다."""
    with_hand = [r for r in rows if r["has_hand"]]
    without = [r for r in rows if not r["has_hand"]]
    angles = [r["angle"] for r in with_hand if r["angle"] is not None]
    p50, p95 = percentiles([r["ms"] for r in rows]) if rows else (0.0, 0.0)
    return {
        "with_hand": len(with_hand),
        "found": sum(r["hands"] > 0 for r in with_hand),
        "pointing": sum(r["pointing"] for r in with_hand),
        "within": sum(a <= ANGLE_OK_DEG for a in angles),
        "median_angle": statistics.median(angles) if angles else None,
        "without_hand": len(without),
        "false_pointing": sum(r["pointing"] for r in without),
        "p50_ms": p50,
        "p95_ms": p95,
    }


def format_row(confidence: float, min_gap_deg: float, m: dict, tiles: bool = False) -> str:
    n = m["with_hand"]
    angle = "-" if m["median_angle"] is None else f"{m['median_angle']:.0f}°"
    mark = "+조각" if tiles else ""
    return (
        f"{confidence:.2f}{mark} | {min_gap_deg:.0f}° | {m['found']}/{n} | {m['pointing']}/{n} | "
        f"{m['within']} | {angle} | {m['false_pointing']}/{m['without_hand']} | "
        f"{m['p50_ms']:.1f}/{m['p95_ms']:.1f}"
    )


def false_pointing_files(rows: list[dict]) -> list[str]:
    """손 없음 사진인데 가리키는 손을 고른 사진."""
    return [r["file"] for r in rows if r["pointing"] and not r["has_hand"]]


def find_hands(detector, images: dict[str, np.ndarray]) -> tuple[dict, dict[str, float]]:
    """사진마다 찾은 손과 걸린 시간(ms). 첫 호출은 모델을 올리느라 느려서 재기 전에 한 번 버린다."""
    detector.detect(next(iter(images.values())))
    hands, detect_ms = {}, {}
    for file, image in images.items():
        started = time.perf_counter()
        hands[file] = detector.detect(image)
        detect_ms[file] = (time.perf_counter() - started) * 1000
    return hands, detect_ms


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def draw(image: np.ndarray, hands: list[HandPose], chosen: HandPose | None, gt: BBox) -> np.ndarray:
    """관절(고른 손은 초록, 나머지는 회색) · 검지 끝에서 뻗은 선 · 정답 박스(빨강)를 그린 사본."""
    out = image.copy()
    cv2.rectangle(out, (int(gt.x1), int(gt.y1)), (int(gt.x2), int(gt.y2)), (0, 0, 255), 3)
    for hand in hands:
        color = (0, 220, 0) if hand is chosen else (160, 160, 160)
        for x, y in hand.landmarks:
            cv2.circle(out, (int(x), int(y)), 5, color, -1)
    ray = pointing_ray(chosen) if chosen is not None else None
    if ray is not None:
        (ox, oy), (dx, dy) = ray
        reach = sum(image.shape[:2])  # 화면 밖까지 뻗게
        end = (int(ox + dx * reach), int(oy + dy * reach))
        cv2.line(out, (int(ox), int(oy)), end, (0, 220, 0), 3)
    return out


def save_image(path: Path, image: np.ndarray) -> None:
    """cv2.imwrite 는 윈도우에서 경로에 한글이 있으면 조용히 실패해서, 인코딩한 바이트로 쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"사진을 인코딩하지 못했다: {path}")
    data.tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="손 관절 검출률 · 가리킴 판정 · 방향 오차")
    parser.add_argument("--labels", type=Path, default=POINTING_DIR / "labels.csv")
    parser.add_argument("--raw", type=Path, default=POINTING_DIR / "raw")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--confidences", type=float, nargs="+", default=[0.5, 0.3, 0.1])
    parser.add_argument("--gaps", type=float, nargs="+", default=[0.0, 15.0, 30.0])
    parser.add_argument("--out", type=Path, default=POINTING_DIR / "runs" / "hands")
    parser.add_argument("--draw", action="store_true", help="기본 기준 결과를 그림으로 저장")
    args = parser.parse_args()

    if args.split == "test":
        print("주의: test 를 연다 — 최종 측정 때만. 보면서 고치면 test 숫자가 실력이 아니게 된다.")
    samples = load_samples(args.labels, args.split, sorted(DEFAULT_SESSIONS[args.split]))
    if not samples:
        raise SystemExit(f"{args.split} 로 표시된 사진이 없다: {args.labels}")
    images = {}
    for s in samples:
        image = read_image(args.raw / s["file"])
        if image is None:
            raise SystemExit(f"사진을 못 읽었다: {args.raw / s['file']}")
        images[s["file"]] = image

    # 지금 쓰는 기준(hands.py 상수)은 늘 잰다. CSV · 그림은 그 기준으로 남긴다.
    confidences = sorted({*args.confidences, MIN_HAND_CONFIDENCE}, reverse=True)
    gaps = sorted({*args.gaps, POINTING_MIN_GAP_DEG})
    n_hand = sum(s["has_hand"] for s in samples)
    print(f"\n{args.split} {len(samples)}장 (손 있음 {n_hand} · 없음 {len(samples) - n_hand})\n")
    print(
        "검출 기준 | 판정 기준 | 손 찾음 | 가리킴 통과 | 30° 안 | 각도 중앙값 | "
        "손 없음 오탐 | p50/p95 ms"
    )

    default_rows, default_hands = None, None
    for confidence in confidences:
        # 기준별 줄은 MediaPipe 한 번(조각 없음)이다. 판정은 hand_rows 처럼 15° 규칙.
        detector = HandDetector(min_confidence=confidence, classifier=None, tiles=False)
        hands, detect_ms = find_hands(detector, images)
        for gap in gaps:
            rows = hand_rows(samples, hands, detect_ms, gap)
            print(format_row(confidence, gap, summarize(rows)))
            if confidence == MIN_HAND_CONFIDENCE and gap == POINTING_MIN_GAP_DEG:
                default_rows, default_hands = rows, hands

    tile_hands, tile_ms = find_hands(HandDetector(classifier=None, tiles=True), images)
    tile_rows = hand_rows(samples, tile_hands, tile_ms, POINTING_MIN_GAP_DEG)
    print(format_row(MIN_HAND_CONFIDENCE, POINTING_MIN_GAP_DEG, summarize(tile_rows), tiles=True))

    print(
        f"\n기본 기준({MIN_HAND_CONFIDENCE} · {POINTING_MIN_GAP_DEG:.0f}°) 손 없음 오탐: ", end=""
    )
    print(", ".join(false_pointing_files(default_rows)) or "없음")
    print("조각 확대 손 없음 오탐: ", end="")
    print(", ".join(false_pointing_files(tile_rows)) or "없음")
    path = write_csv(args.out / f"hands_{args.split}.csv", default_rows)
    print(f"사진별 결과: {path}")

    if args.draw:
        for s in samples:
            found = default_hands[s["file"]]
            chosen = pick_pointing_hand(found, classifier=None)
            picture = draw(images[s["file"]], found, chosen, s["gt"])
            save_image(args.out / args.split / s["file"], picture)
        print(f"그림: {args.out / args.split}")


if __name__ == "__main__":
    main()
