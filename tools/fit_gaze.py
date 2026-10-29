"""v3 시선 분포 맞추기 — 가리킨 물건이 화면 어디에 왔는지 평균 · 공분산을 낸다.  담당: 최홍묵

안경 카메라는 머리를 따라 움직여서, 사용자가 보고 가리킨 물건이 화면의 일정한 자리에 오는
경향이 있다. tools/label_targets.py 로 표시한 학습용 세션(split=train, 기본 s05)의 정답 박스
중심을 사진 너비 · 높이로 나눈 0~1 좌표로 모아 2D 가우시안(평균 · 공분산)을 맞춘다.
공분산은 n 으로 나눈다(최대우도 추정).
마지막에 찍는 상수 두 줄을 src/jarviseo/pointing/cues.py 에 그대로 붙인다.

valid · test 로는 맞추지 않는다. 채점할 사진으로 단서를 맞추면 v3 숫자가 실력보다 부풀려진다.

실행 (레포 맨 위 폴더에서):
    python tools/fit_gaze.py
    python tools/fit_gaze.py --sessions 5
"""

import argparse
from pathlib import Path

import numpy as np
from label_targets import LABELS_PATH, RAW_DIR, read_csv, read_image

DEFAULT_SESSIONS = (5,)
# 이보다 적으면 평균 · 공분산이 사진 몇 장의 운에 좌우된다.
MIN_POINTS = 10

Mean = tuple[float, float]
Cov = tuple[tuple[float, float], tuple[float, float]]


def train_rows(rows: list[dict[str, str]], sessions: list[int]) -> list[dict[str, str]]:
    """학습용 세션에서 쓸 줄만. 뺀 사진(x)과 정답 박스가 없는 줄은 건너뛴다."""
    return [
        r
        for r in rows
        # session 은 CSV 라 글자다. 정수로 바꿔야 [5] 와 비교된다.
        if r["split"] == "train"
        and int(r["session"]) in sessions
        and r["excluded"] != "1"
        and r["gt_x1"]
    ]


def centers(rows: list[dict[str, str]], sizes: dict[str, tuple[int, int]]) -> np.ndarray:
    """정답 박스 중심을 사진 너비 · 높이로 나눈 (x, y) 0~1 좌표들. sizes 는 파일 → (너비, 높이).

    나눠야 1920×1080 과 1280×720 사진에서 같은 자리가 같은 값이 된다.
    """
    points = []
    for r in rows:
        width, height = sizes[r["file"]]
        x = (float(r["gt_x1"]) + float(r["gt_x2"])) / 2 / width
        y = (float(r["gt_y1"]) + float(r["gt_y2"])) / 2 / height
        points.append((x, y))
    return np.array(points, dtype=float).reshape(-1, 2)


def fit(points: np.ndarray) -> tuple[Mean, Cov]:
    """평균과 공분산(n 으로 나눔). 점이 MIN_POINTS 보다 적거나 한 줄에 놓이면 ValueError.

    한 줄에 놓인 점은 공분산의 역행렬이 없어서, 시선 점수(마할라노비스 거리)를 계산할 수 없다.
    """
    if len(points) < MIN_POINTS:
        raise ValueError(
            f"점이 {len(points)}개뿐이다 — {MIN_POINTS}개 이상 있어야 분포를 믿을 수 있다"
        )
    cov = np.cov(points, rowvar=False, bias=True)
    if np.linalg.matrix_rank(cov) < 2:
        raise ValueError("점이 한 줄에 놓여 공분산을 뒤집을 수 없다")
    mean = points.mean(axis=0)
    return (float(mean[0]), float(mean[1])), (
        (float(cov[0, 0]), float(cov[0, 1])),
        (float(cov[1, 0]), float(cov[1, 1])),
    )


def summary(n: int, mean: Mean, cov: Cov) -> list[str]:
    """사람이 읽을 요약. 화면 가운데(가운데 단서 v1 의 기준)와 얼마나 어긋났는지를 같이 보인다."""
    sx, sy = cov[0][0] ** 0.5, cov[1][1] ** 0.5
    corr = cov[0][1] / (sx * sy)
    return [
        f"점 {n}개",
        f"평균 x {mean[0]:.3f} · y {mean[1]:.3f} "
        f"(화면 가운데보다 x {mean[0] - 0.5:+.3f} · y {mean[1] - 0.5:+.3f})",
        f"표준편차 x {sx:.3f} · y {sy:.3f} · 상관 {corr:+.2f}",
    ]


def constant_lines(mean: Mean, cov: Cov) -> list[str]:
    """cues.py 에 그대로 붙일 상수 두 줄. 소수 4자리."""
    (a, b), (c, d) = cov
    return [
        f"GAZE_MEAN = ({mean[0]:.4f}, {mean[1]:.4f})",
        f"GAZE_COV = (({a:.4f}, {b:.4f}), ({c:.4f}, {d:.4f}))",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 시선 분포(평균 · 공분산) 맞추기")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH, help="정답 표시 CSV")
    parser.add_argument("--raw", type=Path, default=RAW_DIR, help="촬영 사진 최상위 폴더")
    parser.add_argument(
        "--sessions", type=int, nargs="+", default=list(DEFAULT_SESSIONS), help="학습용 세션"
    )
    args = parser.parse_args()

    rows = train_rows(read_csv(args.labels), args.sessions)
    if not rows:
        raise SystemExit(
            f"세션 {args.sessions} 에 표시된 학습용(train) 줄이 없다: {args.labels}\n"
            "  먼저 python tools/label_targets.py --session 5 --split train"
        )
    # 사진 크기는 사진을 열어 읽는다. 세션마다 해상도가 다를 수 있다(1920×1080 · 1280×720).
    sizes = {}
    for r in rows:
        path = args.raw / r["file"]
        image = read_image(path) if path.exists() else None
        if image is None:
            raise SystemExit(f"사진을 못 읽었다: {path}")
        height, width = image.shape[:2]
        sizes[r["file"]] = (width, height)

    points = centers(rows, sizes)
    try:
        mean, cov = fit(points)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    print("\n".join(summary(len(points), mean, cov)))
    print("\n# src/jarviseo/pointing/cues.py 에 붙일 값")
    print("\n".join(constant_lines(mean, cov)))


if __name__ == "__main__":
    main()
