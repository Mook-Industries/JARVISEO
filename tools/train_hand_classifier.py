"""손 관절 표 → 가리키는 손 분류기(R1, MLP) 학습 · 사람 단위 3겹 평가.  담당: 최홍묵

tools/hand_clips.py 가 만든 landmarks.csv 로 "가리키는 손인가"를 학습한다(#223).

1. 사람 단위 3겹: p1 · p2 · p3 를 차례로 시험으로 빼고 나머지 두 사람으로 학습한다.
   같은 사람 손을 학습과 시험에 섞으면 처음 보는 사람 손에서의 실력보다 점수가 부풀려진다.
2. 문턱: 3겹에서 모은 확률로, 가리킴 재현율이 15° 규칙(R0) 이상이 되는 가장 높은 문턱을 고른다.
   valid 사진을 보고 고르지 않는다 — 고르는 순간 valid 가 시험이 아니게 된다.
3. 세 사람 전부로 다시 학습해 그 문턱과 함께 src/jarviseo/pointing/hand_r1.npz 로 내보낸다.

학습만 scikit-learn 을 쓴다(맥 · 윈도우에만 깔린다). 실행 쪽(hand_classifier.py)은 numpy 만 쓴다.

실행 (레포 맨 위 폴더에서):
    python tools/train_hand_classifier.py           # 3겹 표를 찍고 가중치를 내보낸다
    python tools/train_hand_classifier.py --force   # 이미 있는 가중치를 덮어쓴다
"""

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from jarviseo import config
from jarviseo.pointing.hand_classifier import HandClassifier, features
from jarviseo.pointing.hands import NUM_LANDMARKS, HandPose

HIDDEN = (64, 32)
SEED = 0
# 높은 문턱부터 본다. 0.01 까지 내려도 R0 재현율을 못 넘으면 가장 낮은 것을 쓴다.
THRESHOLDS = (0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01)


def pose_from_row(row: dict[str, str]) -> HandPose:
    """landmarks.csv 한 줄 → HandPose. 실행할 때와 같은 features 를 쓰려고 손 모양으로 되돌린다."""
    world = tuple(
        (float(row[f"w{i}_x"]), float(row[f"w{i}_y"]), float(row[f"w{i}_z"]))
        for i in range(NUM_LANDMARKS)
    )
    pixel = tuple((float(row[f"p{i}_x"]), float(row[f"p{i}_y"])) for i in range(NUM_LANDMARKS))
    return HandPose(
        landmarks=pixel, world=world, handedness=row["handedness"], score=float(row["hand_score"])
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    """이름표가 있는 손만. 손 없는 프레임과 뺀 손(가리킴 구간의 손 둘)은 뺀다."""
    with path.open(encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f) if row["label"] != ""]


def rates(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(가리킴 재현율, 오탐률). 해당 줄이 없으면 0."""
    positives, negatives = y == 1, y == 0
    recall = float(pred[positives].mean()) if positives.any() else 0.0
    false_alarm = float(pred[negatives].mean()) if negatives.any() else 0.0
    return recall, false_alarm


def pick_threshold(prob: np.ndarray, y: np.ndarray, target_recall: float) -> float:
    """재현율이 target 이상이 되는 가장 높은 문턱. 끝까지 못 넘으면 가장 낮은 후보."""
    for threshold in THRESHOLDS:
        if rates(prob >= threshold, y)[0] >= target_recall:
            return threshold
    return THRESHOLDS[-1]


def balanced_index(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """가리킴 줄을 아님 줄 수만큼 복원 추출로 늘린 줄 번호.

    가리킴 1 : 아님 9 라서 그대로 학습하면 "늘 아님"이 정확도 90% 로 이긴다.
    MLPClassifier 에는 class_weight 가 없어서 줄 수로 맞춘다.
    """
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    extra = rng.choice(positives, size=len(negatives), replace=True)
    return np.concatenate([negatives, extra])


def fit_mlp(x: np.ndarray, y: np.ndarray, seed: int = SEED):
    """표준화 + MLP. sklearn 은 여기서만 불러온다(CI · 서버에는 없다)."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    index = balanced_index(y, np.random.default_rng(seed))
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(HIDDEN, max_iter=500, early_stopping=True, random_state=seed),
    )
    return model.fit(x[index], y[index])


def fit_logistic(x: np.ndarray, y: np.ndarray):
    """참고용 비교 — 같은 특징으로 선형 모델만 쓰면 어디까지 되나."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
    )
    return model.fit(x, y)


def out_of_fold(fit, x: np.ndarray, y: np.ndarray, person: np.ndarray) -> np.ndarray:
    """사람 하나씩 빼고 학습해 그 사람의 확률을 모은다."""
    prob = np.zeros(len(y))
    for held in sorted(set(person)):
        test = person == held
        prob[test] = fit(x[~test], y[~test]).predict_proba(x[test])[:, 1]
    return prob


def to_classifier(model, threshold: float, note: str) -> HandClassifier:
    """sklearn 파이프라인 → numpy 로 계산하는 HandClassifier."""
    scaler, mlp = model[0], model[-1]
    return HandClassifier(
        mean=scaler.mean_,
        scale=scaler.scale_,
        weights=tuple(mlp.coefs_),
        biases=tuple(mlp.intercepts_),
        threshold=threshold,
        note=note,
    )


def table(
    name: str, pred: np.ndarray, y: np.ndarray, groups: Sequence[tuple[str, np.ndarray]]
) -> str:
    """한 방법의 전체 · 묶음별 재현율 / 오탐률 한 줄."""
    recall, false_alarm = rates(pred, y)
    parts = [f"{name:14} 전체 {recall:6.1%} / {false_alarm:5.1%}"]
    for group, mask in groups:
        r, f = rates(pred[mask], y[mask])
        parts.append(
            f"{group} {r:6.1%} / {f:5.1%}" if (y[mask] == 1).any() else f"{group} - / {f:5.1%}"
        )
    return " · ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="손 관절 → 가리키는 손 분류기(R1 MLP)")
    parser.add_argument(
        "--landmarks", type=Path, default=config.DATASETS_DIR / "hand" / "landmarks.csv"
    )
    parser.add_argument("--out", type=Path, default=config.HAND_CLASSIFIER_PATH)
    parser.add_argument("--force", action="store_true", help="이미 있는 가중치를 덮어쓴다")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists() and not args.force:
        raise SystemExit(
            f"이미 있습니다: {args.out} — 덮어쓰려면 --force (회차 가중치가 조용히 바뀌지 않게)"
        )
    rows = load_rows(args.landmarks)
    x = np.array([features(pose_from_row(row)) for row in rows])
    y = np.array([int(row["label"]) for row in rows])
    rule = np.array([int(row["rule"]) for row in rows])
    person = np.array([row["person"] for row in rows])
    cls = np.array([row["cls"] for row in rows])

    r0_recall, _ = rates(rule == 1, y)
    mlp_prob = out_of_fold(fit_mlp, x, y, person)
    threshold = pick_threshold(mlp_prob, y, r0_recall)
    logistic_prob = out_of_fold(fit_logistic, x, y, person)
    logistic_threshold = pick_threshold(logistic_prob, y, r0_recall)

    people = [(p, person == p) for p in sorted(set(person))]
    shapes = [(c, cls == c) for c in ("point", "hold", "open", "fist", "rest", "other")]
    print(f"손 {len(y)}개 (가리킴 {int(y.sum())} · 아님 {int((y == 0).sum())}) — 사람 단위 3겹")
    print(f"문턱: MLP {threshold} · 로지스틱 {logistic_threshold} (R0 재현율 {r0_recall:.1%} 이상)")
    for title, groups in (("사람별 재현율 / 오탐률", people), ("손 모양별", shapes)):
        print(f"\n[{title}]")
        print("  " + table("R0 15° 규칙", rule == 1, y, groups))
        print("  " + table("R1 MLP", mlp_prob >= threshold, y, groups))
        print("  " + table("로지스틱(참고)", logistic_prob >= logistic_threshold, y, groups))

    recall, false_alarm = rates(mlp_prob >= threshold, y)
    note = (
        f"R1 MLP{HIDDEN} seed {SEED} · 손 {len(y)} · 사람 3겹 문턱 {threshold} "
        f"재현율 {recall:.3f} 오탐률 {false_alarm:.3f} "
        f"(R0 {r0_recall:.3f} / {rates(rule == 1, y)[1]:.3f})"
    )
    classifier = to_classifier(fit_mlp(x, y), threshold, note)
    classifier.save(args.out)
    print(f"\n저장: {args.out} ({args.out.stat().st_size / 1024:.0f}KB) — {note}")


if __name__ == "__main__":
    main()
