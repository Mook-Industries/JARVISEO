"""가리킴 ablation — 정답 표시한 사진으로 v1~v5 를 채점하고 기록한다.  담당: 최홍묵

tools/label_targets.py 가 만든 labels.csv 의 사진마다 한 번 검출하고, 버전마다 켤 단서만 켜서
resolve_target 으로 고른 박스를 정답 박스와 비교한다(IoU 0.5 이상이면 맞음).
결과는 DB eval_run(가중치 × 버전마다 한 줄, 지표 묶음) · eval_sample(사진마다 맞음/틀림)과
사진별 CSV(실패 분류 · margin · 크롭 담김 · 지연)에 남기고, 같은 조건의 이전 실행과 비교표를 찍는다.
대표 지표는 정답률과 확신 오답률(되묻지 않고 골랐는데 틀림) 한 쌍이다.

버전은 단서를 하나씩 쌓아 켠다(types.CueKind 순서). 아직 구현 안 된 단서가 든 버전은 건너뛴다.
v2+C 는 단서가 아니라 대비책(손 우선 크롭, #205)을 v2 위에 얹은 임시 버전이다. 손 크롭은 박스를
고르지 않아 IoU 로는 맞을 수 없으므로, 정답률 · 확신 오답률에 섞지 않고 크롭 담김으로만 센다.
v2+lang · v2+lang+ctx 는 시선(v3) 없이 말 단서 · 이어 묻기 단서를 v2 위에 쌓은 임시 버전이다(#166).
v3 이 생긴 뒤(#245)에도 남긴다. 실제 비서의 기본값(시선 · 이어 묻기는 꺼 둠)이 v2+lang 이다.
고르는 말 · 직전 대상이 든 발화는 말 · 맥락 세션(s04)에 있어서 --sessions 4 로 따로 잰다.
검출은 가중치마다 사진마다 한 번만 해서, 버전 차이가 단서에서만 나오게 한다.
STT · VLM 없이 촬영 때 적은 발화로 채점해서, 가리킴 모델 실력만 따로 잰다.

검출기 회차(R0 · R1 · R2 …)는 --weights 에 같이 적어 한 실행에서 나란히 잰다(#167).
같은 코드 · 같은 사진 · 같은 손으로 재야 표의 차이가 가중치 몫이 된다. 첫 가중치가 비교 기준이다.
무엇으로 쟀는지(커밋 · 가중치 지문 · 세션 · 장비)는 eval_run.metrics["conditions"] 에 남는다.

실행 (레포 맨 위 폴더에서):
    python scripts/run_ablation.py --note "v1 기준선"
    python scripts/run_ablation.py --variants v1 v2 --note "v2 손 방향 추가"
    python scripts/run_ablation.py --variants v1 v2 v2+C --note "손 우선 크롭"
    python scripts/run_ablation.py --weights yolov8n.pt data/models/detector_r1.pt --note "R0 vs R1"
    python scripts/run_ablation.py --sessions 4 --variants v2 v2+lang v2+lang+ctx \
        --weights yolov8n.pt data/models/detector_r2.pt --note "말 · 맥락 세션"
    python scripts/run_ablation.py --variants v2 v3 v4 v5 \
        --weights yolov8n.pt data/models/detector_r2.pt --note "시선 단서"
    python scripts/run_ablation.py --split test --note "최종"     # 10/24 최종 측정 때만
"""

import argparse
import csv
import hashlib
import platform
import subprocess
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select

from jarviseo import config
from jarviseo.memory import EvalRun, EvalSample, MemoryStore
from jarviseo.pointing.crop import crop_region, hand_crop_region
from jarviseo.pointing.hands import HandDetector, HandPose, pick_pointing_hand
from jarviseo.pointing.resolver import IMPLEMENTED_CUES, resolve_target
from jarviseo.types import BBox, CueKind, Detection, TargetResolution

POINTING_DIR = config.DATASETS_DIR / "pointing"
# --sessions 를 안 주면 재는 세션. valid 62장 = 세션 1 · 2. 말 · 맥락 세션(s04)을 표시해도
# 명령을 안 바꾸면 62장 숫자가 그대로라서, 회차 비교 기준이 흔들리지 않는다.
DEFAULT_SESSIONS: dict[str, tuple[int, ...]] = {"valid": (1, 2), "test": (3,)}
# 단서를 하나씩 쌓아 켠다. ablation 표의 한 줄이 한 버전이다.
VARIANTS: dict[str, frozenset[CueKind]] = {
    "v1": frozenset({CueKind.CENTER}),
    "v2": frozenset({CueKind.CENTER, CueKind.POINT}),
    "v3": frozenset({CueKind.CENTER, CueKind.POINT, CueKind.GAZE}),
    "v4": frozenset({CueKind.CENTER, CueKind.POINT, CueKind.GAZE, CueKind.LANG}),
    "v5": frozenset(CueKind),
    # 임시 버전: 단서는 v2 그대로, 선 위에 후보가 없으면 손 우선 크롭(v2+lang 과 같은 방식의 이름).
    "v2+C": frozenset({CueKind.CENTER, CueKind.POINT}),
    # 임시 버전: v3(시선)이 없던 동안 누적 버전 v4 · v5 대신 v2 위에 쌓아 쟀다(10/22 결정 ④).
    # v3 이 생긴 뒤(#245)에도 남긴다 — 시선을 꺼 둔 실제 비서의 기본값이 v2+lang 이라 비교 기준이다.
    "v2+lang": frozenset({CueKind.CENTER, CueKind.POINT, CueKind.LANG}),
    "v2+lang+ctx": frozenset({CueKind.CENTER, CueKind.POINT, CueKind.LANG, CueKind.CTX}),
}
# 손 우선 크롭을 켜는 버전. 단서가 아니라 대비책이라 단서 표(VARIANTS)와 따로 적는다.
HAND_CROP_VARIANTS = frozenset({"v2+C"})
# 검출 평가의 표준 기준. 고른 박스와 정답 박스가 절반 넘게 겹치면 맞음.
IOU_MATCH = 0.5
# 틀렸는데 margin 이 이 이상이면 "높은 확신으로 틀림". 되묻지 않고 단정한 오답이다.
HIGH_CONFIDENCE = 0.5
# 고른 박스의 크롭(실제 비서와 같은 여백 15%)이 정답 박스를 이만큼 담으면 "크롭 담김".
# 손 우선 크롭처럼 정답 박스와 딱 맞지 않아도 VLM 이 답할 수 있는 크롭을 IoU 와 따로 잰다.
CROP_CONTAINED = 0.6
# 크롭이 사진의 이만큼을 넘게 덮으면 담김으로 세지 않는다. 무엇이든 담기기 때문이다.
# valid 에서 정답 크롭은 넓이 41% 이하, 냉장고 · 노트북 헛 박스 크롭은 53% 이상이었다.
CROP_MAX_AREA = 0.5
# 손 우선 크롭으로 답한 사진. 실패도 정답도 아니라 크롭 담김(crop_contained)으로 따로 본다.
HAND_CROP = "손 방향 크롭"
# 실패 분류(멘토링 2차의 6분류에서 "후보 없음"을 둘로 나눔) + 정답 + 손 방향 크롭.
# 손 방향 크롭이 아니면 위에서 먼저 걸리는 것으로 정한다(judge).
OUTCOMES = (
    "정답",
    "되묻는 게 맞음",
    "후보 없어 되물음",
    "후보 없는데 답함",
    "불필요한 되묻기",
    "방향 오판",
    "높은 확신으로 틀림",
    "다른 후보 1위",
    HAND_CROP,
)
CSV_COLUMNS = [
    "file",
    "gt_label",
    "pred_label",
    "outcome",
    "correct",
    "margin",
    "has_hand",
    "hand_used",
    "ambiguous",
    "candidate_hit",
    "crop_covered",
    "crop_area",
    "crop_contained",
    "hand_crop",
    "detect_ms",
    "resolve_ms",
]


def load_samples(path: Path, split: str, sessions: list[int]) -> list[dict]:
    """labels.csv 에서 그 split · 세션의 쓸 수 있는 사진만 읽어, 정답 박스 · 표시를 풀어 둔다."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    samples = []
    for row in rows:
        # session 은 CSV 라 글자다. 정수로 바꿔야 [1, 2] 와 비교된다.
        if row["split"] != split or row["excluded"] == "1" or int(row["session"]) not in sessions:
            continue
        samples.append(
            {
                "file": row["file"],
                "utterance": row["utterance"],
                "gt": BBox(*(float(row[k]) for k in ("gt_x1", "gt_y1", "gt_x2", "gt_y2"))),
                "gt_label": row["gt_label"],
                # 0/1 로 저장했다. bool("0") 은 참이라 글자로 비교한다.
                "has_hand": row["has_hand"] == "1",
                "ambiguous": row["ambiguous"] == "1",
                "prev_label": row["prev_label"] or None,
            }
        )
    return samples


def implemented_variants(names: list[str]) -> tuple[list[str], list[str]]:
    """(돌릴 버전, 미구현이라 건너뛸 버전). 구현 안 된 단서를 켜면 resolve_target 이 멈춘다."""
    ready = [v for v in names if VARIANTS[v] <= IMPLEMENTED_CUES]
    return ready, [v for v in names if v not in ready]


def judge(
    sample: dict, resolution: TargetResolution, candidate_hit: bool, uses_point: bool
) -> tuple[bool, str]:
    """(맞음 여부, 결과 분류). 위에서 먼저 걸리는 것으로 정한다.

    정답이 후보에 없으면 단서가 아무리 좋아도 못 맞히므로, 되묻기보다 "후보 없음"을 먼저 본다.
    검출기 몫의 실패를 단서 몫으로 세면 v2~v5 의 개선이 흐려진다.
    그 안에서 되물었는지(안전)와 엉뚱한 물건을 단정했는지(위험)는 나눈다. 하나로 묶으면
    단정한 오답이 "후보 없음"에 묻혀 높은 확신 오답이 0% 로 보인다.
    resolver 는 되물을 때 chosen 이 늘 None 이다(후보 0개도 되물음).
    """
    chosen = resolution.chosen
    if chosen is not None and chosen.detection.bbox.iou(sample["gt"]) >= IOU_MATCH:
        return True, "정답"
    if resolution.needs_clarify and sample["ambiguous"]:
        return True, "되묻는 게 맞음"  # 사람도 못 고르는 장면에서는 묻는 게 정답
    if not candidate_hit:
        return False, "후보 없어 되물음" if resolution.needs_clarify else "후보 없는데 답함"
    if resolution.needs_clarify:
        return False, "불필요한 되묻기"
    if sample["has_hand"] and uses_point:
        return False, "방향 오판"
    if resolution.margin >= HIGH_CONFIDENCE:
        return False, "높은 확신으로 틀림"
    return False, "다른 후보 1위"


def crop_coverage(gt: BBox, region: BBox, frame_size: tuple[int, int]) -> tuple[float, float]:
    """(정답 박스 중 크롭 안에 든 비율, 크롭 넓이 ÷ 사진 넓이). frame_size 는 (너비, 높이).

    IoU 가 아니라 "정답이 얼마나 담겼나"를 본다. 크롭이 정답보다 커도 VLM 은 그 안에서 답할 수 있다.
    겹친 넓이는 types.BBox 를 바꾸지 않고(세 사람의 계약) 여기서 계산한다.
    """
    inter_w = max(0.0, min(gt.x2, region.x2) - max(gt.x1, region.x1))
    inter_h = max(0.0, min(gt.y2, region.y2) - max(gt.y1, region.y1))
    covered = inter_w * inter_h / gt.area if gt.area > 0 else 0.0
    width, height = frame_size
    return covered, region.area / (width * height)


def is_crop_contained(covered: float | None, area: float | None) -> bool:
    """정답을 CROP_CONTAINED 이상 담고 크롭이 사진의 CROP_MAX_AREA 이하이면 담김.

    되물어서 크롭이 없으면(None) 담김이 아니다.
    """
    if covered is None or area is None:
        return False
    return covered >= CROP_CONTAINED and area <= CROP_MAX_AREA


def evaluate(
    samples: list[dict],
    candidates: dict[str, list[Detection]],
    detect_ms: dict[str, float],
    variant: str,
    frame_size: dict[str, tuple[int, int]],
    hands: dict[str, HandPose | None],
    raw: Path | None = None,
) -> list[dict]:
    """한 버전으로 모든 사진을 채점한다. 검출 · 손은 밖에서 한 번 해 둔 결과를 쓴다.

    말 단서를 켠 버전은 색("노란 캔")을 보려고 raw 폴더에서 사진을 다시 읽는다.
    main 은 메모리를 아끼려고 검출한 사진을 버리고, 색은 박스 안의 원본 픽셀이 필요하다.
    raw 가 없으면 색만 기권한다.
    """
    cues = VARIANTS[variant]
    rows = []
    for s in samples:
        found = candidates[s["file"]]
        size = frame_size[s["file"]]
        hit = any(d.bbox.iou(s["gt"]) >= IOU_MATCH for d in found)
        # 시간을 재기 전에 읽는다. 실제 비서는 사진을 이미 들고 있어서
        # 디스크 읽기는 resolve 지연이 아니다.
        image = None
        if raw is not None and CueKind.LANG in cues:
            image = read_image(raw / s["file"])
        started = time.perf_counter()
        resolution = resolve_target(
            candidates=found,
            hand=hands[s["file"]],
            gaze_vector=None,
            text=s["utterance"],
            frame_size=size,
            prev_target_label=s["prev_label"],
            enabled_cues=set(cues),
            image=image,
        )
        # 실제 비서는 손 크롭 판단도 resolve 단계 안에서 한다. 같이 재야 지연이 맞는다.
        hand_crop = None
        if variant in HAND_CROP_VARIANTS:
            hand_crop = hand_crop_region(found, hands[s["file"]], size)
        resolve_ms = (time.perf_counter() - started) * 1000
        # 손 단서를 켰어도 손이 없으면 기권한다. 실제로 점수를 낸 사진만 "방향 오판"으로 센다.
        hand_used = any(
            cs.kind is CueKind.POINT for c in resolution.candidates[:1] for cs in c.cue_scores
        )
        if hand_crop is not None:
            # 박스를 고르지 않았으니 IoU 로는 맞을 수 없다. 정답 · 확신 오답 · 답함에 안 넣고
            # 크롭 담김으로만 본다. 빗나간 손 크롭은 "손 크롭 n장(담김 m)"의 n − m 으로 보인다.
            correct, outcome, chosen, region = False, HAND_CROP, None, hand_crop
        else:
            correct, outcome = judge(s, resolution, hit, hand_used)
            chosen = resolution.chosen
            region = crop_region(chosen.detection.bbox, size) if chosen is not None else None
        # 되물었으면 VLM 에 보낼 크롭이 없다 → 못 담은 것으로 센다(비율 · 넓이는 비움).
        covered = area = None
        if region is not None:
            covered, area = crop_coverage(s["gt"], region, size)
        rows.append(
            {
                "file": s["file"],
                "gt_label": s["gt_label"],
                "pred_label": chosen.detection.label if chosen is not None else None,
                "outcome": outcome,
                "correct": correct,
                "margin": round(resolution.margin, 3),
                "has_hand": s["has_hand"],
                "hand_used": hand_used,
                "ambiguous": s["ambiguous"],
                "candidate_hit": hit,
                "crop_covered": None if covered is None else round(covered, 3),
                "crop_area": None if area is None else round(area, 3),
                "crop_contained": is_crop_contained(covered, area),
                "hand_crop": hand_crop is not None,
                "answered": chosen is not None,
                # 손 크롭이면 되묻지 않고 그 크롭으로 답한다. 되묻기율이 실제로 준다.
                "clarified": resolution.needs_clarify and hand_crop is None,
                # 되묻지 않고 골랐는데 틀림. margin 은 안 본다 —
                # 0.15 를 넘으면 비서는 똑같이 단정하는 말투로 답한다.
                "confident_error": chosen is not None and not correct,
                "detect_ms": round(detect_ms[s["file"]], 1),
                "resolve_ms": round(resolve_ms, 3),
            }
        )
    return rows


def missing_weights(weights: list[str]) -> list[str]:
    """폴더까지 적었는데 파일이 없는 가중치.

    이름만 적은 것(yolov8n.pt)은 처음 쓰는 PC 에서 ultralytics 가 받아 오게 둔다.
    """
    return [w for w in weights if Path(w).name != w and not Path(w).exists()]


def weights_fingerprint(path: Path) -> dict:
    """가중치 파일 이름과 내용 지문(sha256 앞 12자리).

    Colab 은 회차마다 같은 이름(best.pt)을 줘서, 이름만으로는 R1 과 R2 를 가를 수 없다.
    """
    with Path(path).open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"name": Path(path).name, "sha256": digest[:12]}


def weights_label(fingerprint: dict) -> str:
    """표에 찍는 가중치 이름. 예: detector_r1.pt(3fa2c1)"""
    return f"{fingerprint['name']}({fingerprint['sha256'][:6]})"


def git_commit() -> str:
    """지금 코드의 짧은 커밋 번호. 커밋 안 한 수정이 있으면 +dirty. git 을 못 부르면 unknown.

    고친 채로 재면 같은 커밋 번호라도 코드가 다르다. 기록만 보고 "같은 코드였나"를 가르려고 붙인다.
    """
    git = {
        "cwd": config.PROJECT_ROOT,
        "capture_output": True,
        "encoding": "utf-8",
        "errors": "replace",  # 윈도우 git 이 cp949 로 오류 문장을 내도 여기서 멈추지 않게
        "check": False,
    }
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], **git)
        status = subprocess.run(["git", "status", "--porcelain"], **git)
    except FileNotFoundError:  # git 이 PATH 에 없다(윈도우에서 흔함)
        return "unknown"
    if head.returncode != 0:
        return "unknown"
    return head.stdout.strip() + ("+dirty" if status.stdout.strip() else "")


def package_version(name: str) -> str:
    """설치된 패키지 버전. import 하지 않고 읽어서 CI(ultralytics 없음)에서도 돈다."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "none"


def run_conditions(weights: dict, sessions: list[int], conf: float) -> dict:
    """이번 실행을 무엇으로 쟀나. eval_run.metrics["conditions"] 에 들어간다.

    ERD 가 원본이라 열을 늘리지 않고 metrics(JSON)에 둔다.
    컴퓨터 이름은 비교에 필요 없어 넣지 않는다.
    """
    return {
        "commit": git_commit(),
        "weights": weights,
        "sessions": sorted(sessions),
        "conf": conf,
        "device": f"{platform.system()} {platform.machine()}",
        "ultralytics": package_version("ultralytics"),
    }


def same_conditions(old: dict | None, new: dict) -> bool:
    """같은 가중치(지문) · 같은 세션이면 같은 조건이다.

    커밋은 안 본다. 같은 가중치에서 코드를 고친 효과를 보려는 비교다.
    조건 기록이 없는 옛 실행(run 1~3)은 무엇으로 쟀는지 몰라서 비교하지 않는다.
    """
    if not old:
        return False
    same_weights = old.get("weights", {}).get("sha256") == new["weights"]["sha256"]
    return same_weights and old.get("sessions") == new["sessions"]


def hit_changes(base: dict[str, bool], other: dict[str, bool]) -> tuple[list[str], list[str]]:
    """기준 가중치 대비 (새로 잡힌 사진, 새로 놓친 사진). 값은 사진마다 "정답이 후보에 있나".

    재현율 숫자만 보면 "+3장"이 사실은 7장 얻고 4장 잃은 것일 수 있다. 잃은 사진을 열어 봐야
    R3 에서 보강할 장면이 보인다.
    """
    gained = sorted(f for f, hit in other.items() if hit and not base.get(f))
    lost = sorted(f for f, hit in base.items() if hit and not other.get(f))
    return gained, lost


def percentiles(values: list[float]) -> tuple[float, float]:
    """p50 · p95. measure_latency.py · store.latency_percentiles 와 같은 nearest-rank."""
    ordered = sorted(values)

    def pick(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return pick(0.50), pick(0.95)


def _rate(part: list[dict], key: str) -> float | None:
    """part 에서 key 가 참인 비율. part 가 비면 None(0 으로 두면 "전부 틀림"으로 읽힌다)."""
    return round(sum(r[key] for r in part) / len(part), 3) if part else None


def summarize(rows: list[dict]) -> dict:
    """한 버전의 지표. eval_run.metrics 에 그대로 들어간다.

    확신 오답률 · 크롭 담김률의 분모는 정답률처럼 전체 장수다. 되물은 사진을 빼면
    많이 되묻는 버전일수록 비율이 부풀려진다. high_conf_error_rate 는 run 1~7 과 비교하려고 남긴다.
    """
    counts = Counter(r["outcome"] for r in rows)
    detect_p50, detect_p95 = percentiles([r["detect_ms"] for r in rows])
    resolve_p50, resolve_p95 = percentiles([r["resolve_ms"] for r in rows])
    return {
        "accuracy": _rate(rows, "correct"),
        "confident_error_rate": _rate(rows, "confident_error"),
        "answered_accuracy": _rate([r for r in rows if r["answered"]], "correct"),
        "reask_rate": _rate(rows, "clarified"),
        "candidate_recall": _rate(rows, "candidate_hit"),
        "crop_contained_rate": _rate(rows, "crop_contained"),
        "accuracy_with_hand": _rate([r for r in rows if r["has_hand"]], "correct"),
        "accuracy_without_hand": _rate([r for r in rows if not r["has_hand"]], "correct"),
        "high_conf_error_rate": round(counts["높은 확신으로 틀림"] / len(rows), 3),
        # 장수가 작아 비율보다 "9(9)"처럼 읽는 편이 낫다. 손 크롭을 안 켠 버전은 0(0).
        "hand_crop_count": sum(r["hand_crop"] for r in rows),
        "hand_crop_contained": sum(r["hand_crop"] and r["crop_contained"] for r in rows),
        "outcomes": {name: counts[name] for name in OUTCOMES},
        "detect_p50_ms": detect_p50,
        "detect_p95_ms": detect_p95,
        "resolve_p50_ms": resolve_p50,
        "resolve_p95_ms": resolve_p95,
    }


def save_run(store: MemoryStore, variant: str, rows: list[dict], metrics: dict, note: str) -> int:
    """eval_run 한 줄과 eval_sample 사진 수만큼을 같은 트랜잭션으로 남기고 run_id 를 돌려준다."""
    with store.session() as db:
        run = EvalRun(variant=variant, sample_count=len(rows), metrics=metrics, note=note or None)
        db.add(run)
        db.flush()  # run_id 를 받아야 eval_sample 이 가리킬 수 있다
        for r in rows:
            # turn_id 는 비운다. 실제 대화 턴이 아니라 평가 사진이다.
            db.add(
                EvalSample(
                    run_id=run.run_id,
                    gt_label=r["gt_label"],
                    pred_label=r["pred_label"],
                    is_correct=r["correct"],
                )
            )
        return run.run_id


def previous_metrics(
    store: MemoryStore, variant: str, before: int, conditions: dict
) -> dict | None:
    """같은 버전 · 같은 조건의 바로 이전 실행의 지표 묶음. 없으면 None.

    가중치가 다른 실행과 비교하면 그 차이가 코드 몫인지 가중치 몫인지 표만 보고 모른다.
    JSON 안의 값으로 거르는 SQL 은 SQLite 와 Postgres 문법이 달라서,
    같은 버전을 다 가져와 여기서 거른다. 정답률 · 확신 오답률을 같이 비교하려고 묶음째 돌려준다.
    """
    with store.session() as db:
        stmt = (
            select(EvalRun.metrics)
            .where(EvalRun.variant == variant, EvalRun.run_id < before)
            .order_by(EvalRun.run_id.desc())
        )
        history = list(db.scalars(stmt))
    for metrics in history:
        if same_conditions((metrics or {}).get("conditions"), conditions):
            return metrics
    return None


def write_csv(out_dir: Path, run_id: int, variant: str, rows: list[dict]) -> Path:
    """사진별 결과. 파일 이름의 run_id 로 DB 의 실행과 짝지어 실패 사진을 찾아간다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}_{variant}.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}"


def _with_delta(metrics: dict, previous: dict | None, key: str) -> str:
    """지표 하나를 %로, 이전 실행에도 그 값이 있으면 차이(%p)를 붙여서.

    확신 오답률이 없던 옛 실행(run 4~7)과는 차이를 찍지 않는다.
    없는 값을 0 으로 빼면 엉뚱한 차이가 된다.
    """
    text = _pct(metrics[key])
    old = (previous or {}).get(key)
    if old is not None and metrics[key] is not None:
        text += f" ({(metrics[key] - old) * 100:+.1f})"
    return text


def format_row(weights: str, variant: str, metrics: dict, n: int, previous: dict | None) -> str:
    """비교표 한 줄. 대표 지표 정답률 · 확신 오답률을 붙여 두고 이전 실행 대비 차이(%p)를 단다.

    정답률만 보면 되묻기를 줄여 단정한 오답이 늘어도 개선처럼 보인다
    (되묻기 기준 0.15 → 0.05: 정답 +1 · 확신 오답 +7).
    """
    hand = f"{_pct(metrics['accuracy_with_hand'])}/{_pct(metrics['accuracy_without_hand'])}"
    hand_crop = f"{metrics['hand_crop_count']}({metrics['hand_crop_contained']})"
    return (
        f"{weights} | {variant} | {n} | {_with_delta(metrics, previous, 'accuracy')} | "
        f"{_with_delta(metrics, previous, 'confident_error_rate')} | "
        f"{_pct(metrics['answered_accuracy'])} | {_pct(metrics['reask_rate'])} | {hand} | "
        f"{_pct(metrics['candidate_recall'])} | {_pct(metrics['crop_contained_rate'])} | "
        f"{hand_crop} | {metrics['resolve_p50_ms']:.2f}/{metrics['resolve_p95_ms']:.2f}"
    )


def read_image(path: Path) -> np.ndarray | None:
    """cv2.imread 는 윈도우에서 경로에 한글이 있으면 조용히 실패해서, 바이트로 읽어 푼다."""
    data = np.fromfile(path, dtype=np.uint8) if path.exists() else np.empty(0, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description="가리킴 v1~v5 ablation")
    parser.add_argument("--labels", type=Path, default=POINTING_DIR / "labels.csv")
    parser.add_argument("--raw", type=Path, default=POINTING_DIR / "raw")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument(
        "--sessions", type=int, nargs="+", help="잴 세션. 안 주면 valid 는 1 2, test 는 3"
    )
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument(
        "--weights",
        nargs="+",
        default=[config.DETECTOR_WEIGHTS],
        help="검출기 가중치. 여러 개면 한 실행에서 나란히 재고 첫 번째와 비교한다",
    )
    parser.add_argument("--db", default=config.DATABASE_URL, help="결과를 남길 DB")
    parser.add_argument("--note", default="", help="이번 실행 메모, 예: v2 손 방향 추가")
    parser.add_argument("--out", type=Path, default=POINTING_DIR / "runs", help="사진별 CSV 폴더")
    args = parser.parse_args()

    if args.split == "test":
        print("주의: test 를 연다 — 최종 측정 때만. 보면서 고치면 test 숫자가 실력이 아니게 된다.")
    sessions = sorted(args.sessions or DEFAULT_SESSIONS[args.split])
    samples = load_samples(args.labels, args.split, sessions)
    if not samples:
        raise SystemExit(f"{args.split} 세션 {sessions} 로 표시된 사진이 없다: {args.labels}")
    ready, skipped = implemented_variants(args.variants)
    if skipped:
        print(f"미구현이라 건너뜀: {', '.join(skipped)}")
    if not ready:
        raise SystemExit("돌릴 수 있는 버전이 없다")
    missing = missing_weights(args.weights)
    if missing:
        raise SystemExit(
            f"가중치 파일이 없다: {', '.join(missing)} — "
            "Drive 의 runs/rN/weights/best.pt 를 data/models/detector_rN.pt 로 받는다"
        )

    # ultralytics 는 CI 에 없어서 여기서 불러온다.
    from jarviseo.pointing.detector import ObjectDetector

    # 손 단서를 켠 버전이 있을 때만 손을 찾는다. 모델 파일이 없어도 v1 은 돌아야 한다.
    hand_detector = None
    if any(CueKind.POINT in VARIANTS[v] for v in ready):
        hand_detector = HandDetector()
    # 쓰는 쪽이라 테이블이 없으면 만든다(이미 있으면 아무것도 안 한다).
    store = MemoryStore(database_url=args.db)
    store.init_schema()

    frame_size, hands, hand_ms = {}, {}, []
    seen: dict[str, str] = {}  # 지문 → 명령에 적은 가중치
    base = None  # 첫 가중치의 (이름, 후보 재현율, 사진별 후보 적중)
    for index, weights in enumerate(args.weights):
        detector = ObjectDetector(weights)
        # 첫 호출은 모델을 올리느라 느리다(실측 861ms, 그다음 30ms 안팎). 재기 전에 한 번 버린다.
        detector.detect(np.zeros((640, 640, 3), np.uint8))
        fingerprint = weights_fingerprint(detector.weights_path)
        same = seen.get(fingerprint["sha256"])
        if same is not None:
            raise SystemExit(f"같은 가중치를 두 번 적었다: {weights} = {same}")
        seen[fingerprint["sha256"]] = weights
        label = weights_label(fingerprint)
        conditions = run_conditions(fingerprint, sessions, detector.conf_threshold)

        # 사진은 가중치마다 다시 읽는다. 1080p 62장을 다 들고 있으면 약 370MB 라
        # 저사양 PC 에 무겁다.
        candidates, detect_ms = {}, {}
        for i, s in enumerate(samples):
            image = read_image(args.raw / s["file"])
            if image is None:
                raise SystemExit(f"사진을 못 읽었다: {args.raw / s['file']}")
            started = time.perf_counter()
            candidates[s["file"]] = detector.detect(image)
            detect_ms[s["file"]] = (time.perf_counter() - started) * 1000
            # 손은 YOLO 가중치와 상관없다. 첫 가중치 때 한 번만 찾고 모든 가중치가 같은 손을 쓴다.
            if index > 0:
                continue
            height, width = image.shape[:2]
            frame_size[s["file"]] = (width, height)
            hands[s["file"]] = None
            if hand_detector is not None:
                if i == 0:
                    hand_detector.detect(image)  # 첫 호출은 모델을 올리느라 느려서 버린다
                started = time.perf_counter()
                hands[s["file"]] = pick_pointing_hand(hand_detector.detect(image))
                hand_ms.append((time.perf_counter() - started) * 1000)

        # 검출은 모든 버전이 같은 결과를 쓰므로 버전 줄마다가 아니라 가중치마다 한 번만 찍는다.
        detect_p50, detect_p95 = percentiles(list(detect_ms.values()))
        print(
            f"\n{label} · {args.split} {len(samples)}장 (세션 {' '.join(map(str, sessions))}) · "
            f"검출 p50 {detect_p50:.1f}ms · p95 {detect_p95:.1f}ms"
        )
        if index == 0 and hand_ms:
            hand_p50, hand_p95 = percentiles(hand_ms)
            found = sum(h is not None for h in hands.values())
            print(f"가리키는 손 {found}장 · 손 p50 {hand_p50:.1f}ms · p95 {hand_p95:.1f}ms")
        print(
            "가중치 | 버전 | 장수 | 정답률 | 확신오답률 | 답한 것 정답률 | 되묻기율 | "
            "손있음/손없음 | 후보재현율 | 크롭담김 | 손크롭(담김) | resolve p50/p95 ms"
        )
        for variant in ready:
            rows = evaluate(
                samples, candidates, detect_ms, variant, frame_size, hands, raw=args.raw
            )
            metrics = summarize(rows)
            metrics["conditions"] = conditions
            run_id = save_run(store, variant, rows, metrics, args.note)
            previous = previous_metrics(store, variant, run_id, conditions)
            print(format_row(label, variant, metrics, len(rows), previous))
            path = write_csv(args.out, run_id, variant, rows)
            # 손 방향 크롭은 실패가 아니라 따로 센 것이라 빼고 찍는다(표의 손크롭 칸).
            not_failed = (*OUTCOMES[:2], HAND_CROP)
            failed = {k: v for k, v in metrics["outcomes"].items() if v and k not in not_failed}
            print(f"    run {run_id} · 실패 {failed or '없음'} · {path}")

        # 정답이 후보에 있나는 단서와 상관없는 검출기 숫자라, 어느 버전의 결과로 봐도 같다.
        hits = {r["file"]: r["candidate_hit"] for r in rows}
        if base is None:
            base = (label, metrics["candidate_recall"], hits)
            continue
        base_label, base_recall, base_hits = base
        gained, lost = hit_changes(base_hits, hits)
        change = (metrics["candidate_recall"] - base_recall) * 100
        print(
            f"\n{label} vs 기준 {base_label}: 후보 재현율 {_pct(base_recall)} → "
            f"{_pct(metrics['candidate_recall'])} ({change:+.1f}%p) · "
            f"새로 잡힌 {len(gained)}장 · 새로 놓친 {len(lost)}장"
        )
        for mark, files in (("+", gained), ("-", lost)):
            for file in files:
                print(f"    {mark} {file}")


if __name__ == "__main__":
    main()
