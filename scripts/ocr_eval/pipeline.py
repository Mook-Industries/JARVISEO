"""성분표 판정 전체 평가 — 사진 한 장을 웹 서버와 똑같이 판정해 놓친 알레르겐을 센다.  담당: 권용현

evaluate.py 는 OCR 글자만 비교한다. 이 스크립트는 바코드 · 품질 검사 · 영역 찾기 · OCR ·
구간 나누기 · 판정까지 웹 서버(`/api/ingredient`, 알레르기 '모두')와 같은 길로 돌려서
사용자가 실제로 듣는 결과를 잰다.

평가 세트 (Git 에 올리지 않음)
    data/datasets/ocr_eval/      개발 세트 21장. 규칙을 고칠 때마다 봐서 이미 맞춰져 있다.
    data/datasets/ocr_holdout/   홀드아웃 세트. 규칙을 고칠 때는 보지 않고, PR 마다 한 번만 잰다.

    사진.jpg + 사진.txt. txt 에서 쓰는 줄:
        # allergens: 우유,대두,밀        들었거나 혼입 가능한 알레르겐 (계란·달걀은 알류로 센다)
        # 실패유형: 작음·흐림             찍을 때 일부러 넣은 어려움. 없으면 '보통'
    실패유형은 멘토 분류 네 가지 중 하나: 작음·흐림 / 반사·굴곡 / 줄바꿈·띄어쓰기 / 함유·혼입

홀드아웃 찍는 법 (docs/experiments.md '홀드아웃 세트' 참고)
    - 개발 세트에 없는 제품만. 같은 제품을 다른 각도로 찍은 것도 개발 세트에 있으면 뺀다.
    - 실패유형마다 5장 이상 + 보통 5장 이상. 휴대폰 웹(대시보드)과 웹캠으로 섞어 찍는다.
    - 정답은 찍은 사람이 아니라 다른 팀원이 성분표를 보고 적는다.

실행:
    $env:PYTHONPATH="src"
    python scripts/ocr_eval/pipeline.py                                  # 개발 세트
    python scripts/ocr_eval/pipeline.py --data data/datasets/ocr_holdout
    python scripts/ocr_eval/pipeline.py --out data/look/holdout.json      # 사진별 결과 저장

출력: 사진별 놓친 알레르겐 · 다시 찍기 · 시간, 실패유형별 놓침 표, 합계.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2

from jarviseo.api.main import evidence
from jarviseo.nutrition.allergen import KNOWN_ALLERGENS
from jarviseo.nutrition.service import make_checker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "datasets" / "ocr_eval"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TYPES = ["작음·흐림", "반사·굴곡", "줄바꿈·띄어쓰기", "함유·혼입", "보통"]


def read_answer(txt: Path) -> tuple[set[str], str]:
    """정답 txt 에서 (알레르겐, 실패유형)."""
    gt: set[str] = set()
    kind = "보통"
    for line in txt.read_text(encoding="utf-8").splitlines():
        if line.startswith("# allergens:"):
            gt = {a.strip() for a in line.split(":", 1)[1].split(",") if a.strip()}
            gt = {"알류" if a in ("계란", "달걀") else a for a in gt}
        elif line.startswith("# 실패유형:"):
            kind = line.split(":", 1)[1].strip() or kind
    return gt, kind


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, help="사진별 결과를 JSON 으로 저장")
    args = ap.parse_args()

    allergens = sorted(KNOWN_ALLERGENS)
    checker = make_checker()
    imgs = sorted(p for p in args.data.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rows = []
    for img in imgs:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            print(f"{img.name}: 정답 txt 없음, 건너뜀")
            continue
        gt, kind = read_answer(txt)
        t = time.perf_counter()
        r = checker.check(cv2.imread(str(img)), allergens)
        ms = round((time.perf_counter() - t) * 1000)
        ev = evidence(r, allergens, checker.synonyms)
        found = {e["allergen"] for e in ev if e["status"] in ("contains", "may")}
        row = {"img": img.name, "kind": kind, "gt": len(gt), "miss": sorted(gt - found),
               "extra": sorted(found - gt), "retake": r.retake or "", "ms": ms}  # fmt: skip
        rows.append(row)
        note = f" (다시 찍기: {row['retake']})" if row["retake"] else ""
        print(f"{img.name:28s} {kind:8s} 놓침 {row['miss']}{note} {ms}ms", flush=True)

    if not rows:
        print("평가할 사진이 없습니다")
        return
    by = defaultdict(list)
    for x in rows:
        by[x["kind"]].append(x)
    print("\n| 실패유형 | 사진 | 놓침 / 정답 | 모두 찾은 사진 | 다시 찍기 |")
    print("|---|---|---|---|---|")
    for k in [*TYPES, *sorted(set(by) - set(TYPES))]:
        xs = by.get(k)
        if not xs:
            continue
        miss = sum(len(x["miss"]) for x in xs)
        gt = sum(x["gt"] for x in xs)
        ok = sum(not x["miss"] for x in xs)
        rt = sum(bool(x["retake"]) for x in xs)
        print(f"| {k} | {len(xs)} | {miss} / {gt} | {ok} / {len(xs)} | {rt} |")
    miss = sum(len(x["miss"]) for x in rows)
    gt = sum(x["gt"] for x in rows)
    avg = sum(x["ms"] for x in rows) / len(rows) / 1000
    rt = sum(bool(x["retake"]) for x in rows)
    print(f"\n합계: 놓침 {miss} / {gt} · 다시 찍기 {rt}장 · 평균 {avg:.1f}초")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
