"""실제 성분표 사진으로 OCR 을 돌려 본다.  담당: 권용현

사진(또는 폴더)을 넣으면 줄별 판독 결과·신뢰도·걸린 시간과
원재료명 / 알레르기 표시 / 같은 제조시설 구간 분리 결과를 출력한다.
여러 장이면 마지막에 요약표를 낸다. OCR 신뢰도 기준값(OCR_CONF_THRESHOLD)을
정하거나 PR·experiments.md 에 실측치를 적을 때 쓴다.

처음 실행하면 EasyOCR 모델(약 100MB)을 내려받는다.

실행:
    python scripts/check_ingredient_ocr.py 사진.jpg
    python scripts/check_ingredient_ocr.py data/samples/          # 폴더 안 사진 전부
    python scripts/check_ingredient_ocr.py 사진.jpg --bbox 100,200,900,700 --scale 3
    python scripts/check_ingredient_ocr.py 사진.jpg --save out/     # 확대한 성분표 이미지 저장
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from jarviseo import config
from jarviseo.nutrition.ocr import (
    crop_and_upscale,
    join_lines,
    mean_confidence,
    parse_ingredient_text,
    read_panel,
    split_sections,
    split_terms,
)
from jarviseo.types import BBox

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imread(path: Path) -> np.ndarray:
    # cv2.imread 는 윈도우에서 한글 경로를 못 연다. 바이트로 읽어서 디코드한다.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 열 수 없습니다: {path}")
    return image


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        else:
            files.append(p)
    return files


def check(path: Path, bbox: BBox | None, scale: float, save: Path | None) -> dict:
    image = imread(path)
    h, w = image.shape[:2]
    panel = crop_and_upscale(image, bbox or BBox(0, 0, w, h), scale=scale)
    if save:
        save.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", panel)[1].tofile(str(save / f"{path.stem}_panel.png"))

    t0 = time.perf_counter()
    lines = read_panel(panel)
    read_ms = (time.perf_counter() - t0) * 1000
    conf = mean_confidence(lines)
    sec = split_sections(join_lines(lines))
    ingredients = parse_ingredient_text(sec.ingredients)
    notice = split_terms(sec.allergen_notice)
    ok = conf >= config.OCR_CONF_THRESHOLD

    print(f"\n=== {path.name}  원본 {w}x{h} → 성분표 {panel.shape[1]}x{panel.shape[0]}")
    print(f"판독 {read_ms:.0f}ms · 평균 신뢰도 {conf:.2f} "
          f"({'통과' if ok else '실패'}, 기준 {config.OCR_CONF_THRESHOLD})")  # fmt: skip
    for ln in lines:
        print(f"  {ln.confidence:.2f} | {ln.text}")
    print(f"원재료명 머리말: {'찾음' if sec.has_ingredient_header else '못 찾음'}")
    print(f"원재료 ({len(ingredients)}개): {ingredients}")
    print(f"알레르기 표시: {notice}")
    print(f"같은 제조시설: {sec.cross_contamination or '-'}")
    return {"name": path.name, "conf": conf, "ms": read_ms, "header": sec.has_ingredient_header,
            "n": len(ingredients), "ok": ok}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description="성분표 사진 OCR 확인")
    ap.add_argument("paths", nargs="+", help="사진 파일 또는 폴더")
    ap.add_argument("--bbox", help="성분표 영역 x1,y1,x2,y2 (없으면 사진 전체)")
    ap.add_argument("--scale", type=float, default=2.0, help="확대 배율 (기본 2)")
    ap.add_argument("--save", type=Path, help="확대한 성분표 이미지를 저장할 폴더")
    args = ap.parse_args()

    bbox = BBox(*map(float, args.bbox.split(","))) if args.bbox else None
    files = collect(args.paths)
    if not files:
        raise SystemExit("사진을 찾지 못했습니다.")

    t0 = time.perf_counter()
    read_panel(np.full((32, 32, 3), 255, np.uint8))  # 모델 로드를 판독 시간에서 빼려고 먼저 한 번
    print(f"EasyOCR 모델 로드 {time.perf_counter() - t0:.1f}s")

    rows = [check(f, bbox, args.scale, args.save) for f in files]
    if len(rows) > 1:
        print("\n=== 요약")
        print(f"{'파일':<28} {'신뢰도':>6} {'판독ms':>7} {'머리말':>6} {'성분수':>6}")
        for r in rows:
            print(f"{r['name'][:28]:<28} {r['conf']:>6.2f} {r['ms']:>7.0f} "
                  f"{'O' if r['header'] else 'X':>6} {r['n']:>6}")  # fmt: skip
        confs = sorted(r["conf"] for r in rows)
        mid = confs[len(confs) // 2]
        passed = sum(r["ok"] for r in rows)
        print(f"신뢰도 최소 {confs[0]:.2f} · 중앙 {mid:.2f} · 최대 {confs[-1]:.2f}"
              f" · 기준 통과 {passed}/{len(rows)}")  # fmt: skip


if __name__ == "__main__":
    main()
