"""사진 한 장으로 성분 판정을 돌려 보는 스크립트.  담당: 권용현

사진을 넣으면 앱이 읽어 줄 문장 하나가 나온다.
    해당 제품에는 밀, 대두가 들어 있고, 새우, 땅콩은 섞였을 수 있어요.
안에서는 다른 모듈이 부르는 입구(IngredientChecker.answer)를 그대로 부른다.
사진 → 바코드 → 푸드QR → (못 찾으면) 성분표 OCR → 알레르기 판정 → 문장.

--detail 을 주면 바코드·판정 출처·단계별 시간까지 찍는다(어디가 느린지 볼 때).
OCR 은 .env 에 JARVISEO_OCR_REMOTE_URL 이 있으면 Colab GPU, 없으면 이 PC CPU 로 돈다.
캐시(DB)는 쓰지 않는다. 매번 실제 조회 경로를 탄다.

사용:
    python scripts/check_ingredient.py 사진.jpg --allergens 계란 땅콩 새우
    python scripts/check_ingredient.py data/samples/ --allergens 우유          # 폴더 안 사진 전부
    python scripts/check_ingredient.py 사진.jpg --allergens 계란 --detail     # 단계별 시간까지
    python scripts/check_ingredient.py 사진.jpg --allergens 계란 --region     # 성분표 영역만 읽기
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from jarviseo import config
from jarviseo.nutrition import ocr
from jarviseo.nutrition.service import make_checker

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STEP_NAMES = {"barcode": "바코드·조회", "quality": "품질 검사", "ocr": "OCR", "match": "판정"}


def imread(path: Path) -> np.ndarray:
    # cv2.imread 는 윈도우에서 한글 경로를 못 연다. 바이트로 읽어서 디코딩한다.
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


def source_of(result) -> str:
    """어디서 판정했는지. 바코드 조회로 끝났으면 그 출처, 아니면 OCR."""
    lookup = result.barcode
    if result.retake:
        return "재촬영 안내"
    if lookup is not None and lookup.hit and not result.ocr_text:
        return f"바코드 조회 ({lookup.product.source})"
    return "성분표 OCR"


def report(path: Path, result, elapsed_ms: int) -> None:
    lookup = result.barcode
    print(f"\n=== {path.name}")
    if lookup is not None and lookup.barcode is not None:
        print(f"바코드   : {lookup.barcode.code}")
    else:
        reason = lookup.reason if lookup is not None else ""
        print(f"바코드   : 못 읽음 {f'({reason})' if reason else ''}")
    if lookup is not None and lookup.product is not None and lookup.product.product_name:
        print(f"제품     : {lookup.product.product_name}")
    print(f"판정 출처: {source_of(result)}")
    print(f"판정     : {result.judgement.verdict.value}")
    if result.ocr_text:
        conf = f"{result.ocr_conf:.2f}" if result.ocr_conf is not None else "-"
        rot = f", {result.rotation}도 돌려 읽음" if result.rotation else ""
        print(f"OCR      : 신뢰도 {conf}, 재처리 {'함' if result.reprocessed else '안 함'}{rot}")
        if result.panel_region is not None:
            b = result.panel_region
            print(f"성분표   : ({int(b.x1)},{int(b.y1)})-({int(b.x2)},{int(b.y2)}) 만 읽음")
    print(f"문장     : {result.retake or result.message}")
    steps = ", ".join(f"{STEP_NAMES.get(k, k)} {v}ms" for k, v in result.timings_ms.items())
    print(f"시간     : 전체 {elapsed_ms}ms ({steps})")


def main() -> None:
    ap = argparse.ArgumentParser(description="사진 한 장으로 성분 판정 확인")
    ap.add_argument("paths", nargs="+", help="사진 파일 또는 폴더")
    ap.add_argument("--allergens", nargs="+", required=True, help="내 알레르기 (예: 계란 땅콩)")
    ap.add_argument(
        "--region",
        action=argparse.BooleanOptionalAction,
        default=config.OCR_AUTO_REGION,
        help="성분표 영역을 먼저 찾아 그 부분만 읽는다 (기본: .env 의 JARVISEO_OCR_AUTO_REGION)",
    )
    ap.add_argument("--detail", action="store_true", help="바코드·판정 출처·단계별 시간까지 찍는다")
    args = ap.parse_args()

    reader = None
    if args.detail:
        print(f"푸드QR   : {'사용' if config.FOOD_QR_API_KEY else '인증키 없음 → 건너뜀'}")
        print(f"OCR      : {'Colab GPU' if config.OCR_REMOTE_URL else '이 PC CPU'}")
        print(f"알레르기 : {', '.join(args.allergens)}")
        print(f"영역찾기 : {'켬' if args.region else '끔'}")
        # OCR 모델 준비는 앱 시작 때 한 번만 한다. 사진마다 시간에 섞이지 않게 따로 잰다.
        t0 = time.perf_counter()
        reader = ocr._get_reader()
        reader.readtext(np.full((64, 256, 3), 255, dtype=np.uint8))
        print(f"OCR 준비 : {int((time.perf_counter() - t0) * 1000)}ms (앱 시작 때 한 번)")

    checker = make_checker(store=None, reader=reader, auto_region=args.region)
    files = collect(args.paths)
    try:
        for path in files:
            image = imread(path)
            if args.detail:
                t0 = time.perf_counter()
                result = checker.check(image, args.allergens)
                report(path, result, int((time.perf_counter() - t0) * 1000))
                continue
            # 다른 모듈이 부르는 입구 그대로: 크롭 하나 + 성분 분석이 필요하다(True).
            text = checker.answer(image, True, args.allergens)
            print(f"{path.name}: {text}" if len(files) > 1 else text)
    finally:
        checker.close()


if __name__ == "__main__":
    main()
