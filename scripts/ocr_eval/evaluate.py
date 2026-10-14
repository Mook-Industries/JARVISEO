"""성분표 OCR 평가 — 엔진·전처리 비교.  담당: 권용현

평가 세트(성분표 사진 + 사람이 적은 정답 글자)로 OCR 엔진을 같은 조건에서 비교한다.
학습(파인튜닝)을 할지 정하기 전에, 학습 없이 되는 것(전처리·엔진 교체)부터 숫자로 본다.

평가 세트: data/datasets/ocr_eval/ (Drive, Git 에 올리지 않음)
    사진.jpg  +  사진.txt (정답)

    정답 .txt 형식 — '#' 으로 시작하는 줄은 설정, 나머지가 정답 글자다.
        # bbox: 325,600,550,1050        성분표 영역 x1,y1,x2,y2 (없으면 사진 전체)
        # allergens: 우유,메밀,땅콩      들었거나 혼입 가능한 알레르겐 (없으면 정답 글자에서 찾음)
        원재료명 정제수, 오렌지농축액A(스페인산), ...
        우유,메밀,땅콩,대두,밀,복숭아,토마토,아황산류,호두,잣 혼입가능.
    정답은 bbox 안에 보이는 글자를 위→아래로 그대로 적는다. 줄바꿈·띄어쓰기는 비교할 때 지운다.
    --init 으로 만든 초안에는 '# draft' 줄이 있다. 눈으로 고친 뒤 그 줄을 지워야 평가에 들어간다.

엔진:
    easyocr       지금 파이프라인 그대로 (EasyOCR 한국어+영어)
    easyocr-pre   흑백 → 대비 보정(CLAHE) → 샤프닝 후 EasyOCR
    paddle        PaddleOCR 한국어 모델. torch 와 섞이지 않게 venv 를 따로 만들어 돌린다
                    py -3.12 -m venv .venv-paddle
                    .venv-paddle\Scripts\pip install paddlepaddle paddleocr
                    $env:PYTHONPATH="src"
                    .venv-paddle\Scripts\python scripts\ocr_eval\evaluate.py --engines paddle

지표 (사진마다 내고 엔진별로 모은다):
    CER          문자 오류율 = 편집 거리 ÷ 정답 글자 수. 낮을수록 좋다. 공백은 빼고 센다
    알레르겐 재현율  정답 글자에서 찾은 알레르겐 중 OCR 글자에서도 찾은 비율 (놓치면 사고다)
    알레르겐 정밀도  OCR 글자에서 찾은 알레르겐 중 정답에도 있는 비율 (틀리면 과잉 경고)
    머리말 / 혼입 문장   원재료명 머리말·같은 제조시설(혼입) 문장을 정답과 똑같이 찾았나
    신뢰도 · 판독ms   평균 신뢰도와 판독 시간 (모델 로드 시간은 뺀다)
알레르겐은 판정 코드와 같은 allergen.find_allergens 와 동의어 사전으로 찾는다.

실행:
    python scripts/ocr_eval/evaluate.py --init       # 정답 없는 사진에 초안 .txt 만들기
    python scripts/ocr_eval/evaluate.py              # easyocr, easyocr-pre 비교
    python scripts/ocr_eval/evaluate.py --engines easyocr,easyocr-pre,paddle --scale 3

결과: data/datasets/ocr_eval/results/ 에
    ocr-<날짜시각>-images.csv    사진 × 엔진별 지표와 OCR 글자
    ocr-<날짜시각>-summary.csv   엔진별 평균
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from jarviseo.nutrition.allergen import find_allergens, load_synonyms, normalize_term
from jarviseo.nutrition.ocr import (
    PanelSections,
    _normalize,  # OCR 글자와 정답에 같은 정규화를 쓰려고 가져온다
    crop_and_upscale,
    join_lines,
    mean_confidence,
    read_panel,
    split_sections,
)
from jarviseo.types import BBox, OCRLine

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDER = ROOT / "data" / "datasets" / "ocr_eval"
SYNONYMS = ROOT / "data" / "datasets" / "allergen_synonyms.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DRAFT_MARK = "# draft"


# --------------------------------------------------------------------------
# 평가 세트 읽기
# --------------------------------------------------------------------------


def imread(path: Path) -> np.ndarray:
    # cv2.imread 는 윈도우에서 한글 경로를 못 연다. 바이트로 읽어서 디코드한다.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 열 수 없습니다: {path}")
    return image


@dataclass
class Label:
    text: str  # 정답 글자
    bbox: BBox | None = None  # 성분표 영역. 없으면 사진 전체
    allergens: set[str] | None = (
        None  # '# allergens:' 로 적은 정답 알레르겐. 없으면 글자에서 찾는다
    )
    draft: bool = False  # --init 초안을 아직 안 고쳤나


def read_label(path: Path) -> Label:
    """정답 .txt 를 읽는다. '#' 줄은 설정(draft·bbox·allergens), 나머지는 정답 글자."""
    label, body = Label(text=""), []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        key, _, value = s.partition(":")
        key, value = key.lower().replace(" ", ""), value.strip()
        if s.lower().startswith(DRAFT_MARK):
            label.draft = True
        elif key == "#bbox" and value:
            label.bbox = BBox(*map(float, value.split(",")))
        elif key == "#allergens" and value:
            label.allergens = {normalize_term(a) for a in value.split(",") if a.strip()}
        elif not s.startswith("#"):
            body.append(line)
    label.text = "\n".join(body).strip()
    return label


def collect(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


# --------------------------------------------------------------------------
# 엔진
# --------------------------------------------------------------------------


def preprocess(image: np.ndarray) -> np.ndarray:
    """흑백 → 대비 보정(CLAHE) → 언샤프 마스크. EasyOCR 이 BGR 을 받으므로 3채널로 돌려준다.

    CLAHE 는 반사나 그늘로 한쪽만 어두운 성분표에서 글자와 바탕 차이를 부분마다 키운다.
    확대(INTER_CUBIC)로 무뎌진 글자 경계는 언샤프 마스크로 다시 세운다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


class PaddleReader:
    """PaddleOCR 를 easyocr.Reader.readtext 모양(OcrReader)으로 감싼다. read_panel 에 넣는다."""

    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        try:  # 3.x
            # enable_mkldnn=False: paddle 3.3 윈도우 CPU 에서 oneDNN 을 켜면 추론 때
            # "ConvertPirAttribute2RuntimeAttribute not support" 로 멈춘다.
            self._ocr = PaddleOCR(
                lang="korean",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
            self._v3 = True
        except (TypeError, ValueError):  # 2.x 는 위 인자를 모른다
            self._ocr = PaddleOCR(lang="korean", use_angle_cls=False, show_log=False)
            self._v3 = False

    def readtext(self, image: np.ndarray, **_) -> list:
        out = []
        if self._v3:
            for res in self._ocr.predict(image):
                r = dict(res)
                out += zip(r["rec_polys"], r["rec_texts"], r["rec_scores"], strict=False)
        else:
            for page in self._ocr.ocr(image, cls=False) or []:
                for pts, (text, score) in page or []:
                    out.append((pts, text, score))
        return [(np.asarray(p).tolist(), t, float(c)) for p, t, c in out]


Engine = Callable[[np.ndarray], list[OCRLine]]


def make_engine(name: str) -> Engine:
    if name == "easyocr":
        return lambda panel: read_panel(panel)
    if name == "easyocr-pre":
        return lambda panel: read_panel(preprocess(panel))
    if name == "paddle":
        try:
            reader = PaddleReader()
        except ImportError as e:
            raise SystemExit("paddle 을 쓰려면 먼저: pip install paddlepaddle paddleocr") from e
        return lambda panel: read_panel(panel, reader=reader)
    raise SystemExit(f"모르는 엔진: {name} (easyocr, easyocr-pre, paddle)")


# --------------------------------------------------------------------------
# 지표
# --------------------------------------------------------------------------


def compact(text: str) -> str:
    """비교용 글자. 괄호·쉼표 모양을 맞추고(_normalize) 공백을 모두 지운다(normalize_term)."""
    return normalize_term(_normalize(text))


def edit_distance(a: str, b: str) -> int:
    """레벤슈타인 거리. 성분표 한 장이 수백 자라 두 줄 DP 로 충분하다."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, truth: str) -> float:
    p, t = compact(pred), compact(truth)
    return edit_distance(p, t) / len(t) if t else float(bool(p))


def sections_text(sec: PanelSections) -> str:
    return " ".join((sec.ingredients, sec.allergen_notice, sec.cross_contamination))


def score(pred_text: str, label: Label, synonyms: dict[str, list[str]]) -> dict:
    """OCR 글자 한 장을 정답과 비교한다. 엔진별로 모으기 쉽게 개수(분자·분모)로 돌려준다.

    알레르겐은 판정이 읽는 세 구간(원재료·알레르기 표시·혼입 문장)에서 찾는다. 판정과 같은 조건이다.
    정답 알레르겐은 '# allergens:' 가 있으면 그것을 쓴다. 구간 분리 규칙이 보관 방법 문구까지
    원재료로 삼키면("얼지 않게"의 '게') 정답 글자에서 찾은 목록이 틀리기 때문이다.
    이때 OCR 쪽에서 '게'가 나오면 오탐(extra)으로 잡힌다. 실제 판정에서도 과잉 경고가 되는 경우다.
    """
    gt_sec, pr_sec = split_sections(_normalize(label.text)), split_sections(pred_text)
    # '# allergens:' 에 '달걀'처럼 다른 표기로 적어도 표준명('알류')으로 맞춰 비교한다
    canonical = {t: a for a, ts in synonyms.items() for t in ts}
    gt_all = {canonical.get(a, a) for a in label.allergens or ()} or set(
        find_allergens(sections_text(gt_sec), synonyms)
    )
    pr_all = set(find_allergens(sections_text(pr_sec), synonyms))
    # OCR 이 정답 글자를 그대로 읽었을 때 찾는 알레르겐. 이보다 못 찾은 만큼이 OCR 탓이고,
    # 여기서도 못 찾는 것은 구간 분리·사전(파서) 탓이다.
    ceil_all = set(find_allergens(sections_text(gt_sec), synonyms))
    return {
        "cer": cer(pred_text, label.text),
        "gt_allergens": len(gt_all),
        "ceil_allergens": len(gt_all & ceil_all),
        "hit_allergens": len(gt_all & pr_all),
        "pred_allergens": len(pr_all),
        "missed": ",".join(sorted(gt_all - pr_all)),
        "extra": ",".join(sorted(pr_all - gt_all)),
        "header_ok": gt_sec.has_ingredient_header == pr_sec.has_ingredient_header,
        "gt_cross": bool(gt_sec.cross_contamination),
        "cross_ok": bool(gt_sec.cross_contamination) == bool(pr_sec.cross_contamination),
    }


def summarize(rows: list[dict]) -> dict:
    gt = sum(r["gt_allergens"] for r in rows)
    pred = sum(r["pred_allergens"] for r in rows)
    hit = sum(r["hit_allergens"] for r in rows)
    ceil = sum(r["ceil_allergens"] for r in rows)
    return {
        "images": len(rows),
        "cer_mean": statistics.mean(r["cer"] for r in rows),
        "allergen_recall": hit / gt if gt else float("nan"),
        "allergen_recall_ceiling": ceil / gt if gt else float("nan"),
        "allergen_precision": hit / pred if pred else float("nan"),
        "header_ok": sum(r["header_ok"] for r in rows) / len(rows),
        "cross_ok": sum(r["cross_ok"] for r in rows) / len(rows),
        "conf_mean": statistics.mean(r["conf"] for r in rows),
        "ms_median": statistics.median(r["ms"] for r in rows),
    }


# --------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------


def init_labels(images: list[Path], scale: float) -> None:
    """정답이 없는 사진마다 EasyOCR 결과로 초안 .txt 를 만든다. 정답을 처음부터 치는 수고를 던다.

    초안은 OCR 결과라 틀린 글자투성이다. 사진을 보며 고치고 '# draft' 줄을 지운다.
    고치지 않은 초안을 그대로 쓰면 OCR 을 OCR 로 채점하게 되므로 평가에서 뺀다.
    """
    todo = [p for p in images if not p.with_suffix(".txt").exists()]
    if not todo:
        print("정답 없는 사진이 없습니다.")
        return
    engine = make_engine("easyocr")
    for path in todo:
        image = imread(path)
        h, w = image.shape[:2]
        text = "\n".join(ln.text for ln in engine(crop_and_upscale(image, BBox(0, 0, w, h), scale)))
        header = f"{DRAFT_MARK} — 사진을 보고 아래 글자를 고친 뒤 이 줄을 지우세요"
        path.with_suffix(".txt").write_text(
            f"{header}\n# bbox: \n# allergens: \n{text}\n", encoding="utf-8"
        )
        print(f"초안: {path.with_suffix('.txt').name}")
    print("\n# bbox 에 성분표 영역(x1,y1,x2,y2)을 적으면 그 부분만 평가합니다. 비우면 사진 전체.")


def fmt(v: float, pct: bool = False) -> str:
    if v != v:  # nan
        return "-"
    return f"{v * 100:.0f}%" if pct else f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="성분표 OCR 엔진·전처리 비교")
    ap.add_argument("--folder", type=Path, default=DEFAULT_FOLDER, help="평가 세트 폴더")
    ap.add_argument("--engines", default="easyocr,easyocr-pre", help="쉼표로 구분")
    ap.add_argument("--scale", type=float, default=3.0, help="확대 배율 (긴 변 2560px 상한)")
    ap.add_argument("--init", action="store_true", help="정답 없는 사진에 초안 .txt 만들기")
    args = ap.parse_args()

    images = collect(args.folder) if args.folder.is_dir() else []
    if not images:
        raise SystemExit(f"사진이 없습니다: {args.folder}")
    if args.init:
        init_labels(images, args.scale)
        return

    synonyms = load_synonyms(SYNONYMS)
    cases: list[tuple[Path, Label]] = []
    for path in images:
        label_path = path.with_suffix(".txt")
        if not label_path.exists():
            print(f"건너뜀 (정답 없음): {path.name}")
            continue
        label = read_label(label_path)
        if label.draft or not label.text:
            print(f"건너뜀 ({'초안' if label.draft else '정답 비어 있음'}): {label_path.name}")
            continue
        cases.append((path, label))
    if not cases:
        raise SystemExit("평가할 사진이 없습니다. --init 으로 초안을 만들고 고쳐 주세요.")

    names = [n.strip() for n in args.engines.split(",") if n.strip()]
    rows: list[dict] = []
    for name in names:
        t0 = time.perf_counter()
        engine = make_engine(name)
        engine(np.full((32, 32, 3), 255, np.uint8))  # 모델 로드를 판독 시간에서 빼려고 먼저 한 번
        print(f"\n=== {name}  (모델 로드 {time.perf_counter() - t0:.1f}s)")
        for path, label in cases:
            image = imread(path)
            h, w = image.shape[:2]
            panel = crop_and_upscale(image, label.bbox or BBox(0, 0, w, h), scale=args.scale)
            t1 = time.perf_counter()
            lines = engine(panel)
            ms = (time.perf_counter() - t1) * 1000
            text = join_lines(lines)
            row = {"engine": name, "image": path.name, "conf": mean_confidence(lines), "ms": ms,
                   **score(text, label, synonyms), "text": text}  # fmt: skip
            rows.append(row)
            print(f"  {path.name[:30]:<30} CER {row['cer']:.3f} · 알레르겐 "
                  f"{row['hit_allergens']}/{row['gt_allergens']}"
                  f"{' 놓침:' + row['missed'] if row['missed'] else ''}"
                  f"{' 오탐:' + row['extra'] if row['extra'] else ''}"
                  f" · 신뢰도 {row['conf']:.2f} · {ms:.0f}ms")  # fmt: skip

    summary = [{"engine": n, **summarize([r for r in rows if r["engine"] == n])} for n in names]
    ceiling = fmt(summary[0]["allergen_recall_ceiling"], True)
    print(f"\n=== 요약 (사진 {len(cases)}장, 배율 {args.scale})")
    print(f"알레르겐 재현율 상한 {ceiling} — OCR 이 정답 글자를 그대로 읽어도 파서가 찾는 비율")
    print(f"{'엔진':<12} {'CER':>6} {'알레르겐재현':>8} {'정밀':>6} {'머리말':>6} {'혼입문장':>6} "
          f"{'신뢰도':>6} {'판독ms':>7}")  # fmt: skip
    for s in summary:
        recall, prec = fmt(s["allergen_recall"], True), fmt(s["allergen_precision"], True)
        header, cross = fmt(s["header_ok"], True), fmt(s["cross_ok"], True)
        print(f"{s['engine']:<12} {fmt(s['cer_mean']):>6} {recall:>10} {prec:>7} {header:>7} "
              f"{cross:>8} {fmt(s['conf_mean']):>8} {s['ms_median']:>8.0f}")  # fmt: skip

    out = args.folder / "results"
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    for suffix, data in (("images", rows), ("summary", summary)):
        path = out / f"ocr-{stamp}-{suffix}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as f:  # 엑셀에서 한글이 안 깨지게 BOM
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        print(f"저장: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
