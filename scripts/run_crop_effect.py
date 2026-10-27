"""크롭 효과 실험 — VLM 에 무엇을 보내야 가리킨 물건을 맞게 말하나.  담당: 최홍묵

"사진을 통째로 GPT 에 주면 되지 않나?"에 숫자로 답한다(#234). 가리킴 질문 사진마다 같은 질문을
VLM 에 네 가지 사진으로 보내고, 답이 가리킨 물건을 맞게 말했는지 채점한다.

    full  전체 사진 + 질문 그대로. 그냥 통째로 주는 경우
    ours  우리 파이프라인이 고른 크롭. 실제 비서의 resolve · crop_target 노드와 같게 자르고
          같은 맥락 문장을 붙인다. 되물으면 실제 비서처럼 VLM 을 부르지 않는다
    gt    정답 박스 크롭. 대상을 완벽히 골랐을 때의 천장이다.
          gt 와 ours 의 차이가 대상 고르기 몫이다
    mark  전체 사진에 ours 가 고른 곳을 초록 사각형으로 그린다. 고른 곳이 ours 와 같아서
          "자르기 vs 표시하기"만 다르다

대상은 글자 라우터가 가리킴으로 보내는 질문뿐이다. 성분 질문은 VLM 을 안 부르고, 장면 · 찾기
질문은 원래 전체 사진을 보내서 크롭과 상관이 없다. test 는 최종 1회까지 봉인이라 valid 만 연다.

채점은 같은 VLM 모델에게 정답 크롭 · 정답 이름 · 답을 주고 맞음 / 틀림 / 모름 으로 받는다.
채점기도 틀릴 수 있어서 --sample 로 뽑은 표본을 사람이 채점하고 --agreement 로 일치율을 잰다.
결과는 run_ablation 과 같은 방식으로 eval_run(조건마다 한 줄) · eval_sample(답마다) · 답별 CSV 에
남긴다. 무엇으로 쟀는지(커밋 · 가중치 지문 · 모델 · 반복)는 eval_run.metrics["conditions"] 에 있다.

실행 (레포 맨 위 폴더에서):
    python scripts/run_crop_effect.py --dry-run --out /tmp/crop_effect     # VLM 없이 입력 사진만
    python scripts/run_crop_effect.py --limit 3 --repeats 1 --db sqlite:////tmp/x.db --out /tmp/x
    python scripts/run_crop_effect.py --weights data/models/detector_r2.pt --sample 30 \\
        --note "크롭 효과"
    python scripts/run_crop_effect.py --agreement data/datasets/pointing/runs/40_judge_check.csv
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI, OpenAIError
from run_ablation import (
    POINTING_DIR,
    git_commit,
    load_samples,
    missing_weights,
    read_image,
    save_run,
    weights_fingerprint,
)

from jarviseo import config
from jarviseo.memory import MemoryStore
from jarviseo.pointing.crop import crop_target as cut_target
from jarviseo.pointing.crop import fit_for_vlm, hand_crop_region
from jarviseo.pointing.hands import HandDetector, HandPose, pick_pointing_hand
from jarviseo.pointing.resolver import resolve_target
from jarviseo.types import BBox, Detection, Frame, Intent
from jarviseo.vlm.client import VLMClient, encode_image
from jarviseo.workflow.nodes.answer import HAND_CONTEXT, POINTING_CONTEXT
from jarviseo.workflow.router import classify_intent

# valid 62장 = 세션 1 · 2, 말 · 맥락 세션 = 4. 결과는 둘을 따로 본다(set_name).
DEFAULT_SESSIONS = (1, 2, 4)
CONDITIONS = ("full", "ours", "gt", "mark")
# 같은 사진도 VLM 답이 매번 달라서(#229 턴 22: 같은 문장 6번 중 3번만 같은 말) 한 번은 운이다.
DEFAULT_REPEATS = 2
# 한 번에 보내는 호출 수. 하나씩 보내면 VLM 과 채점이 각각 수백 번이라 30분 가까이 걸린다.
WORKERS = 4
MARK_COLOR = (0, 255, 0)  # BGR 초록
MARK_CONTEXT = (
    "사진에 초록색 사각형으로 표시한 것이 사용자가 가리킨 물건이다. "
    "사각형 안의 물건을 보고 답한다.\n"
    "질문: {question}"
)
# 여러 물건을 늘어놓기만 한 답은 '모름'이다. 전체 사진에 "이거 뭐야?"를 물으면 VLM 이 보이는 걸
# 다 나열하곤 하는데, 엉뚱한 물건을 단정한 것이 아니라 못 고른 것이라 '틀림'으로 세면 안 된다.
JUDGE_INSTRUCTIONS = """\
너는 음성 비서의 대답을 채점한다.
사진은 사용자가 가리킨 물건을 잘라낸 것이고, 이것이 정답이다. 참고 이름은 영어로 준다.
비서의 대답이 사진 속 그 물건을 말했으면 '맞음', 다른 물건을 말했으면 '틀림',
무엇인지 모른다고 했거나 여러 물건을 늘어놓고 어느 것인지 정하지 않았으면 '모름'이라고 답한다.
이름이 달라도 같은 물건이면 맞음이다(예: 텀블러와 물병).
그 물건을 맞게 말했으면 상표 · 내용물 같은 세부를 모른다고 덧붙여도 맞음이다.
맞음, 틀림, 모름 가운데 한 낱말만 답한다."""
JUDGE_MAX_TOKENS = 20
VERDICTS = ("맞음", "틀림", "모름")
# 판정 밖의 결과. 안 부름 = ours · mark 에서 되물어 VLM 을 안 부름(실제 비서와 같음).
NOT_CALLED, CALL_FAILED, JUDGE_FAILED = "안 부름", "호출 실패", "채점 실패"
OUTCOMES = (*VERDICTS, NOT_CALLED, CALL_FAILED, JUDGE_FAILED)
CSV_COLUMNS = [
    "file",
    "set",
    "repeat",
    "condition",
    "ours",
    "chosen_label",
    "gt_label",
    "utterance",
    "question",
    "answer",
    "verdict",
]
CHECK_COLUMNS = ["file", "condition", "repeat", "utterance", "answer", "judge", "human"]


def pointing_samples(samples: list[dict]) -> list[dict]:
    """글자 라우터가 가리킴으로 보내는 질문의 사진만. 크롭이 VLM 으로 가는 길은 이것뿐이다."""
    return [s for s in samples if classify_intent(s["utterance"]) is Intent.POINTING]


def set_name(file: str) -> str:
    """결과를 나눠 볼 묶음. s04 는 질문에 색 · 위치 말이 들어 있어 valid 62장과 따로 본다."""
    return "s04" if file.startswith("s04/") else "valid"


def draw_mark(image: np.ndarray, box: BBox) -> np.ndarray:
    """사본에 사각형을 그린다(원본은 다른 조건이 쓴다). 두께는 사진 높이에 맞춰 1080 에서 5px."""
    marked = image.copy()
    thickness = max(3, image.shape[0] // 200)
    corners = (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))
    cv2.rectangle(marked, *corners, MARK_COLOR, thickness)
    return marked


def build_inputs(
    sample: dict, image: np.ndarray, candidates: list[Detection], hand: HandPose | None
) -> tuple[dict[str, tuple[np.ndarray, str] | None], dict]:
    """조건마다 (VLM 에 보낼 사진, 질문 글자). 그 조건에서 VLM 을 안 부르면 None.

    ours 는 workflow/nodes/pointing.py 의 resolve · crop_target 을 그대로 따른다. 다르게 고르면
    ours 숫자가 실제 비서의 숫자가 아니게 된다. 단서는 실제 비서의 기본값(enabled_cues=None)이다.
    크롭은 답 노드처럼 fit_for_vlm 을 거친다. full · mark 는 원본 그대로 두고 VLMClient 가 줄인다.
    두 번째 값은 ours 가 한 일(고름 / 손 크롭 / 되물음)과 고른 이름이다.
    """
    height, width = image.shape[:2]
    size = (width, height)
    question = sample["utterance"]
    resolution = resolve_target(
        candidates=candidates,
        hand=hand,
        gaze_vector=None,
        text=question,
        frame_size=size,
        prev_target_label=sample["prev_label"],
        image=image,
    )
    # 손 우선 크롭이 걸리면 고른 후보보다 먼저다(resolve 노드가 chosen 을 지운다).
    region = hand_crop_region(candidates, hand, size)
    if region is not None:
        # 손 크롭 사각형은 이미 최종 크기라 여백을 더하지 않는다(crop_target 노드와 같음).
        crop = cut_target(image, region, margin=0.0)
        ours = fit_for_vlm(crop), HAND_CONTEXT.format(question=question)
        mark_box, info = region, {"ours": "손 크롭", "chosen_label": ""}
    elif resolution.chosen is not None:
        box = resolution.chosen.detection.bbox
        ours = fit_for_vlm(cut_target(image, box)), POINTING_CONTEXT.format(question=question)
        mark_box, info = box, {"ours": "고름", "chosen_label": resolution.chosen.detection.label}
    else:
        ours = mark_box = None
        info = {"ours": "되물음", "chosen_label": ""}
    gt = fit_for_vlm(cut_target(image, sample["gt"])), POINTING_CONTEXT.format(question=question)
    # ours 가 안 부르면 mark 도 안 부른다. 둘은 고른 곳이 같아야 자르기 vs 표시하기만 다르다.
    mark = None
    if mark_box is not None:
        mark = draw_mark(image, mark_box), MARK_CONTEXT.format(question=question)
    inputs = {"full": (image, question), "ours": ours, "gt": gt, "mark": mark}
    return inputs, info


def parse_verdict(text: str | None) -> str:
    """채점기 답에서 판정 낱말을 꺼낸다. 따옴표 · 마침표는 떼고 앞부분만 본다.

    판정 낱말로 시작하지 않으면 채점 실패다.
    """
    if not text:
        return JUDGE_FAILED
    head = text.strip().strip("'\"‘’“”.")
    return next((v for v in VERDICTS if head.startswith(v)), JUDGE_FAILED)


def judge_answer(
    client: OpenAI, model: str, gt_image: np.ndarray, gt_label: str, question: str, answer: str
) -> str:
    """정답 크롭 · 정답 이름 · 질문 · 답을 주고 맞음 / 틀림 / 모름 을 받는다.

    VLMClient.ask 는 지시문이 비서 문장으로 고정이라 채점에 못 쓴다. 호출 옵션은
    vlm/client.py 와 같게 둔다. 질문은 맥락 문장을 뺀 사용자 질문 그대로 준다.
    """
    text = f"참고 이름: {gt_label}\n사용자 질문: {question}\n비서 대답: {answer}"
    content = [
        {"type": "input_text", "text": text},
        {"type": "input_image", "image_url": encode_image(gt_image), "detail": "auto"},
    ]
    try:
        response = client.responses.create(
            model=model,
            instructions=JUDGE_INSTRUCTIONS,
            input=[{"role": "user", "content": content}],
            reasoning={"effort": "none"},
            max_output_tokens=JUDGE_MAX_TOKENS,
        )
    except OpenAIError:
        return JUDGE_FAILED
    return parse_verdict(response.output_text)


def ask_and_judge(vlm: VLMClient, judge: Callable[..., str], job: dict) -> dict:
    """VLM 에 한 번 묻고 그 답을 채점한다. 답이 없으면(호출 실패) 채점하지 않는다."""
    image, prompt = job["input"]
    answer = vlm.ask(Frame(image=image, timestamp=0.0, source_id="crop_effect"), prompt)
    if answer is None:
        verdict = CALL_FAILED
    else:
        verdict = judge(job["gt_image"], job["row"]["gt_label"], job["row"]["utterance"], answer)
    return {**job["row"], "question": prompt, "answer": answer or "", "verdict": verdict}


def stable_rate(rows: list[dict]) -> float | None:
    """반복끼리 판정이 같은 사진의 비율. 두 번 이상 판정된 사진만 센다."""
    verdicts: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["verdict"] in VERDICTS:
            verdicts[r["file"]].append(r["verdict"])
    repeated = [v for v in verdicts.values() if len(v) >= 2]
    if not repeated:
        return None
    return round(sum(len(set(v)) == 1 for v in repeated) / len(repeated), 3)


def _metrics(rows: list[dict]) -> dict:
    counts = Counter(r["verdict"] for r in rows)
    n = len(rows)
    judged = sum(counts[v] for v in VERDICTS)
    return {
        "n": n,
        "correct_rate": round(counts["맞음"] / n, 3),
        # 다른 물건을 말한 답. 되묻지 않고 단정한 오답이라 run_ablation 의 확신 오답률 자리다.
        "wrong_rate": round(counts["틀림"] / n, 3),
        "unsure_rate": round(counts["모름"] / n, 3),
        "not_called_rate": round(counts[NOT_CALLED] / n, 3),
        "answered_accuracy": round(counts["맞음"] / judged, 3) if judged else None,
        "stable_rate": stable_rate(rows),
        "counts": {name: counts[name] for name in OUTCOMES},
    }


def summarize(rows: list[dict]) -> dict:
    """한 조건의 지표를 묶음마다(valid · s04)와 전체로. 분모는 사진 수 × 반복이다.

    되물어 안 부른 답을 분모에서 빼면 많이 되묻는 조건일수록 맞음률이 부풀려진다
    (run_ablation 의 확신 오답률 분모와 같은 이유). 부른 것만의 맞음률은 answered_accuracy 다.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["set"]].append(r)
        groups["all"].append(r)
    return {name: _metrics(part) for name, part in sorted(groups.items())}


def write_rows_csv(out_dir: Path, run_id: int, condition: str, rows: list[dict]) -> Path:
    """답별 결과. 파일 이름의 run_id 로 DB 의 실행과 짝짓는다.

    이름에 콜론(crop:full)을 쓰지 않는다. 윈도우는 파일 이름에 콜론을 못 쓴다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}_crop-{condition}.csv"
    ordered = sorted(rows, key=lambda r: (r["file"], r["repeat"]))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    return path


def record(
    store: MemoryStore, rows: list[dict], conditions: dict, note: str, out_dir: Path
) -> list[tuple[str, int, dict, Path]]:
    """조건마다 eval_run 한 줄 · eval_sample 답마다 · CSV 하나. (조건, run_id, 지표, CSV) 목록."""
    results = []
    for condition in CONDITIONS:
        part = [r for r in rows if r["condition"] == condition]
        metrics = summarize(part)
        metrics["conditions"] = conditions
        # eval_sample 은 run_ablation 의 save_run 을 그대로 쓴다. 맞음 = 채점기가 '맞음'이라 한 답.
        samples = [
            {
                "gt_label": r["gt_label"],
                "pred_label": r["verdict"],
                "correct": r["verdict"] == "맞음",
            }
            for r in part
        ]
        run_id = save_run(store, f"crop-{condition}", samples, metrics, note)
        results.append(
            (condition, run_id, metrics, write_rows_csv(out_dir, run_id, condition, part))
        )
    return results


def pick_check_sample(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """사람이 채점할 표본. 판정마다 돌아가며 뽑아 맞음 · 틀림 · 모름이 고르게 섞이게 한다.

    맞음만 잔뜩 뽑으면 채점기가 '틀림' · '모름'을 맞게 가르는지 볼 수 없다.
    """
    rng = random.Random(seed)
    pools = {v: [r for r in rows if r["verdict"] == v] for v in VERDICTS}
    for pool in pools.values():
        rng.shuffle(pool)
    picked: list[dict] = []
    while len(picked) < n and any(pools.values()):
        for verdict in VERDICTS:
            if pools[verdict] and len(picked) < n:
                picked.append(pools[verdict].pop())
    return picked


def write_check_csv(path: Path, rows: list[dict]) -> Path:
    """사람이 human 칸에 맞음 / 틀림 / 모름 을 채울 표.

    utf-8-sig(BOM 붙음)로 쓴다. 윈도우 엑셀은 BOM 이 없으면 한글을 깨뜨려 연다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CHECK_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r[k] for k in CHECK_COLUMNS[:5]}, "judge": r["verdict"]})
    return path


def agreement(path: Path) -> tuple[int, int, list[dict]]:
    """(채점기와 사람이 같은 수, 사람이 채점한 수, 어긋난 줄).

    human 이 비었거나 세 판정 밖이면 안 센다.

    엑셀 · 넘버스로 저장하면 머리에 BOM 이 붙을 수 있어 utf-8-sig 로 읽는다.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["human"].strip() in VERDICTS]
    mismatched = [r for r in rows if r["human"].strip() != r["judge"]]
    return len(rows) - len(mismatched), len(rows), mismatched


def save_inputs(out_dir: Path, file: str, inputs: dict) -> None:
    """--dry-run 용. 조건마다 VLM 에 보낼 사진을 JPEG 로 남긴다. 안 부르는 조건은 건너뛴다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file).with_suffix("").as_posix().replace("/", "_")
    for condition, value in inputs.items():
        if value is None:
            continue
        ok, jpeg = cv2.imencode(".jpg", value[0])
        if ok:
            # cv2.imwrite 는 윈도우에서 경로에 한글이 있으면 조용히 실패해서 바이트로 쓴다.
            jpeg.tofile(out_dir / f"{stem}_{condition}.jpg")


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}"


def format_row(condition: str, name: str, m: dict) -> str:
    return (
        f"{condition} | {name} | {m['n']} | {_pct(m['correct_rate'])} | {_pct(m['wrong_rate'])} | "
        f"{_pct(m['unsure_rate'])} | {_pct(m['not_called_rate'])} | "
        f"{_pct(m['answered_accuracy'])} | {_pct(m['stable_rate'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="크롭 효과 — VLM 에 보낸 사진별 답 정답률")
    parser.add_argument("--labels", type=Path, default=POINTING_DIR / "labels.csv")
    parser.add_argument("--raw", type=Path, default=POINTING_DIR / "raw")
    parser.add_argument(
        "--sessions", type=int, nargs="+", default=list(DEFAULT_SESSIONS), help="valid 세션"
    )
    parser.add_argument("--weights", default=config.DETECTOR_WEIGHTS, help="검출기 가중치")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="같은 입력 반복")
    parser.add_argument("--limit", type=int, help="앞에서 이만큼만 (시험 호출용)")
    parser.add_argument("--dry-run", action="store_true", help="VLM 없이 입력 사진만 --out 에")
    parser.add_argument("--sample", type=int, default=0, help="사람 채점용 표본 수")
    parser.add_argument("--agreement", type=Path, help="사람이 채운 표본으로 일치율만 찍는다")
    parser.add_argument("--db", default=config.DATABASE_URL, help="결과를 남길 DB")
    parser.add_argument("--note", default="", help="이번 실행 메모")
    parser.add_argument("--out", type=Path, default=POINTING_DIR / "runs", help="CSV 폴더")
    args = parser.parse_args()

    if args.agreement is not None:
        matched, total, mismatched = agreement(args.agreement)
        if not total:
            raise SystemExit(f"human 칸을 채운 줄이 없다: {args.agreement}")
        print(f"채점기 · 사람 일치 {matched}/{total} ({matched / total:.1%})")
        for r in mismatched:
            print(f"    {r['file']} · {r['condition']} · 채점기 {r['judge']} · 사람 {r['human']}")
        return

    samples = pointing_samples(load_samples(args.labels, "valid", sorted(args.sessions)))
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"세션 {sorted(args.sessions)} 에 가리킴 질문 사진이 없다: {args.labels}")
    # 검출을 다 돌린 뒤에 키가 없어 멈추면 시간만 버린다. 먼저 본다.
    if not args.dry_run and not config.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY 가 없다 — .env 에 넣거나 --dry-run 으로 입력만 본다")
    missing = missing_weights([args.weights])
    if missing:
        raise SystemExit(f"가중치 파일이 없다: {missing[0]}")

    # ultralytics 는 CI 에 없어서 여기서 불러온다.
    from jarviseo.pointing.detector import ObjectDetector

    detector = ObjectDetector(args.weights)
    hand_detector = HandDetector()
    vlm = None if args.dry_run else VLMClient()
    judge = None if vlm is None else partial(judge_answer, vlm.client, vlm.model)

    rows: list[dict] = []
    kinds: Counter = Counter()
    # 사진은 하나씩 읽고 그 사진의 호출이 끝나면 버린다.
    # 1080p 원본을 다 들고 있으면 저사양 PC 에 무겁다.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, s in enumerate(samples, 1):
            image = read_image(args.raw / s["file"])
            if image is None:
                raise SystemExit(f"사진을 못 읽었다: {args.raw / s['file']}")
            hand = pick_pointing_hand(hand_detector.detect(image))
            inputs, info = build_inputs(s, image, detector.detect(image), hand)
            kinds[info["ours"]] += 1
            if args.dry_run:
                save_inputs(args.out / "dry_run", s["file"], inputs)
                continue
            base = {
                "file": s["file"],
                "set": set_name(s["file"]),
                "gt_label": s["gt_label"],
                "utterance": s["utterance"],
                **info,
            }
            jobs = []
            for repeat in range(1, args.repeats + 1):
                for condition in CONDITIONS:
                    row = {**base, "repeat": repeat, "condition": condition}
                    if inputs[condition] is None:
                        rows.append({**row, "question": "", "answer": "", "verdict": NOT_CALLED})
                    else:
                        jobs.append(
                            {"row": row, "input": inputs[condition], "gt_image": inputs["gt"][0]}
                        )
            rows += pool.map(partial(ask_and_judge, vlm, judge), jobs)
            print(f"{i}/{len(samples)} {s['file']} · ours {info['ours']} · 호출 {len(jobs)}")

    print(
        f"\n{len(samples)}장 · ours 고름 {kinds['고름']} · 손 크롭 {kinds['손 크롭']} · "
        f"되물음 {kinds['되물음']}"
    )
    if args.dry_run:
        print(f"입력 사진: {args.out / 'dry_run'}")
        return

    store = MemoryStore(database_url=args.db)
    store.init_schema()
    conditions = {
        "commit": git_commit(),
        "weights": weights_fingerprint(detector.weights_path),
        "sessions": sorted(args.sessions),
        "vlm_model": vlm.model,
        "judge_model": vlm.model,
        "repeats": args.repeats,
        "samples": len(samples),
    }
    results = record(store, rows, conditions, args.note, args.out)
    print("\n조건 | 묶음 | 답 수 | 맞음 | 틀림 | 모름 | 안 부름 | 부른 것 맞음 | 반복 일치")
    for condition, run_id, metrics, path in results:
        for name in ("valid", "s04", "all"):
            if name in metrics:
                print(format_row(condition, name, metrics[name]))
        failed = {k: metrics["all"]["counts"][k] for k in (CALL_FAILED, JUDGE_FAILED)}
        print(f"    run {run_id} · 실패 {failed} · {path}")
    if args.sample:
        check = pick_check_sample(rows, args.sample)
        path = write_check_csv(args.out / f"{results[0][1]}_judge_check.csv", check)
        print(f"\n사람 채점 표본 {len(check)}줄: {path} (human 칸에 맞음 / 틀림 / 모름)")


if __name__ == "__main__":
    main()
