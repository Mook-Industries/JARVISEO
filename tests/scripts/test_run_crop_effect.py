"""크롭 효과 실험 — 가리킴 질문 고르기, 조건 네 가지 입력(고름 · 되물음 · 손 크롭),
채점기 판정 읽기 · 호출 실패 · 채점 실패, 지표 분모 · 반복 일치, 기록(eval_run · CSV 이름),
사람 채점 표본 · 일치율.

VLM · YOLO · MediaPipe 없이 정해 둔 후보 · 손과 가짜 VLM · 가짜 채점 클라이언트로 돈다.
DB 기록은 store fixture 로 SQLite 와 실제 Postgres 에서 돈다.
"""

import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from openai import OpenAIError
from sqlalchemy import select

from jarviseo.memory import EvalRun, EvalSample
from jarviseo.pointing.crop import crop_target, fit_for_vlm, hand_crop_region
from jarviseo.pointing.hands import HandPose
from jarviseo.types import BBox, Detection
from jarviseo.vlm.client import FakeVLMClient
from jarviseo.workflow.nodes.answer import HAND_CONTEXT, POINTING_CONTEXT

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
# run_crop_effect 가 같은 폴더의 run_ablation 을 import 한다.
# 스크립트로 실행하면 그 폴더가 경로에 든다.
sys.path.append(str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location("run_crop_effect", _SCRIPTS / "run_crop_effect.py")
run_crop_effect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_crop_effect)

FRAME = (1920, 1080)
CENTER = BBox(860, 440, 1060, 640)  # 화면 가운데 컵
LEFT = BBox(400, 440, 600, 640)
RIGHT = BBox(1320, 440, 1520, 640)  # LEFT 와 화면 가운데를 두고 좌우 대칭
GT = BBox(100, 100, 500, 400)  # 고른 박스와 크기가 다른 정답 박스
QUESTION = "이거 뭐야?"


def _image() -> np.ndarray:
    return np.full((FRAME[1], FRAME[0], 3), 128, np.uint8)


def _det(bbox: BBox, label: str = "cup") -> Detection:
    return Detection(label=label, confidence=0.9, bbox=bbox)


def _hand(base: tuple[float, float], tip: tuple[float, float]) -> HandPose:
    """검지 뿌리(5) → 끝(8)만 의미 있는 손. 손 크롭은 방향만 쓴다."""
    landmarks = [(0.0, 0.0)] * 21
    landmarks[5], landmarks[8] = base, tip
    return HandPose(tuple(landmarks), ((0.0, 0.0, 0.0),) * 21, "Right", 0.9)


def _sample(utterance: str = QUESTION, gt: BBox = GT, file: str = "s01/a.jpg") -> dict:
    return {
        "file": file,
        "utterance": utterance,
        "gt": gt,
        "gt_label": "cup",
        "has_hand": False,
        "ambiguous": False,
        "prev_label": None,
    }


def _row(verdict: str, *, file: str = "s01/a.jpg", repeat: int = 1, condition: str = "full"):
    return {
        "file": file,
        "set": run_crop_effect.set_name(file),
        "repeat": repeat,
        "condition": condition,
        "ours": "고름",
        "chosen_label": "cup",
        "gt_label": "cup",
        "utterance": QUESTION,
        "question": QUESTION,
        "answer": "컵이에요.",
        "verdict": verdict,
    }


def _job(prompt: str = QUESTION) -> dict:
    row = {"file": "s01/a.jpg", "gt_label": "cup", "utterance": QUESTION, "condition": "ours"}
    return {"row": row, "input": (_image(), prompt), "gt_image": np.zeros((50, 50, 3), np.uint8)}


# --- 대상 사진 ----------------------------------------------------------------


def test_pointing_samples_keep_pointing_questions_only():
    """성분 질문은 VLM 을 안 부르고, 영역 질문은 원래 전체 사진이라 뺀다."""
    samples = [_sample(u) for u in (QUESTION, "이거 먹어도 돼?", "왼쪽에는 뭐가 있지?")]

    kept = run_crop_effect.pointing_samples(samples)

    assert [s["utterance"] for s in kept] == [QUESTION]


def test_set_name_keeps_s04_apart():
    """색 · 위치 말이 든 s04 는 valid 62장과 따로 본다."""
    assert run_crop_effect.set_name("s01/a.jpg") == "valid"
    assert run_crop_effect.set_name("s02/b.jpg") == "valid"
    assert run_crop_effect.set_name("s04/c.jpg") == "s04"


# --- 조건 네 가지 입력 ----------------------------------------------------------


def test_chosen_box_goes_to_ours_mark_and_gt():
    """고른 박스는 여백 15% 크롭 + 가리킴 맥락으로, 정답 박스는 따로, 전체 사진은 질문만."""
    image = _image()

    inputs, info = run_crop_effect.build_inputs(_sample(), image, [_det(CENTER)], None)

    assert info == {"ours": "고름", "chosen_label": "cup"}
    full_image, full_question = inputs["full"]
    assert full_image is image and full_question == QUESTION  # 맥락 문장 없이 그냥 통째로
    ours_image, ours_question = inputs["ours"]
    assert ours_image.shape == fit_for_vlm(crop_target(image, CENTER)).shape  # 260×260
    assert ours_question == POINTING_CONTEXT.format(question=QUESTION)
    gt_image, gt_question = inputs["gt"]
    assert gt_image.shape == fit_for_vlm(crop_target(image, GT)).shape  # 390×520
    assert gt_question == POINTING_CONTEXT.format(question=QUESTION)
    mark_image, mark_question = inputs["mark"]
    assert mark_image.shape == image.shape
    assert tuple(mark_image[440, 960]) == run_crop_effect.MARK_COLOR  # 고른 박스 윗변
    assert tuple(image[440, 960]) == (128, 128, 128)  # 원본은 그대로(다른 조건이 쓴다)
    assert mark_question == run_crop_effect.MARK_CONTEXT.format(question=QUESTION)


def test_clarify_skips_ours_and_mark():
    """좌우 대칭 컵을 손 없이 물으면 되묻는다. 실제 비서처럼 ours · mark 는 VLM 을 안 부른다."""
    inputs, info = run_crop_effect.build_inputs(
        _sample(), _image(), [_det(LEFT), _det(RIGHT)], None
    )

    assert info == {"ours": "되물음", "chosen_label": ""}
    assert inputs["ours"] is None
    assert inputs["mark"] is None
    assert inputs["full"] is not None
    assert inputs["gt"] is not None


def test_hand_crop_uses_hand_context_without_margin():
    """가리키는 손이 있는데 후보가 없으면 손 우선 크롭. 사각형이 최종 크기라 여백을 안 더한다."""
    image = _image()
    hand = _hand(base=(1068.0, 936.0), tip=(1100.0, 900.0))  # 화면 아래에서 오른쪽 위를 가리킴
    region = hand_crop_region([], hand, FRAME)

    inputs, info = run_crop_effect.build_inputs(_sample(), image, [], hand)

    assert info == {"ours": "손 크롭", "chosen_label": ""}
    ours_image, ours_question = inputs["ours"]
    assert ours_image.shape == (int(region.height), int(region.width), 3)  # 한 변 0.6H = 648
    assert ours_question == HAND_CONTEXT.format(question=QUESTION)
    mark_image, _ = inputs["mark"]
    assert tuple(mark_image[int(region.y1), int(region.x1) + 50]) == run_crop_effect.MARK_COLOR
    assert tuple(image[int(region.y1), int(region.x1) + 50]) == (128, 128, 128)


# --- 묻기 · 채점 ----------------------------------------------------------------


class _Responses:
    """client.responses 의 가짜. 받은 인자를 남기고 정해 둔 글자를 돌려주거나 에러를 낸다."""

    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self.text, self.error = text, error
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_text=self.text)


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("맞음", "맞음"),
        ("'틀림'.", "틀림"),
        ("모름입니다", "모름"),
        ("잘 모르겠다", run_crop_effect.JUDGE_FAILED),
        ("", run_crop_effect.JUDGE_FAILED),
        (None, run_crop_effect.JUDGE_FAILED),
    ],
)
def test_parse_verdict(text, verdict):
    """따옴표 · 마침표는 떼고 앞부분의 판정 낱말을 읽는다. 다른 말이면 채점 실패."""
    assert run_crop_effect.parse_verdict(text) == verdict


def test_judge_answer_sends_gt_image_and_reads_verdict():
    """채점기에는 정답 크롭 사진과 정답 이름 · 질문 · 답이 가고, 한 낱말 판정을 돌려받는다."""
    client = SimpleNamespace(responses=_Responses("맞음"))

    verdict = run_crop_effect.judge_answer(
        client, "model", np.zeros((50, 50, 3), np.uint8), "cup", QUESTION, "컵이에요."
    )

    assert verdict == "맞음"
    kwargs = client.responses.kwargs
    assert kwargs["instructions"] == run_crop_effect.JUDGE_INSTRUCTIONS
    text, image = kwargs["input"][0]["content"]
    assert "참고 이름: cup" in text["text"]
    assert f"사용자 질문: {QUESTION}" in text["text"]
    assert "비서 대답: 컵이에요." in text["text"]
    assert image["image_url"].startswith("data:image/jpeg;base64,")


def test_judge_api_error_is_judge_failed():
    """채점 호출이 실패하면 맞음 · 틀림에 섞지 않고 채점 실패로 둔다."""
    client = SimpleNamespace(responses=_Responses(error=OpenAIError("down")))

    verdict = run_crop_effect.judge_answer(
        client, "model", np.zeros((50, 50, 3), np.uint8), "cup", QUESTION, "컵이에요."
    )

    assert verdict == run_crop_effect.JUDGE_FAILED


def test_answer_is_judged_with_user_question():
    """VLM 에는 맥락 문장이 붙은 질문이, 채점기에는 맥락 없는 사용자 질문이 간다."""
    prompt = POINTING_CONTEXT.format(question=QUESTION)
    vlm = FakeVLMClient(["마우스예요."])
    judged = []

    def judge(gt_image, gt_label, question, answer):
        judged.append((gt_label, question, answer))
        return "틀림"

    row = run_crop_effect.ask_and_judge(vlm, judge, _job(prompt))

    assert row["verdict"] == "틀림"
    assert row["answer"] == "마우스예요."
    assert row["question"] == prompt
    assert judged == [("cup", QUESTION, "마우스예요.")]
    ((_, asked, _),) = vlm.asked
    assert asked == prompt


def test_call_failure_is_not_judged():
    """VLM 이 답을 못 주면 호출 실패로 두고 채점기를 부르지 않는다."""
    judged = []

    row = run_crop_effect.ask_and_judge(
        FakeVLMClient([None]), lambda *args: judged.append(args) or "맞음", _job()
    )

    assert row["verdict"] == run_crop_effect.CALL_FAILED
    assert judged == []


# --- 지표 · 기록 ----------------------------------------------------------------


def test_summarize_divides_by_all_answers_and_counts_stability():
    """분모는 안 부른 답까지 전체다. 반복 일치는 두 번 이상 판정된 사진만 센다."""
    not_called = run_crop_effect.NOT_CALLED
    rows = [
        _row("맞음", file="s01/a.jpg", repeat=1),
        _row("맞음", file="s01/a.jpg", repeat=2),
        _row(not_called, file="s01/b.jpg", repeat=1),
        _row(not_called, file="s01/b.jpg", repeat=2),
        _row("틀림", file="s04/c.jpg", repeat=1),
        _row("모름", file="s04/c.jpg", repeat=2),
    ]

    metrics = run_crop_effect.summarize(rows)

    valid, s04 = metrics["valid"], metrics["s04"]
    assert valid["n"] == 4
    assert valid["correct_rate"] == 0.5  # 안 부른 2개를 빼면 1.0 으로 부풀려진다
    assert valid["not_called_rate"] == 0.5
    assert valid["answered_accuracy"] == 1.0
    assert valid["stable_rate"] == 1.0  # a 만 셈. b 는 판정이 없다
    assert s04["wrong_rate"] == 0.5
    assert s04["unsure_rate"] == 0.5
    assert s04["stable_rate"] == 0.0
    assert metrics["all"]["n"] == 6


def test_record_writes_one_run_per_condition(store, tmp_path):
    """조건마다 eval_run 한 줄 · 답마다 eval_sample · CSV 하나. 파일 이름에 콜론이 없다."""
    rows = [
        _row(verdict, condition=condition)
        for condition, verdict in zip(
            run_crop_effect.CONDITIONS, ("틀림", "맞음", "맞음", "모름"), strict=True
        )
    ]

    results = run_crop_effect.record(store, rows, {"commit": "abc"}, "시험", tmp_path)

    assert [condition for condition, *_ in results] == list(run_crop_effect.CONDITIONS)
    with store.session() as db:
        runs = db.scalars(select(EvalRun).order_by(EvalRun.run_id)).all()
        assert [run.variant for run in runs] == ["crop-full", "crop-ours", "crop-gt", "crop-mark"]
        assert runs[0].metrics["conditions"] == {"commit": "abc"}
        assert runs[0].note == "시험"
        samples = db.scalars(select(EvalSample).order_by(EvalSample.sample_id)).all()
        assert [(s.pred_label, s.is_correct) for s in samples] == [
            ("틀림", False),
            ("맞음", True),
            ("맞음", True),
            ("모름", False),
        ]
    for condition, run_id, _, path in results:
        assert path.name == f"{run_id}_crop-{condition}.csv"
        assert ":" not in path.name  # 윈도우는 파일 이름에 콜론을 못 쓴다
    with results[0][3].open(encoding="utf-8") as f:
        assert [r["verdict"] for r in csv.DictReader(f)] == ["틀림"]


# --- 사람 채점 표본 · 일치율 ------------------------------------------------------


def test_check_sample_mixes_verdicts():
    """판정마다 돌아가며 뽑는다. 맞음이 대부분이어도 틀림 · 모름이 빠지지 않는다."""
    rows = [_row("맞음") for _ in range(10)] + [_row("틀림"), _row("틀림"), _row("모름")]
    rows.append(_row(run_crop_effect.NOT_CALLED))  # 판정이 없는 답은 뽑지 않는다

    picked = run_crop_effect.pick_check_sample(rows, 6)

    assert sorted(r["verdict"] for r in picked) == sorted(["맞음"] * 3 + ["틀림"] * 2 + ["모름"])


def test_check_csv_round_trip_counts_agreement(tmp_path):
    """표본 표는 BOM 을 붙여 쓰고, 사람이 채운 칸만 일치율에 센다."""
    path = run_crop_effect.write_check_csv(
        tmp_path / "check.csv", [_row("맞음"), _row("틀림"), _row("모름")]
    )
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")  # 윈도우 엑셀이 한글을 안 깨뜨리게
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row, human in zip(rows, ("맞음", "맞음", ""), strict=True):
        row["human"] = human  # 사람은 두 줄만 채웠다
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_crop_effect.CHECK_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    matched, total, mismatched = run_crop_effect.agreement(path)

    assert (matched, total) == (1, 2)
    assert [r["judge"] for r in mismatched] == ["틀림"]


# --- 실행 전 확인 ----------------------------------------------------------------


def _labels(tmp_path: Path, utterance: str) -> Path:
    path = tmp_path / "labels.csv"
    header = (
        "file,session,split,utterance,gt_x1,gt_y1,gt_x2,gt_y2,gt_label,"
        "has_hand,ambiguous,prev_label,excluded\n"
    )
    path.write_text(header + f"s01/a.jpg,1,valid,{utterance},1,1,9,9,cup,0,0,,0\n", "utf-8")
    return path


def test_main_stops_without_api_key(tmp_path, monkeypatch):
    """키가 없으면 검출을 돌리기 전에 멈춘다. 다 돌린 뒤 멈추면 시간만 버린다."""
    monkeypatch.setattr(run_crop_effect.config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(
        sys, "argv", ["run_crop_effect.py", "--labels", str(_labels(tmp_path, QUESTION))]
    )

    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        run_crop_effect.main()


def test_main_stops_without_pointing_question(tmp_path, monkeypatch):
    """가리킴 질문 사진이 하나도 없으면 멈춘다(성분 질문만 있는 정답표)."""
    labels = _labels(tmp_path, "이거 먹어도 돼?")
    monkeypatch.setattr(sys, "argv", ["run_crop_effect.py", "--labels", str(labels)])

    with pytest.raises(SystemExit, match="가리킴 질문 사진이 없다"):
        run_crop_effect.main()
