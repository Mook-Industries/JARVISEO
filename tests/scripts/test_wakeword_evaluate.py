"""웨이크워드 평가 집계 — 임계값 고르기 · 마이크별 나누기 · 녹음 없는 기준 칸 빼기.

모델은 부르지 않는다. 파일별 최고 점수와 background 프레임 점수를 직접 만들어 넣는다.
"""

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "wakeword" / "evaluate.py"
_spec = importlib.util.spec_from_file_location("wakeword_evaluate", _SCRIPT)
evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate)

LAPTOP, CAMERA = "노트북 마이크", "카메라 마이크"
_KEY = ("threshold", "metric", "source", "env", "mic", "speech")


def _row(label, env, score, mic=LAPTOP, source="mic", file=""):
    return {
        "file": file or f"{label}-{env}-{mic}-{score}",
        "label": label,
        "source": source,
        "mic": mic,
        "env": env,
        "speech": "",
        "max_score": f"{score:.4f}",
    }


def _values(summary, th, metric, env, mic=""):
    """(count, n). 실제 녹음 행 중 임계값 · 지표 · 환경 · 마이크가 맞는 하나(말투 행은 뺀다)."""
    key = (th, metric, "mic", env, mic, "")
    (s,) = [s for s in summary if tuple(s[k] for k in _KEY) == key]
    return s["count"], s["n"]


def test_summarize_counts_at_given_threshold():
    """주어진 임계값(0.97)에서 FRR · FAR 를 센다."""
    rows = [
        _row("wake", "quiet", 0.99),
        _row("wake", "quiet", 0.95),
        _row("similar", "quiet", 0.98),
    ]

    summary = evaluate.summarize(rows, {}, [0.5, 0.97])

    assert _values(summary, 0.97, "FRR", "quiet") == (1, 2)  # 0.95 는 0.97 에 못 미쳤다
    assert _values(summary, 0.97, "FAR", "quiet") == (1, 1)
    assert _values(summary, 0.5, "FRR", "quiet") == (0, 2)


def test_summarize_splits_by_mic_only_with_two_mics():
    """실제 녹음 마이크가 둘 이상이면 마이크별 줄을 더하고, 하나면 더하지 않는다."""
    one = [_row("wake", "quiet", 0.99), _row("wake", "quiet", 0.5)]
    two = one + [_row("wake", "quiet", 0.5, mic=CAMERA)]

    assert {s["mic"] for s in evaluate.summarize(one, {}, [0.97])} == {""}
    summary = evaluate.summarize(two, {}, [0.97])
    assert _values(summary, 0.97, "FRR", "quiet") == (2, 3)  # 전체 행
    assert _values(summary, 0.97, "FRR", "quiet", LAPTOP) == (1, 2)
    assert _values(summary, 0.97, "FRR", "quiet", CAMERA) == (1, 1)


def test_passing_skips_criteria_without_recordings():
    """녹음이 없는 기준 칸(일반 대화)은 빼고 판정하고, 뺀 칸을 알려 준다."""
    hour = round(3600 * evaluate.SAMPLE_RATE / evaluate.FRAME)  # 1시간 분량의 프레임
    quiet_tv = np.zeros(hour)
    loud_tv = quiet_tv.copy()
    loud_tv[100] = 0.99  # 한 번 깨어났다
    rows = [
        _row("wake", "quiet", 0.99),
        _row("wake", "noisy", 0.99),
        _row("similar", "quiet", 0.1),
        _row("background", "tv", 0.0, file="tv.wav"),
    ]

    ok, missing = evaluate.passing(evaluate.summarize(rows, {"tv.wav": quiet_tv}, [0.97]), [0.97])
    assert ok == [0.97]
    assert missing == [("FA/h", "talk")]

    ok, _ = evaluate.passing(evaluate.summarize(rows, {"tv.wav": loud_tv}, [0.97]), [0.97])
    assert ok == []  # TV 에서 한 번 깨어나면 불합격
