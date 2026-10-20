"""음성 루프 지연 집계 — p50 · p95 와 턴별 csv.

마이크 · 스피커 · API 는 부르지 않는다. 구간 값을 직접 넣는다.
"""

import csv
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "voice_loop.py"
_spec = importlib.util.spec_from_file_location("voice_loop", _SCRIPT)
voice_loop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(voice_loop)


def test_percentiles_pick_real_values():
    """p50 · p95 는 measure_latency.py 와 같이 실제 있었던 값을 고른다(nearest-rank)."""
    values = [float(v) for v in range(1, 21)]  # 1~20

    assert voice_loop.percentiles(values) == (11.0, 20.0)
    assert voice_loop.percentiles([5.0]) == (5.0, 5.0)


def test_csv_log_writes_each_turn_at_once(tmp_path):
    """턴마다 바로 한 줄씩 쓴다. 없는 구간은 비우고, 머리줄에만 BOM 이 붙는다."""
    path = tmp_path / "latency" / "voice_loop.csv"
    log = voice_loop.CsvLog(path)
    log({"질문": "이거 뭐야?", "끊김": "", "STT 왕복": 812.4, "trigger": "WAKEWORD"})

    with path.open(encoding="utf-8-sig") as f:  # 첫 턴만 쓰고 꺼져도 남아 있다
        (first,) = list(csv.DictReader(f))
    assert (first["턴"], first["질문"], first["STT 왕복"]) == ("1", "이거 뭐야?", "812")
    assert first["감지 → 멈춤"] == ""

    log({"질문": "그만", "끊김": "O", "감지 → 멈춤": 180.0, "trigger": "BARGE_IN"})
    raw = path.read_bytes()
    assert raw.count("﻿".encode()) == 1
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [r["턴"] for r in rows] == ["1", "2"]
    assert (rows[1]["trigger"], rows[1]["감지 → 멈춤"]) == ("BARGE_IN", "180")
