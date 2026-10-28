"""성능 리포트 데이터 — 팀이 남긴 기록만 보여 준다(목업 숫자 대신).

    단계별 지연    턴마다 그래프가 session_turn · turn_voice · turn_inference 에 적은 값
                  (memory/store.py 의 STAGE_COLUMNS). 0ms 이하는 뺀다 — 입력창 질문은 받아쓰기 ·
                  읽기가 없어 0 으로 남고, 시계가 맞지 않은 턴은 전체가 음수로 남아 중앙값을 0 으로
                  끌어내렸다(2026-10-27: 음성 인식 · 음성 합성 중앙값 0ms).
                  계산은 latency_percentiles 와 같은 nearest-rank 다.
    평가 실행      eval_run 테이블 (scripts/run_ablation.py 가 --db 로 남긴다, 최홍묵)
    팀 실측 결과   docs/experiments.md 에 팀원이 적은 최종 숫자. 표의 출처(절 · 날짜)를 같이 단다

docs 의 숫자를 바꾸면 RESULTS 도 같이 바꾼다. 여기에 숫자를 새로 만들지 않는다.
"""

from __future__ import annotations

from sqlalchemy import select

from jarviseo.memory.models import EvalRun, SessionTurn
from jarviseo.memory.store import STAGE_COLUMNS

__all__ = ["KPIS", "LATENCY_STAGES", "RESULTS", "report"]

# 리포트에 보이는 단계(memory/store.py 의 STAGE_COLUMNS 이름)
LATENCY_STAGES = (
    ("stt", "음성 인식", "STT"),
    ("detect", "물체 · 손 검출", "DETECT"),
    ("vlm", "장면 이해", "VLM"),
    ("tts", "음성 합성 첫 소리", "TTS"),
    ("total", "말 끝 → 대답 첫 소리", "TOTAL"),
)

# (기능, 담당, 지표, 결과, 기준, 통과, 출처) — docs/experiments.md 그대로
RESULTS = [
    (
        "웨이크워드 “자비서”",
        "문태현",
        "FRR(조용 · 소음) · FAR",
        "5.6% · 12.5% · 5%",
        "≤ 10% · ≤ 20% · ≤ 10%",
        True,
        "합격 기준 #117 · 임계값 0.97 (TV 30분 3회는 미달)",
    ),
    (
        "시연 마이크 “자비서” 20개",
        "문태현",
        "FRR",
        "1/20",
        "—",
        True,
        "합격 기준 #117",
    ),
    (
        "받아쓰기 (gpt-transcribe)",
        "문태현",
        "정답 · CER · 중앙 지연",
        "16/20 · 0.070 · 892ms",
        "—",
        True,
        "합격 기준 #117",
    ),
    (
        "기억 근거율",
        "문태현",
        "근거 · 모름 · 지어냄",
        "15 · 15 · 0",
        "지어냄 0",
        True,
        "합격 기준 #117 · 질문 30개",
    ),
    (
        "가리킴 (R0 yolov8n · v2, 지금 기본값)",
        "최홍묵",
        "정답률 · 되묻기율 · 후보 재현율",
        "22.6% · 72.6% · 50.0%",
        "—",
        False,
        "R0 · R2 · R3 가리킴 valid 62장 run 32 · 33 (2026-10-27)",
    ),
    (
        "가리킴 (R2 detector_r2 · v2)",
        "최홍묵",
        "정답률 · 되묻기율 · 후보 재현율",
        "32.3% · 54.8% · 61.3%",
        "—",
        False,
        "R0 · R2 · R3 가리킴 valid 62장 run 34 · 35 (2026-10-27)",
    ),
    (
        "성분표 OCR (Google Vision + 전처리)",
        "권용현",
        "놓친 알레르기 · 다시 찍기",
        "6 / 139 · 0장 (21장)",
        "놓침 0",
        False,
        "반사 점수 오탐 · 혼입 목록 오판 · 실패 유형 분류 (2026-10-26)",
    ),
]


# 위 카드 4개 (이름, 값, 단위, 막대 %, 설명, 기준·출처) — RESULTS 와 같은 출처
KPIS = [
    (
        "지시 대상 특정 정확도",
        "22.6",
        "%",
        22.6,
        "가리킴 valid 62장 · R0(지금 기본값) · v2 (최홍묵)",
        "답한 것만 보면 64.7% · R2 는 32.3%",
    ),
    ("되묻기 비율", "72.6", "%", 72.6, "같은 실행에서 되물은 비율", "낮을수록 좋음 · R2 는 54.8%"),
    (
        "웨이크워드 FRR (조용)",
        "5.6",
        "%",
        5.6,
        "임계값 0.97 · 소음 12.5% (문태현)",
        "기준 ≤ 10% · 통과",
    ),
    (
        "성분표 놓친 알레르기",
        "6",
        "/139",
        4.3,
        "21장 · Google Vision + 전처리 (권용현)",
        "기준 0 · 다시 찍기 0장",
    ),
]


def latency(store, stage: str) -> dict[str, float]:
    """한 단계의 p50 · p95 · 개수. 0ms 이하(입력창 질문의 받아쓰기 · 읽기, 음수 전체)는 뺀다."""
    empty = {"p50": 0.0, "p95": 0.0, "count": 0}
    column = STAGE_COLUMNS.get(stage)
    if column is None:
        return empty
    with store.session() as db:
        values = sorted(float(v) for v in db.scalars(select(column).where(column > 0)))
    if not values:
        return empty

    def pick(q: float) -> float:  # nearest-rank (latency_percentiles 와 같다)
        return values[min(len(values) - 1, int(q * len(values)))]

    return {"p50": pick(0.50), "p95": pick(0.95), "count": len(values)}


def report(store) -> dict:
    latency_rows = []
    for stage, ko, en in LATENCY_STAGES:
        latency_rows.append({"stage": stage, "name": ko, "en": en, **latency(store, stage)})
    with store.session() as db:
        runs = [
            {
                "run_id": r.run_id,
                "variant": r.variant,
                "run_at": r.run_at.isoformat() if r.run_at else None,
                "samples": r.sample_count,
                "metrics": r.metrics or {},
                "note": r.note,
            }
            for r in db.scalars(select(EvalRun).order_by(EvalRun.run_at).limit(60))
        ]
        turns = len(db.scalars(select(SessionTurn.turn_id)).all())
    keys = ("model", "owner", "metric", "now", "goal", "ok", "source")
    return {
        "latency": latency_rows,
        "turns": turns,
        "eval_runs": runs,
        "results": [dict(zip(keys, row, strict=True)) for row in RESULTS],
        "kpis": [
            dict(zip(("name", "value", "unit", "bar", "desc", "goal"), k, strict=True))
            for k in KPIS
        ],
    }
