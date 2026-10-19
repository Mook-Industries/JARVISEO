"""단계별 지연 리포트 — 세션 하나의 p50 / p95 를 찍는다.  담당: 최홍묵

scripts/assistant.py --db 가 쌓은 세션을 읽어, 단계별(stt · route · detect · vlm · tts · total)
p50 / p95 / 건수와 길별(가리킴 · 성분 · 나머지) total, "잠깐만요"를 읽은 턴 비율을 찍는다.
결과는 docs/experiments.md 에 남긴다.

턴을 여기서 돌리지 않는다. 턴을 돌리는 흐름은 assistant.py 에 있고, 측정용 흐름을 따로 만들면
시연 흐름과 달라져 숫자가 실제를 대표하지 못한다. 이 스크립트는 DB 를 읽기만 한다.

실행 (레포 맨 위 폴더에서):
    python scripts/measure_latency.py                 # 턴이 있는 가장 최근 세션
    python scripts/measure_latency.py --session 3
    python scripts/measure_latency.py --db postgresql+psycopg://jarviseo:jarviseo@localhost:5432/jarviseo

못 재는 것:
    resolve · crop · recall · ingredient 단계는 DB 열이 없어 total 안에만 들어 있다.
    첫 피드백 시각은 열이 없다(turn_voice). 대신 "잠깐만요"를 읽은 턴 비율을 찍는다.
"""

import argparse

from sqlalchemy import func, select

from jarviseo.memory import MemoryStore, SessionTurn, TurnInference, TurnIngredient, TurnVoice

# assistant.py 의 DEFAULT_DB 와 같게 둔다. 돌리고 나서 인자 없이 바로 잴 수 있게.
DEFAULT_DB = "sqlite:///data/assistant.db"
# store.STAGE_COLUMNS 가 DB 열로 옮겨 적는 단계(같은 열의 다른 이름은 뺐다)
STAGES = ("stt", "route", "detect", "vlm", "tts", "total")
PATHS = ("pointing", "ingredient", "other")
# p95 는 느린 쪽 5% 라, 20턴이면 한 턴짜리 값이 되어 흔들린다. 로드맵 기준 30턴.
MIN_TURNS = 30


def latest_session_with_turns(store: MemoryStore) -> int | None:
    """턴이 하나라도 있는 가장 최근 세션. 켰다가 바로 끈 빈 세션은 건너뛴다."""
    with store.session() as db:
        stmt = select(func.max(SessionTurn.session_id))
        return db.scalar(stmt)


def percentiles(values: list[float]) -> tuple[float, float]:
    """p50 · p95. store.latency_percentiles 와 같은 nearest-rank(실제 있었던 값을 고름)."""
    ordered = sorted(values)

    def pick(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return pick(0.50), pick(0.95)


def total_by_path(store: MemoryStore, session_id: int) -> dict[str, list[float]]:
    """길별 total. 성분 기록이 있으면 성분, 검출 시간이 있으면 가리킴, 둘 다 없으면 나머지.

    DB 에 의도 열이 없어서, 실제로 지나간 길의 흔적으로 나눈다. 성분 질문도 가리킴을 먼저 돌아
    검출 시간이 남으므로 성분을 먼저 본다.

    끊긴 턴(다시 불러 답을 버림)은 뺀다. 대답 소리가 안 났는데도 log_turn 이 그때까지의 단계
    시간을 더해 total_ms 를 채우므로(store.py), 세면 total 이 실제보다 짧게 섞인다.
    """
    stmt = (
        select(SessionTurn.total_ms, TurnInference.vision_ms, TurnIngredient.turn_id)
        # 바깥 조인이어야 성분 기록이 없는 턴(대부분)이 빠지지 않는다
        .outerjoin(TurnInference, TurnInference.turn_id == SessionTurn.turn_id)
        .outerjoin(TurnIngredient, TurnIngredient.turn_id == SessionTurn.turn_id)
        .outerjoin(TurnVoice, TurnVoice.turn_id == SessionTurn.turn_id)
        .where(
            SessionTurn.session_id == session_id,
            SessionTurn.total_ms.is_not(None),
            TurnVoice.is_interrupted.is_not(True),
        )
    )
    paths: dict[str, list[float]] = {path: [] for path in PATHS}
    with store.session() as db:
        for total, vision, ingredient in db.execute(stmt):
            if ingredient is not None:
                paths["ingredient"].append(float(total))
            elif vision is not None:
                paths["pointing"].append(float(total))
            else:
                paths["other"].append(float(total))
    return paths


def filler_ratio(store: MemoryStore, session_id: int) -> tuple[int, int]:
    """("잠깐만요"를 읽은 턴 수, 전체 턴 수). 첫 피드백 시각 열이 없어 대신 쓰는 지표다."""
    with store.session() as db:
        turns = db.scalar(select(func.count()).where(SessionTurn.session_id == session_id))
        filler = db.scalar(
            select(func.count())
            .select_from(TurnVoice)
            .join(SessionTurn, SessionTurn.turn_id == TurnVoice.turn_id)
            .where(SessionTurn.session_id == session_id, TurnVoice.is_filler_sent.is_(True))
        )
    return filler or 0, turns or 0


def report(store: MemoryStore, session_id: int) -> str:
    """세션 하나의 리포트 글자. experiments.md 표에 옮기기 쉽게 | 로 나눈다."""
    paths = total_by_path(store, session_id)
    # total 은 끊긴 턴을 뺀 값으로 센다(total_by_path 설명). 다른 단계는 store 의 계산을 그대로 쓴다
    totals = [ms for values in paths.values() for ms in values]
    lines = [f"세션 {session_id}"]
    if len(totals) < MIN_TURNS:
        lines.append(f"주의: {len(totals)}턴뿐이라 p95 를 믿기 어렵다 ({MIN_TURNS}턴 이상 권장)")

    lines += ["", "단계 | 건수 | p50 ms | p95 ms"]
    for stage in STAGES:
        if stage == "total":
            lines.append(_row(stage, totals))
            continue
        p = store.latency_percentiles(stage, session_id)
        if p["count"] == 0:
            lines.append(f"{stage} | 0 | - | -")
        else:
            lines.append(f"{stage} | {p['count']} | {p['p50']:.0f} | {p['p95']:.0f}")

    lines += ["", "길 | 건수 | total p50 | total p95"]
    lines += [_row(path, values) for path, values in paths.items()]

    filler, turns = filler_ratio(store, session_id)
    share = f" ({filler / turns:.0%})" if turns else ""
    lines += ["", f"잠깐만요: {filler}/{turns}턴{share}"]
    return "\n".join(lines)


def _row(name: str, values: list[float]) -> str:
    if not values:
        return f"{name} | 0 | - | -"
    p50, p95 = percentiles(values)
    return f"{name} | {len(values)} | {p50:.0f} | {p95:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="세션 하나의 단계별 지연 리포트")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"읽을 DB (기본 {DEFAULT_DB})")
    parser.add_argument("--session", type=int, help="세션 번호. 없으면 턴이 있는 가장 최근 세션")
    args = parser.parse_args()

    # init_schema() 는 부르지 않는다. 주소를 잘못 주면 빈 DB 를 만들지 말고 에러가 나야 한다.
    store = MemoryStore(database_url=args.db)
    session_id = args.session if args.session is not None else latest_session_with_turns(store)
    if session_id is None:
        raise SystemExit(f"턴이 있는 세션이 없다: {args.db}")
    print(report(store, session_id))


if __name__ == "__main__":
    main()
