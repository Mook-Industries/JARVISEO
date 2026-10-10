"""기억 검색의 지연과 점수를 잰다.  담당: 문태현

1) 임베딩 왕복: 질문 문장을 여러 번 임베딩해 중앙값·p95 를 낸다 (OpenAI 키 필요)
2) 검색 지연: 관찰 기록을 --rows 개(무작위 벡터) 넣고 search 의 DB 쪽 시간을 잰다
3) 점수: 실제 임베딩으로 장면 8개를 넣고, 맞는 질문·상관없는 질문의 점수를 찍는다.
   config.MEMORY_MIN_SCORE 를 정하는 근거로 쓴다 (OpenAI 키 필요)

**테스트용 DB 의 테이블을 지우고 다시 만든다.** 개발 DB 를 가리키지 말 것.

실행:
    set JARVISEO_TEST_DATABASE_URL=postgresql+psycopg://jarviseo:jarviseo@localhost:5433/jarviseo_test
    python scripts/check_memory.py                 # 1,000행, API 측정 포함
    python scripts/check_memory.py --rows 10000 --skip-api
"""

import argparse
import os
import statistics
import time

import numpy as np

from jarviseo.memory import MemoryStore, TextEmbedder, VectorMemory
from jarviseo.memory.embed import DIM
from jarviseo.memory.models import Observation
from jarviseo.types import AssistantResponse, Intent, Utterance

SCENES = [
    "책상 위에 검은색 백팩이 놓여 있다",
    "냉장고 옆 선반에 초록색 물병이 있다",
    "현관 신발장 위에 파란 우산이 걸려 있다",
    "거실 소파 위에 리모컨과 안경집이 있다",
    "주방 식탁에 바나나 두 개와 사과 한 개가 있다",
    "침대 옆 협탁에서 휴대폰이 충전 중이다",
    "초코우유 성분표에 우유와 대두가 적혀 있다",
    "약통에 타이레놀 500mg 이라고 적혀 있다",
]
# (질문, 맞는 장면 번호). None 은 기억과 상관없는 질문이라 아무것도 안 나와야 한다.
QUERIES = [
    ("아까 본 가방 어디 있었지?", 0),
    ("물병 어디 뒀더라", 1),
    ("우산 어디 있었어?", 2),
    ("리모컨 어디 있었지?", 3),
    ("아까 식탁에 과일 뭐 있었어?", 4),
    ("내 폰 어디서 충전하고 있었지?", 5),
    ("아까 본 우유에 뭐 들어 있었지?", 6),
    ("아까 먹은 약 이름이 뭐였지?", 7),
    ("오늘 날씨 어때?", None),
    ("지금 몇 시야?", None),
    ("농담 하나 해 줘", None),
    ("서울에서 부산까지 얼마나 걸려?", None),
]


def _unit(rng: np.random.Generator) -> list[float]:
    v = rng.standard_normal(DIM)
    return (v / np.linalg.norm(v)).tolist()


def _summary(values: list[float]) -> str:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return f"중앙값 {statistics.median(values):.0f}ms, p95 {p95:.0f}ms, 최대 {ordered[-1]:.0f}ms"


class _RandomEmbedder:
    """API 없이 무작위 벡터를 준다. 검색 지연에서 DB 쪽 시간만 보려고 쓴다."""

    model = "random"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.last_latency_ms = 0.0

    def embed(self, text: str) -> list[float]:
        return _unit(self.rng)


def _fresh(url: str) -> tuple[MemoryStore, int, int]:
    store = MemoryStore(url)
    store.drop_all()
    store.init_schema()
    user_id = store.ensure_user()
    session_id = store.start_session(user_id)
    utterance = Utterance(text="측정", started_at=0.0, ended_at=1.0)
    response = AssistantResponse(text="", intent=Intent.GENERAL)
    turn_id = store.log_turn(session_id, utterance, response)
    return store, user_id, turn_id


def measure_embedding(times: int) -> None:
    embedder = TextEmbedder()
    embedder.embed("연결을 미리 열어 둔다")
    latencies = []
    for i in range(times):
        embedder.embed(QUERIES[i % len(QUERIES)][0])
        latencies.append(embedder.last_latency_ms)
    print(f"[임베딩 왕복] {times}회 {_summary(latencies)}")


def measure_search(url: str, rows: int, times: int) -> None:
    rng = np.random.default_rng(0)
    store, user_id, turn_id = _fresh(url)
    with store.session() as db:
        db.add_all(
            Observation(
                user_id=user_id,
                turn_id=turn_id,
                description=f"장면 {i}",
                embedding=_unit(rng),
                embed_model="random",
            )
            for i in range(rows)
        )
    memory = VectorMemory(store, _RandomEmbedder(rng), min_score=-1.0)
    memory.search(user_id, "준비")
    latencies = []
    for _ in range(times):
        memory.search(user_id, "아까 본 그거")
        latencies.append(memory.last_latency_ms)
    print(f"[검색 지연] {store.engine.dialect.name} {rows}행 {times}회 {_summary(latencies)}")


def measure_scores(url: str) -> None:
    store, user_id, turn_id = _fresh(url)
    memory = VectorMemory(store, TextEmbedder(), min_score=-1.0)
    ids = [memory.remember_observation(user_id, turn_id, scene) for scene in SCENES]
    right, wrong, unrelated = [], [], []
    print("[점수] 질문 → 1등 장면(점수) / 맞는 장면 점수")
    for query, answer in QUERIES:
        hits = memory.search(user_id, query, top_k=len(SCENES))
        top = hits[0]
        if answer is None:
            unrelated.append(top.score)
            print(f"  {query} → {top.text} ({top.score:.3f}) / 상관없는 질문")
            continue
        scores = {h.memory_id: h.score for h in hits}
        correct = scores.pop(str(ids[answer]))
        right.append(correct)
        wrong.append(max(scores.values()))
        mark = "O" if top.memory_id == str(ids[answer]) else "X"
        print(f"  {mark} {query} → {top.text} ({top.score:.3f}) / {correct:.3f}")
    print(
        f"  맞는 장면 최저 {min(right):.3f}, 틀린 장면 최고 {max(wrong):.3f}, "
        f"상관없는 질문 1등 최고 {max(unrelated):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=os.getenv("JARVISEO_TEST_DATABASE_URL"))
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--times", type=int, default=30)
    parser.add_argument("--skip-api", action="store_true", help="OpenAI 를 부르는 1)·3) 은 뺀다")
    args = parser.parse_args()
    if not args.db:
        parser.error("--db 나 JARVISEO_TEST_DATABASE_URL 로 테스트용 DB 를 알려 주세요")

    started = time.monotonic()
    if not args.skip_api:
        measure_embedding(args.times)
    measure_search(args.db, args.rows, args.times)
    if not args.skip_api:
        measure_scores(args.db)
    print(f"끝 ({time.monotonic() - started:.0f}초)")


if __name__ == "__main__":
    main()
