"""기억 질문 평가셋으로 기억 검색을 잰다 (1등 정답률, MEMORY_MIN_SCORE 문턱별 결과).  담당: 문태현

평가셋은 data/eval/recall_set.csv 다.
- kind=record   : 기억에 넣을 지난 대화. text 는 그때 사용자 질문, reply 는 자비서 대답이다.
                  그래프가 넣는 것과 같은 모양("질문: … / 대답: …", memory/dialog.py)으로 넣는다
- kind=scene    : (#65 장면 설명 방식) text 를 그대로 넣는다. 옛 평가셋을 --set 으로 다시 잴 때 쓴다
- kind=question : 기억을 묻는 질문. answer 는 맞는 기록 id(둘 다 맞으면 ;), 비었으면 기록에
                  없는 것을 묻는 질문이다. keyword 는 근거 있는 대답에 들어가야 할 말(| 는 또는)

기록을 실제 임베딩(config.EMBED_MODEL, 키 필요)으로 SQLite 임시 DB 에 넣고, 질문마다 모든
기록의 점수를 본다. 그래프는 상위 5개 중 문턱을 넘는 것만 VLM 에 넘긴다.
- 1등 정답률, 5등 안 정답률 (answer 가 있는 질문)
- 기록에 답이 있음: 맞는 기록 글에 keyword 가 들어 있는 질문 수. 없으면 기록을 찾아도
  VLM 이 답할 수 없다(대답에 위치가 빠진 "어디 있었지?" 같은 것)
- 문턱별 정답 유지: 맞는 기록이 5등 안에 있고 점수가 문턱 이상인 질문 수
- 문턱별 빈 결과: 기록에 없는 질문 중 문턱을 넘는 기록이 하나도 없는 질문 수와,
  그런 질문에 VLM 으로 넘어가는 (상관없는) 기록 수의 평균
- 라우터: 질문이 실제로 RECALL 로 가는지 (graph/router.py 의 낱말 규칙)

실행:
    python scripts/eval_recall.py
    python scripts/eval_recall.py --set data/eval/recall_set.csv
"""

import argparse
import csv
import tempfile
from pathlib import Path

from jarviseo import config
from jarviseo.graph.router import route
from jarviseo.memory import MemoryStore, TextEmbedder, VectorMemory
from jarviseo.memory.dialog import dialog_text
from jarviseo.types import AssistantResponse, Intent, Utterance

TOP_K = 5
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def load(path: Path) -> tuple[dict[str, str], list[dict]]:
    """(기록 id → 기억에 넣을 글, 질문 목록)."""
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    records = {}
    for r in rows:
        if r["kind"] == "record":
            records[r["id"]] = dialog_text(r["text"], r["reply"])
        elif r["kind"] == "scene":
            records[r["id"]] = r["text"]
    return records, [r for r in rows if r["kind"] == "question"]


def fresh_memory(folder: str) -> tuple[VectorMemory, int, int]:
    store = MemoryStore(f"sqlite:///{folder}/recall.db")
    store.init_schema()
    user_id = store.ensure_user()
    session_id = store.start_session(user_id)
    utterance = Utterance(text="평가", started_at=0.0, ended_at=1.0)
    response = AssistantResponse(text="", intent=Intent.RECALL)
    turn_id = store.log_turn(session_id, utterance, response)
    # 문턱은 아래에서 따로 본다. 검색은 점수를 다 받는다.
    return VectorMemory(store, TextEmbedder(), min_score=-1.0), user_id, turn_id


def has_keyword(text: str, keyword: str) -> bool:
    return any(word in text for word in keyword.split("|"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", type=Path, default=Path("data/eval/recall_set.csv"))
    args = parser.parse_args()

    records, questions = load(args.set)
    with tempfile.TemporaryDirectory() as folder:
        memory, user_id, turn_id = fresh_memory(folder)
        record_of = {}
        for record_id, text in records.items():
            memory_id = memory.remember_observation(user_id, turn_id, text)
            record_of[str(memory_id)] = record_id
        results = []
        for q in questions:
            hits = memory.search(user_id, q["text"], top_k=len(records))
            ranked = [(record_of[h.memory_id], h.score) for h in hits]  # 점수 높은 순
            answers = set(filter(None, q["answer"].split(";")))
            rank = next((i for i, (rid, _) in enumerate(ranked, 1) if rid in answers), None)
            results.append(
                {
                    **q,
                    "answers": answers,
                    "rank": rank,
                    "score": ranked[rank - 1][1] if rank else None,
                    "top": ranked[0],
                    "top_scores": [score for _, score in ranked[:TOP_K]],
                    "recall": route(q["text"]) is Intent.RECALL,
                    "grounded": any(has_keyword(records[a], q["keyword"]) for a in answers),
                }
            )
        memory.store.engine.dispose()  # 윈도우는 열린 DB 파일을 못 지운다

    print(f"임베딩 {config.EMBED_MODEL}, 기록 {len(records)}개, 질문 {len(questions)}개\n")
    print("질문 → 1등 기록(점수) / 맞는 기록 순위·점수")
    for r in results:
        top_id, top_score = r["top"]
        tail = f"{r['rank']}등 {r['score']:.3f}" if r["answers"] else "기록 없음"
        if r["answers"] and not r["grounded"]:
            tail += f"  [기록에 '{r['keyword']}' 없음]"
        route_note = "" if r["recall"] else "  [라우터: GENERAL]"
        print(f"  {r['id']} {r['text']} → {top_id}({top_score:.3f}) / {tail}{route_note}")

    answerable = [r for r in results if r["answers"]]
    unanswerable = [r for r in results if not r["answers"]]
    top1 = sum(r["rank"] == 1 for r in answerable)
    in_top = [r for r in answerable if r["rank"] and r["rank"] <= TOP_K]
    grounded = [r for r in answerable if r["grounded"]]
    print(f"\n1등 정답 {top1}/{len(answerable)}, {TOP_K}등 안 {len(in_top)}/{len(answerable)}")
    usable = sum(r["rank"] and r["rank"] <= TOP_K for r in grounded)
    print(
        f"기록에 답이 있음 {len(grounded)}/{len(answerable)}, "
        f"그중 {TOP_K}등 안 {usable}/{len(grounded)}"
    )
    print(f"{TOP_K}등 안 점수 {min(r['score'] for r in in_top):.3f}~", end="")
    print(f"{max(r['score'] for r in in_top):.3f}")
    right = [r["score"] for r in answerable if r["rank"] == 1]
    wrong_top = [r["top"][1] for r in answerable if r["rank"] != 1]
    none_top = [r["top"][1] for r in unanswerable]
    if right:
        print(f"맞는 1등 점수 {min(right):.3f}~{max(right):.3f}", end="")
    if wrong_top:
        print(f", 틀린 1등 점수 최고 {max(wrong_top):.3f}", end="")
    if none_top:
        print(f", 기록 없는 질문의 1등 {min(none_top):.3f}~{max(none_top):.3f}")
    missed_route = [r["id"] for r in results if not r["recall"]]
    print(f"라우터가 RECALL 로 안 보낸 질문 {len(missed_route)}개 {missed_route}")

    print(f"\n{'문턱':>5} {'정답 유지':>9} {'기록 없음: 빈 결과':>14} {'넘긴 기록':>8}")
    for t in THRESHOLDS:
        kept = sum(r["score"] >= t for r in in_top)
        empty = sum(r["top"][1] < t for r in unanswerable)
        passed = sum(sum(s >= t for s in r["top_scores"]) for r in unanswerable)
        mean = passed / max(len(unanswerable), 1)
        print(
            f"{t:5.2f} {kept:5}/{len(answerable):<3} {empty:10}/{len(unanswerable):<3} {mean:9.1f}"
        )


if __name__ == "__main__":
    main()
