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
- 라우터: 질문이 실제로 RECALL 로 가는지 (workflow/router.py 의 낱말 규칙)

--answer 를 주면 RAG 응답 근거율도 잰다(#117). 질문마다 그래프와 같은 기록(상위 5개 중
MEMORY_MIN_SCORE 이상)을 실제 VLM(config.VLM_MODEL, 키 필요)에 넣고 대답을 받는다.
사진은 보내지 않는다(frame=None). 기억만 보고 답하는지 보려는 것이다. 대답을 셋으로 나눈다.
- 근거 있음   : 맞는 기록에 답(keyword)이 있고 대답에도 그 말이 있다
- 모른다고 함 : "기억에 없다"처럼 모른다고 답했다. 기록에 답이 없는 질문(기록 없는 질문,
                대답에 위치가 빠진 "어디 있었지?")에서는 이것이 맞는 대답이다
- 확인 필요   : 둘 다 아니다. 지어냈을 수 있어 사람이 대답을 읽고 판정한다
대답은 csv(--out)로 남긴다. 사람 판정 칸에 근거/모름/지어냄을 적어 최종 근거율을 낸다.

실행:
    python scripts/eval_recall.py
    python scripts/eval_recall.py --set data/eval/recall_set.csv
    python scripts/eval_recall.py --answer                 # VLM 30번 호출
"""

import argparse
import csv
import tempfile
from pathlib import Path

from jarviseo import config
from jarviseo.memory import MemoryStore, TextEmbedder, VectorMemory
from jarviseo.memory.dialog import dialog_text
from jarviseo.types import AssistantResponse, Intent, Utterance
from jarviseo.workflow.router import classify_intent

TOP_K = 5
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
# 모른다고 답한 것으로 보는 말. VLM 지시문은 "기록에 없는 내용은 기억에 없다고 말한다"이다.
UNKNOWN = ("기억에 없", "기록에 없", "기억나지 않", "기억이 나지 않", "모르", "알 수 없")
GROUNDED, UNKNOWN_SAID, CHECK = "근거 있음", "모른다고 함", "확인 필요"
ANSWER_FIELDS = [
    "id",
    "text",
    "answer",
    "keyword",
    "기록에 답",
    "넘긴 기록",
    "대답",
    "판정",
    "사람 판정",
]


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
    return bool(keyword) and any(word in text for word in keyword.split("|"))


def judge(reply: str, keyword: str, grounded: bool) -> str:
    """대답 하나를 근거 있음 / 모른다고 함 / 확인 필요 로 나눈다.

    grounded 는 맞는 기록에 답(keyword)이 있는지다. 기록에 답이 없으면 keyword 가 대답에
    나와도 근거가 아니다(기록에 없는 위치를 지어냈을 수 있어서 확인 필요로 둔다).
    """
    if grounded and has_keyword(reply, keyword):
        return GROUNDED
    if any(word in reply for word in UNKNOWN):
        return UNKNOWN_SAID
    return CHECK


def answer_eval(results: list[dict], out: Path) -> None:
    """질문마다 그래프와 같은 기록을 실제 VLM 에 넣고 대답을 나눠 센다. 대답은 out 에 남긴다."""
    from jarviseo.vlm import VLMClient  # 키가 필요한 것은 --answer 일 때만 만든다

    vlm = VLMClient()
    print(f"\n대답 근거율 (VLM {vlm.model}, 사진 없이, 기록 {TOP_K}개 중 문턱 이상)")
    rows = []
    for r in results:
        reply = vlm.ask(None, r["text"], r["sent"]) or ""
        verdict = judge(reply, r["keyword"], r["grounded"])
        rows.append(
            {
                "id": r["id"],
                "text": r["text"],
                "answer": r["answer"],
                "keyword": r["keyword"],
                "기록에 답": "O" if r["grounded"] else "X",
                "넘긴 기록": len(r["sent"]),
                "대답": reply,
                "판정": verdict,
                "사람 판정": "",
            }
        )
        print(f"  {r['id']} [{verdict}] {r['text']} → {reply}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:  # 엑셀에서 한글이 안 깨지게
        writer = csv.DictWriter(f, ANSWER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    groups = {
        "기록에 답이 있는 질문": [x for x in rows if x["기록에 답"] == "O"],
        "기록은 있지만 답(위치 등)이 없는 질문": [
            x for x in rows if x["answer"] and x["기록에 답"] == "X"
        ],
        "기록에 없는 질문": [x for x in rows if not x["answer"]],
    }
    print()
    for name, xs in groups.items():
        counts = {v: sum(x["판정"] == v for x in xs) for v in (GROUNDED, UNKNOWN_SAID, CHECK)}
        print(f"  {name} {len(xs)}개: " + ", ".join(f"{k} {n}" for k, n in counts.items()))
    honest = sum(x["판정"] == (GROUNDED if x["기록에 답"] == "O" else UNKNOWN_SAID) for x in rows)
    check = [x["id"] for x in rows if x["판정"] == CHECK]
    print(f"자동 판정 근거율 {honest}/{len(rows)} (근거로 답했거나, 답이 없을 때 모른다고 함)")
    print(f"확인 필요 {len(check)}개 {check} → {out} 의 사람 판정 칸에 근거/모름/지어냄")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", type=Path, default=Path("data/eval/recall_set.csv"))
    parser.add_argument("--answer", action="store_true", help="실제 VLM 대답으로 근거율도 잰다")
    parser.add_argument("--out", type=Path, default=Path("data/eval/recall_answers.csv"))
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
                    "recall": classify_intent(q["text"]) is Intent.RECALL,
                    "grounded": any(has_keyword(records[a], q["keyword"]) for a in answers),
                    # 그래프가 VLM 에 넘기는 기록: 상위 TOP_K 개 중 문턱 이상
                    "sent": [h for h in hits[:TOP_K] if h.score >= config.MEMORY_MIN_SCORE],
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

    if args.answer:
        answer_eval(results, args.out)


if __name__ == "__main__":
    main()
