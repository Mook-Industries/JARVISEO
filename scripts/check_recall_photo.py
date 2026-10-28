"""기억 질문에 그때 사진을 같이 보내면 대답에 없던 겉모습도 답하는지 확인한다 (#260).  담당: 문태현

노트북(assistant.py)에서 실제로 물었던 "이거 뭐야?" 턴 5개를 임시 SQLite 에 다시 남기고
(대답 글자는 그대로, 색은 말하지 않았다) 실제 임베딩으로 대화 기억에 넣는다.
그다음 "아까 본 … 무슨 색이었지?"를 물어, 기록 글자만 보낼 때(recall_photo=False)와
1등 기록의 크롭을 같이 보낼 때(True)의 대답과 VLM 왕복을 찍는다.
지금 장면은 아무것도 없는 회색 사진이다(지금 사진으로 답하지 않는지 본다).
대조로 기록에 없는 물건(지갑)과 겉모습이 아닌 질문도 묻는다.

사진은 data/captures/session_*/ 에 있다(git 에 없다). OpenAI 키(.env)가 필요하다.

실행 (레포 맨 위 폴더에서):
    python scripts/check_recall_photo.py
    python scripts/check_recall_photo.py --repeat 3
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

# (그때 원본 사진, 그때 대답, 지금 묻는 말, 사진으로 본 답)
TURNS = [
    (
        "data/captures/session_8/03_turn55.jpg",
        "선풍기인 것 같습니다. 앞에 둥근 안전망이 보입니다.",
        "아까 본 선풍기 무슨 색이었지?",
        "흰색",
    ),
    (
        "data/captures/session_8/05_turn56.jpg",
        "옷이 여러 벌 걸려 있는 의자처럼 보입니다.",
        "아까 의자에 걸려 있던 옷 무슨 색이었지?",
        "흰색",
    ),
    (
        "data/captures/session_13/04_turn83.jpg",
        "싱크대가 아니라, 충전 케이스에 들어 있는 무선 이어폰으로 보입니다.",
        "아까 본 이어폰 케이스 무슨 색이었지?",
        "흰색",
    ),
    (
        "data/captures/session_13/05_turn84.jpg",
        "칫솔은 아니고, 여러 케이블을 연결하는 USB 허브로 보입니다.",
        "아까 본 USB 허브 무슨 색이었지?",
        "회색",
    ),
    (
        "data/captures/session_13/06_turn85.jpg",
        "노트북 컴퓨터입니다.",
        "아까 본 노트북 무슨 색이었지?",
        "짙은 회색",
    ),
]
# 대조: 기록에 없는 물건(지어내면 안 됨)과 겉모습이 아닌 질문(사진이 있어도 같은 답)
CONTROLS = [
    ("아까 본 지갑 무슨 색이었지?", "기억에 없음"),
    ("아까 본 노트북이 뭐였지?", "노트북 컴퓨터"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeat", type=int, default=1, help="질문마다 몇 번 물을지")
    args = ap.parse_args()

    from jarviseo.memory import DialogMemory, MemoryStore, VectorMemory
    from jarviseo.types import AssistantResponse, Frame, Intent, Utterance
    from jarviseo.vlm import VLMClient
    from jarviseo.vlm.client import memory_photo
    from jarviseo.workflow.router import classify_intent

    missing = [path for path, *_ in TURNS if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"사진이 없습니다: {missing}")
    db = Path(tempfile.mkdtemp()) / "check_recall_photo.db"
    store = MemoryStore(database_url=f"sqlite:///{db.as_posix()}")
    store.init_schema()
    user_id = store.ensure_user()
    session = store.start_session(user_id, title="check-recall-photo")
    memory = VectorMemory(store)
    dialog = DialogMemory(memory)
    for path, answer, *_ in TURNS:
        now = time.monotonic()
        utterance = Utterance("이거 뭐야?", now - 1.0, now)
        turn_id = store.log_turn(session, utterance, AssistantResponse(answer, Intent.POINTING))
        store.set_image_path(turn_id, path)
        dialog.remember(user_id, turn_id, "이거 뭐야?", answer)
    dialog.close()  # 임베딩 · 저장이 끝날 때까지 기다린다

    blank = Frame(image=np.full((720, 1280, 3), 128, np.uint8), timestamp=0.0, source_id="check")
    clients = {"글자만": VLMClient(recall_photo=False), "사진 같이": VLMClient(recall_photo=True)}
    ms: dict[str, list[float]] = {name: [] for name in clients}
    asks = [(question, truth) for *_, question, truth in TURNS] + CONTROLS
    for question, truth in asks:
        hits = memory.search(user_id, question)
        photo = memory_photo(hits)
        picked = photo[0].text.splitlines()[-1] if photo else "없음"
        print(f"\n{question}  (의도 {classify_intent(question).value}, 정답 {truth})")
        print(f"  기억 {len(hits)}개, 사진 보낸 기록: {picked}")
        for name, vlm in clients.items():
            for _ in range(args.repeat):
                answer = vlm.ask(blank, question, hits)
                if vlm.last_latency_ms is not None:
                    ms[name].append(vlm.last_latency_ms)
                print(f"  {name:5} {vlm.last_latency_ms or 0:5.0f}ms  {answer}")
    print("\nVLM 왕복 중앙값")
    for name, values in ms.items():
        if values:
            print(f"  {name}: {statistics.median(values):.0f}ms ({len(values)}번)")


if __name__ == "__main__":
    main()
