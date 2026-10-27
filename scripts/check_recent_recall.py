"""지난 대화를 묻는 기억 질문("방금 물어봤던 거 뭐지?")을 실제 임베딩 · VLM 으로 본다.  담당: 문태현

임시 SQLite 에 오늘 한 질문 11턴(마지막 셋은 10분 · 5분 · 1분 전)을 실제 흐름처럼 남긴다.
기억에는 그래프 규칙대로 넣는다(성분을 확인할 수 없다는 안내 턴은 안 들어간다). 그다음 기억 질문마다
    전: 뜻으로만 찾은 기록(최근 턴 붙이기 전과 같다)
    후: VectorMemory.search(최근 턴을 시간순으로 붙인다)
를 VLM 에 넘겨 대답을 나란히 찍는다. 사진 없이 글자만 보낸다. OpenAI 키가 필요하다(VLM 10번 남짓).

실행:
    python scripts/check_recent_recall.py
"""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jarviseo.memory import MemoryStore, TextEmbedder, VectorMemory
from jarviseo.memory.dialog import DialogMemory
from jarviseo.memory.models import Observation, SessionTurn
from jarviseo.types import AssistantResponse, Intent, Utterance
from jarviseo.vlm import VLMClient
from jarviseo.workflow.messages import CANNED, INGREDIENT_UNAVAILABLE
from jarviseo.workflow.router import classify_intent

# (질문, 대답, 몇 분 전). 앞의 것들은 오늘 앞서 한 대화다. 서버처럼 기록이 쌓여 있어야
# 뜻으로만 찾을 때 직전 대화가 밀려난다. 마지막 3개가 "방금"의 답이다.
TURNS = [
    ("이거 뭐야?", "초록색 물병이에요.", 300),
    ("이 책 제목 뭐야?", "데미안이라고 적혀 있어요.", 280),
    ("이거 뭐야?", "무선 이어폰 충전 케이스예요.", 240),
    ("저기 있는 거 뭐야?", "벽에 걸린 시계예요. 3시 10분을 가리키고 있어요.", 200),
    ("이 과자 이름 뭐야?", "새우깡이에요.", 160),
    ("이거 무슨 색이야?", "파란 우산이에요.", 120),
    ("이 약 뭐야?", "타이레놀 500mg 이에요.", 90),
    ("왼쪽에 있는 거 뭐야?", "노트북 충전기예요.", 60),
    ("이거 뭐야?", "검은색 무선 마우스예요.", 10),
    ("이거 먹어도 돼?", INGREDIENT_UNAVAILABLE, 5),
    ("이 컵 무슨 색이야?", "빨간 머그컵이에요.", 1),
]
QUESTIONS = [
    "방금 물어봤던거 뭐지?",
    "아까 내가 뭐라고 물어봤지?",
    "마지막으로 물어본 게 뭐였지?",
    "방금 뭐라고 했지?",
    "방금 그거 뭐였지?",
    "아까 본 마우스 무슨 색이었지?",  # 물건을 물음 — 최근 턴을 안 붙인다
]


def main() -> None:
    # 파일로 둔다. sqlite:// 는 연결마다 다른 DB 라 dialog 스레드가 넣은 기록이 안 보인다
    tmp = tempfile.TemporaryDirectory()
    store = MemoryStore(database_url=f"sqlite:///{Path(tmp.name) / 'recall.db'}")
    store.init_schema()
    memory = VectorMemory(store, TextEmbedder())
    dialog = DialogMemory(memory)
    user_id = store.ensure_user()
    session_id = store.start_session(user_id)
    for question, answer, minutes in TURNS:
        utterance = Utterance(text=question, started_at=0.0, ended_at=1.0)
        response = AssistantResponse(text=answer, intent=Intent.GENERAL)
        turn_id = store.log_turn(session_id, utterance, response)
        # 미리 정한 안내는 기억에 안 넣는다(workflow/nodes/memory.py 와 같은 규칙)
        observation_id = None
        if answer not in CANNED:
            observation_id = dialog.remember(user_id, turn_id, question, answer).result()
        when = datetime.now(UTC) - timedelta(minutes=minutes)
        with store.session() as db:
            db.get(SessionTurn, turn_id).asked_at = when
            if observation_id is not None:
                db.get(Observation, observation_id).observed_at = when
        print(f"턴 {minutes:>2}분 전  기억 {'O' if observation_id else 'X'}  {question} → {answer}")
    dialog.close()

    vlm = VLMClient()
    for question in QUESTIONS:
        before = memory._similar(user_id, question, 5, None)
        after = memory.search(user_id, question)
        print(f"\n질문: {question}  (라우터 {classify_intent(question).name})")
        for label, hits in (("전", before), ("후", after)):
            kinds = ", ".join(f"{h.kind}:{h.text.splitlines()[0][4:]}" for h in hits) or "없음"
            print(f"  {label} 기록 {len(hits)}개 [{kinds}]")
            print(f"     대답: {vlm.ask(None, question, hits)}")
    store.engine.dispose()
    tmp.cleanup()


if __name__ == "__main__":
    main()
