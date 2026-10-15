"""대화 기억 저장.  담당: 문태현

턴이 끝나면 사용자의 질문과 자비서의 대답을 한 기록("질문: … / 대답: …")으로 묶어
``observation`` 에 임베딩과 함께 넣는다. 나중에 "아까 그 컵 뭐였지?", "아까 뭐라고 했지?"처럼
지난 대화를 물으면 이 기록을 찾아 VLM 에 근거로 넘긴다.

전에는 질문할 때 찍은 사진을 VLM 에 한 번 더 보여 장면 설명을 받아 넣었다(턴마다 VLM 2번).
10/15 팀 방향에 맞춰 대화 기록을 넣는다. VLM 을 더 부르지 않고 임베딩 한 번만 한다.
그 대신 대답에 위치가 없으면("카드지갑이에요") "어디 있었지?"에는 기억에 없다고 답한다.

임베딩 왕복(약 0.15초)이 다음 질문을 막지 않게 다른 스레드에서 한다.

    dialog = DialogMemory(VectorMemory(store))
    dialog.remember(user_id, turn_id, "이거 뭐야?", "빨간 컵이에요.")   # 바로 돌아온다
    ...
    dialog.close()                                                   # 끝낼 때 밀린 저장을 마저 한다
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor

from jarviseo.memory.vector import VectorMemory

__all__ = ["DialogMemory", "dialog_text"]

log = logging.getLogger(__name__)


def dialog_text(question: str, answer: str) -> str:
    """기억에 넣을 글. 질문 낱말과 대답 낱말이 둘 다 검색에 걸리게 같이 적는다."""
    return f"질문: {question}\n대답: {answer}"


class DialogMemory:
    """턴이 끝난 뒤 질문·대답을 observation 에 넣는 일을 뒤에서 한다."""

    def __init__(self, memory: VectorMemory) -> None:
        self.memory = memory
        # 한 줄로 세워 차례로 한다. 턴마다 하나씩이라 밀릴 일이 거의 없다.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dialog")

    def remember(
        self, user_id: int, turn_id: int, question: str, answer: str
    ) -> Future[int | None]:
        """저장을 맡기고 바로 돌아온다. 결과(observation_id, 못 넣었으면 None)는 Future 로 준다."""
        return self._pool.submit(self._remember, user_id, turn_id, question, answer)

    def close(self) -> None:
        """밀린 저장을 다 끝내고 닫는다. 프로그램을 끝낼 때 부른다."""
        self._pool.shutdown(wait=True)

    def _remember(self, user_id: int, turn_id: int, question: str, answer: str) -> int | None:
        if not question.strip() or not answer.strip():
            return None
        try:
            text = dialog_text(question, answer)
            return self.memory.remember_observation(user_id, turn_id, text)
        except Exception:
            # 다른 스레드라 예외가 Future 안에 묻힌다. 기록을 남기고 음성 루프는 계속 돈다.
            log.exception("대화 기억 저장 실패 (turn_id %s)", turn_id)
            return None
