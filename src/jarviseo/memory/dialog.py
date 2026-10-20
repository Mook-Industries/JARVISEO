"""대화 기억 저장.  담당: 문태현

턴이 끝나면 사용자의 질문과 자비서의 대답을 한 기록("질문: … / 대답: …")으로 묶어
``observation`` 에 임베딩과 함께 넣는다. 나중에 "아까 그 컵 뭐였지?", "아까 뭐라고 했지?"처럼
지난 대화를 물으면 이 기록을 찾아 VLM 에 근거로 넘긴다.

전에는 질문할 때 찍은 사진을 VLM 에 한 번 더 보여 장면 설명을 받아 넣었다(턴마다 VLM 2번).
10/15 팀 방향에 맞춰 대화 기록을 넣는다. VLM 을 더 부르지 않고 임베딩 한 번만 한다.
그 대신 대답에 위치가 없으면("카드지갑이에요") "어디 있었지?"에는 기억에 없다고 답한다.

아무것도 알아보지 못한 대답("안경을 쓴 분이 보이고, 뒤에는 사물함이 있습니다",
"사진에는 시계가 보이지 않아 알 수 없습니다")은 넣지 않는다(says_nothing). 가리킨 것 없이 물은
턴마다 이런 기록이 쌓여 "아까 본 지갑"에도 점수 0.27~0.28 로 찾히며 맞는 기록을 밀어냈다(#132).

임베딩 왕복(약 0.15초)이 다음 질문을 막지 않게 다른 스레드에서 한다.

    dialog = DialogMemory(VectorMemory(store))
    dialog.remember(user_id, turn_id, "이거 뭐야?", "빨간 컵이에요.")   # 바로 돌아온다
    ...
    dialog.close()                                                   # 끝낼 때 밀린 저장을 마저 한다
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor

from jarviseo.memory.vector import VectorMemory

__all__ = ["DialogMemory", "dialog_text", "says_nothing"]

log = logging.getLogger(__name__)

# 대답을 문장·쉼표로 나눈 조각마다 본다. 아래 셋 중 하나인 조각만 있으면 알아본 것이 없다.
# 모른다는 말. "펜처럼 보이는데, … 모르겠습니다"는 앞 조각이 알아본 것이라 남는다.
# 한 조각 안에 알아본 말이 같이 있으면("열쇠는 보이지 않지만 지갑은 책상 위에 있습니다") 남긴다
_UNKNOWN = re.compile(r"알 수 없|모르겠|알기 어렵|확인하기 어렵|확인할 수 없|보이지 않|안 보|죄송")
_FOUND = re.compile(r"있습니다|있어요|있고|입니다|이에요|예요|처럼 보|로 보|것 같")
# 무엇을 물으시는지 되묻는 말
_ASK_BACK = re.compile(r"알려\s?주시면|말씀해\s?주시면|물으시는지|가리켜\s?주시면|다시 보여")
# 사람만 보인다는 말("안경을 쓴 분이 보이고"). "부분이"는 사람이 아니다.
# 손에 든 것·가리킨 것을 말하면 알아본 것이다
_PERSON = re.compile(r"(사람|(^|\s)분).*보(이|입|여)")
_HOLDING = re.compile(r"들고|쥐고|잡고|가리키")
# 뒤·배경만 말하는 조각("뒤에는 사물함이 있습니다")
_BACKGROUND = re.compile(r"^(그\s)?(뒤|뒤쪽|배경)(에는|에|으로는|로는|엔)\s")
# 사람·뒤를 직접 물었으면 그 대답이 곧 답이다("사람 뒤에 뭐가 있지?")
_ASKS_SCENE = re.compile(r"사람|누구|누가|뒤|배경")
_PIECES = re.compile(r"[.!?]\s*|,\s*")


def dialog_text(question: str, answer: str) -> str:
    """기억에 넣을 글. 질문 낱말과 대답 낱말이 둘 다 검색에 걸리게 같이 적는다."""
    return f"질문: {question}\n대답: {answer}"


def says_nothing(question: str, answer: str) -> bool:
    """대답이 아무것도 알아보지 못했는지. 그렇다면 기억할 것이 없다.

    모른다는 말, 되묻는 말, (사람·뒤를 묻지 않았는데) 사람이나 뒤만 말한 조각뿐이면 True.
    """
    asks_scene = _ASKS_SCENE.search(question) is not None
    pieces = [p.strip() for p in _PIECES.split(answer) if p.strip()]
    return bool(pieces) and all(_says_nothing(piece, asks_scene) for piece in pieces)


def _says_nothing(piece: str, asks_scene: bool) -> bool:
    if _ASK_BACK.search(piece) or (_UNKNOWN.search(piece) and not _FOUND.search(piece)):
        return True
    if asks_scene:
        return False
    person_only = _PERSON.search(piece) is not None and _HOLDING.search(piece) is None
    return person_only or _BACKGROUND.search(piece) is not None


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
        if says_nothing(question, answer):
            log.debug("알아본 것이 없는 대답이라 기억에 넣지 않는다 (turn_id %s)", turn_id)
            return None
        try:
            text = dialog_text(question, answer)
            return self.memory.remember_observation(user_id, turn_id, text)
        except Exception:
            # 다른 스레드라 예외가 Future 안에 묻힌다. 기록을 남기고 음성 루프는 계속 돈다.
            log.exception("대화 기억 저장 실패 (turn_id %s)", turn_id)
            return None
