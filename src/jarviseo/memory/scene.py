"""장면 기억 저장.  담당: 문태현

질문할 때 찍은 사진 한 장을 VLM 에 다시 보여 주고 장면 설명("책상 위에 검은 백팩")을 받아
``observation`` 에 임베딩과 함께 넣는다. 나중에 "아까 본 가방"을 물으면 이 설명을 찾는다.

대답 문장을 그대로 넣지 않는 이유: "빨간 컵이에요" 같은 대답에는 어디에 무엇이 같이 있었는지가
빠져서 검색이 잘 안 된다. 그 대신 VLM 을 한 번 더 부르는데, 사용자가 대답을 들은 뒤 다른
스레드에서 하므로 기다리는 시간에는 들어가지 않는다.

    scenes = SceneMemory(VectorMemory(store), vlm.describe)
    scenes.remember(user_id, turn_id, frame)   # 바로 돌아온다
    ...
    scenes.close()                             # 끝낼 때 밀린 저장을 마저 한다
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from jarviseo.memory.vector import VectorMemory
from jarviseo.types import Frame

__all__ = ["SceneMemory"]

log = logging.getLogger(__name__)


class SceneMemory:
    """턴이 끝난 뒤 장면 설명을 만들어 observation 에 넣는 일을 뒤에서 한다."""

    def __init__(self, memory: VectorMemory, describe: Callable[[Frame], str | None]) -> None:
        """describe 는 사진을 받아 장면 설명을 주는 함수다(``VLMClient.describe``)."""
        self.memory = memory
        self.describe = describe
        # 한 줄로 세워 차례로 한다. 턴마다 하나씩이라 밀릴 일이 거의 없다.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene")

    def remember(self, user_id: int, turn_id: int, frame: Frame) -> Future[int | None]:
        """저장을 맡기고 바로 돌아온다. 결과(observation_id, 못 넣었으면 None)는 Future 로 준다."""
        return self._pool.submit(self._remember, user_id, turn_id, frame)

    def close(self) -> None:
        """밀린 저장을 다 끝내고 닫는다. 프로그램을 끝낼 때 부른다."""
        self._pool.shutdown(wait=True)

    def _remember(self, user_id: int, turn_id: int, frame: Frame) -> int | None:
        try:
            scene = self.describe(frame)
            if not scene:
                return None
            return self.memory.remember_observation(user_id, turn_id, scene)
        except Exception:
            # 다른 스레드라 예외가 Future 안에 묻힌다. 기록을 남기고 음성 루프는 계속 돈다.
            log.exception("장면 기억 저장 실패 (turn_id %s)", turn_id)
            return None
