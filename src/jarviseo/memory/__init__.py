"""③④ 소지품 재인식 · 개인 기억 검색 — 담당: C.

"만능 비서"의 근거가 되는 부분이다. 사용자의 정보를 기억해뒀다가
답변에 반영한다. 알레르기 정보도 여기 저장된 프로필에서 나온다.

두 가지를 함께 쓴다.
- SQLite (store.py)  — 정형 로그. 대화 기록, 지연시간, 판정 결과.
- Chroma (vector.py) — 벡터 검색. "아까 본 그거" 같은 애매한 질의용.
"""

from jarviseo.memory.store import MemoryStore
from jarviseo.memory.vector import VectorMemory

__all__ = ["MemoryStore", "VectorMemory"]
