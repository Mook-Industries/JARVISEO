"""③④ 소지품 재인식 · 개인 기억 검색 — 담당: 문태현.

"만능 비서"의 근거가 되는 부분이다. 사용자의 정보를 기억해뒀다가
답변에 반영한다. 알레르기 정보도 여기 저장된 프로필에서 나온다.

저장소를 둘로 나눈 이유
----------------------
- **Postgres** (models.py, store.py) — 정형 기록.
  대화 로그, 단계별 지연, 판정 결과처럼 "몇 번째 턴의 무엇"이 분명한 것들.
  docker compose 로 띄우므로 팀원이 각자 설치할 필요가 없다.

- **Chroma** (vector.py) — 벡터 검색.
  "아까 본 그거"처럼 정확한 이름을 모르는 질의용. 서버로 띄우지 않고
  파일 모드로 쓴다. 한 프로세스만 읽고 쓰기 때문이다.

임베딩은 Chroma 에만 둔다. Postgres 에는 ``chroma_id`` 만 남긴다.
같은 벡터를 두 군데 두면 어긋났을 때 어느 쪽이 맞는지 알 수 없다.
"""

from jarviseo.memory.models import (
    Allergen,
    Base,
    ChatSession,
    EvalRun,
    EvalSample,
    IngredientSynonym,
    Product,
    SessionTurn,
    TurnCandidate,
    TurnInference,
    TurnIngredient,
    User,
    UserAllergen,
    UserSetting,
)
from jarviseo.memory.store import MemoryStore
from jarviseo.memory.vector import VectorMemory

__all__ = [
    "MemoryStore",
    "VectorMemory",
    "Base",
    "User",
    "UserSetting",
    "Allergen",
    "IngredientSynonym",
    "UserAllergen",
    "ChatSession",
    "SessionTurn",
    "TurnInference",
    "TurnCandidate",
    "Product",
    "TurnIngredient",
    "EvalRun",
    "EvalSample",
]
