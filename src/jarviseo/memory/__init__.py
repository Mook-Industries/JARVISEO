"""③④ 소지품 재인식 · 개인 기억 검색 — 담당: 문태현.

"만능 비서"의 근거가 되는 부분이다. 사용자의 정보를 기억해뒀다가
답변에 반영한다. 알레르기 정보도 여기 저장된 프로필에서 나온다.

저장소를 둘로 나눈 이유
----------------------
- **Postgres** (models.py, store.py) — 정형 기록.
  대화 로그, 단계별 지연, 판정 결과처럼 "몇 번째 턴의 무엇"이 분명한 것들.
  docker compose 로 띄우므로 팀원이 각자 설치할 필요가 없다.

- **pgvector** (vector.py) — 벡터 검색.
  "아까 본 그거"처럼 정확한 이름을 모르는 질의용. 같은 Postgres 의 확장이라
  DB 를 하나 더 띄우지 않는다.

임베딩은 그 기록과 같은 행에 둔다. 벡터 DB 를 따로 두면 두 곳이 어긋났을 때
어느 쪽이 맞는지 알 수 없다.
"""

from jarviseo.memory.models import (
    Allergen,
    Base,
    Belonging,
    BelongingImage,
    ChatSession,
    EvalRun,
    EvalSample,
    IngredientSynonym,
    Observation,
    Product,
    SessionTurn,
    TurnCandidate,
    TurnInference,
    TurnIngredient,
    TurnVoice,
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
    "TurnVoice",
    "TurnInference",
    "TurnCandidate",
    "Product",
    "TurnIngredient",
    "Belonging",
    "BelongingImage",
    "Observation",
    "EvalRun",
    "EvalSample",
]
