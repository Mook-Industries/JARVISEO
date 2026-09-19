"""모듈 간 인터페이스 계약.

이 프로젝트에서 가장 먼저 고정되어야 하는 파일이다.

세 사람이 각자 다른 모듈을 만든다. 주고받는 데이터의 모양이 여기서 정해져
있지 않으면, 각자 잘 만들어놓고 합치는 순간 전부 다시 고쳐야 한다.
그래서 모듈 코드를 쓰기 전에 이 파일부터 합의한다.

공통 약속
---------
- 좌표는 전부 픽셀 단위, 원점은 프레임 왼쪽 위 (0, 0).
- 시각(timestamp)은 전부 ``time.monotonic()`` 기준 float 초.
  ``time.time()`` 을 쓰지 않는 이유는 시스템 시계가 도중에 바뀌면
  프레임 순서가 뒤집힐 수 있기 때문이다.
- 점수(score)는 전부 0.0 ~ 1.0 으로 정규화한다.
- 파일 경로는 전부 ``pathlib.Path``. 문자열 결합 금지.

이 파일을 바꿀 때
-----------------
타입이 바뀌면 남의 코드가 조용히 깨진다. 필드를 추가/삭제/개명할 때는
반드시 팀 채널에 공지하고, 커밋 타입 뒤에 ``!`` 를 붙인다.
예: ``feat(types)!: Frame 에 sharpness 필드 추가``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BBox",
    "Frame",
    "Detection",
    "Utterance",
    "Intent",
    "CueKind",
    "CueScore",
    "TargetCandidate",
    "TargetResolution",
    "OCRLine",
    "IngredientPanel",
    "AllergenVerdict",
    "AllergenJudgement",
    "MemoryHit",
    "AssistantResponse",
]


# --------------------------------------------------------------------------
# 기본 단위
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BBox:
    """픽셀 좌표계 사각형. 왼쪽 위 (x1, y1) ~ 오른쪽 아래 (x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def iou(self, other: BBox) -> float:
        """두 박스가 얼마나 겹치는지 0.0~1.0 으로 반환한다."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Frame:
    """카메라에서 읽은 프레임 한 장.

    인수인계서의 ``FrameSource.get_frame() -> np.ndarray`` 스케치를 확장한 것이다.
    ndarray 만 넘기면 "이 사진을 언제 찍었는지"를 알 수 없는데,
    알려진 함정 "질문 시점 프레임 오차"(발화가 끝난 뒤 캡처하면 이미 시선이
    옮겨가 엉뚱한 장면이 잡힘)를 해결하려면 발화 시작 시각과 프레임 시각을
    맞춰봐야 한다. 그래서 timestamp 를 같이 들고 다닌다.
    """

    image: np.ndarray  # BGR, shape (H, W, 3) — OpenCV 기본 채널 순서
    timestamp: float  # time.monotonic() 기준 초
    source_id: str  # "usbcam" / "webcam" / 파일명 등, 로그 추적용
    sharpness: float | None = None  # Laplacian variance. 아직 안 쟀으면 None

    @property
    def size(self) -> tuple[int, int]:
        """(width, height)."""
        h, w = self.image.shape[:2]
        return w, h


@dataclass
class Detection:
    """검출기(YOLO 등)가 찾아낸 물체 하나."""

    label: str  # "fingertip", "ingredient_panel", "cup", ...
    confidence: float  # 0.0 ~ 1.0
    bbox: BBox
    track_id: int | None = None  # 프레임 간 추적을 붙일 경우에만


# --------------------------------------------------------------------------
# 음성 · 의도
# --------------------------------------------------------------------------


@dataclass
class Utterance:
    """STT 가 받아쓴 사용자 발화 한 건.

    started_at 이 핵심이다. 프레임은 발화가 '끝난' 시각이 아니라
    '시작된' 시각을 기준으로 골라야 한다.
    """

    text: str
    started_at: float  # VAD 가 잡은 발화 시작 시각 (monotonic)
    ended_at: float
    confidence: float | None = None


class Intent(StrEnum):
    """라우터가 판단한 질문의 종류.

    GENERAL 은 폴백이다. 특화 모듈이 해당 없다고 판단하면 여기로 떨어지고,
    기본 VLM 응답이 그대로 나간다.
    """

    POINTING = "pointing"  # "저거 뭐야?" — 지시 대상 특정 필요
    INGREDIENT = "ingredient"  # "이거 뭐 들어갔어?" — 성분표 판정 필요
    BELONGING = "belonging"  # "내 가방 어디 있어?" — 소지품 재인식
    RECALL = "recall"  # "아까 본 그거" — 개인 기억 검색
    GENERAL = "general"  # 그 외 전부 — 기본 VLM


# --------------------------------------------------------------------------
# ① 지시 대상 특정  (담당: A · 팀장)
# --------------------------------------------------------------------------


class CueKind(StrEnum):
    """'저거'가 무엇인지 판단할 때 쓰는 단서의 종류.

    ablation(단서를 하나씩 껐다 켜며 기여도를 재는 실험)의 축이 이것이다.
    단서를 추가할 때 여기에 먼저 항목을 넣는다.
    """

    HAND = "hand"  # 손끝에서 뻗은 방향
    GAZE = "gaze"  # 머리/시선 방향
    LANGUAGE = "language"  # "빨간 거", "왼쪽 거" 같은 말 속 단서
    SALIENCE = "salience"  # 화면 중앙·크기 등 눈에 띄는 정도


@dataclass
class CueScore:
    """단서 하나가 후보 물체 하나에 준 점수."""

    kind: CueKind
    score: float  # 0.0 ~ 1.0
    detail: dict[str, Any] = field(default_factory=dict)  # 디버깅·시각화용


@dataclass
class TargetCandidate:
    """'저거'의 후보가 되는 물체 하나와 그 점수."""

    detection: Detection
    cue_scores: list[CueScore] = field(default_factory=list)
    total_score: float = 0.0

    def score_of(self, kind: CueKind) -> float:
        """특정 단서가 준 점수만 꺼낸다. 없으면 0.0 (= 그 단서를 끈 상태)."""
        for cs in self.cue_scores:
            if cs.kind is kind:
                return cs.score
        return 0.0


@dataclass
class TargetResolution:
    """지시 대상 특정의 최종 결과.

    needs_clarify 가 True 면 단정하지 말고 사용자에게 되묻는다.
    "착용형 비서에서 틀린 단정은 침묵보다 나쁘다" — 확정된 설계 결정.
    """

    candidates: list[TargetCandidate]  # total_score 내림차순 정렬
    chosen: TargetCandidate | None  # needs_clarify 면 None 일 수 있음
    margin: float  # 1위 - 2위 점수차. 후보가 1개면 1.0
    needs_clarify: bool
    clarify_question: str | None = None  # "왼쪽 컵이요, 오른쪽 컵이요?"


# --------------------------------------------------------------------------
# ② 식품 성분 판정  (담당: B)
# --------------------------------------------------------------------------


@dataclass
class OCRLine:
    """OCR 이 읽어낸 글자 한 줄."""

    text: str
    confidence: float
    bbox: BBox


@dataclass
class IngredientPanel:
    """검출한 성분표 영역과, 거기서 읽어낸 글자들."""

    panel_bbox: BBox
    lines: list[OCRLine] = field(default_factory=list)
    raw_text: str = ""  # lines 를 이어붙인 원문
    ingredients: list[str] = field(default_factory=list)  # 파싱한 성분명 목록


class AllergenVerdict(StrEnum):
    """알레르기 판정 결과.

    UNCERTAIN 이 핵심이다. 확정된 설계 결정에 따라 정밀도보다 재현율을
    우선하고, 애매하면 SAFE 라고 하지 않고 UNCERTAIN 을 반환한다.
    놓친 알레르겐의 피해가 과잉 경고보다 훨씬 크기 때문이다.
    """

    SAFE = "safe"  # 등록된 알레르겐이 안 보임
    WARN = "warn"  # 알레르겐으로 보이는 성분을 찾음
    UNCERTAIN = "uncertain"  # 글자를 제대로 못 읽었거나 판단 불가


@dataclass
class AllergenJudgement:
    """성분표 판정 결과.

    matched_terms 는 '성분표에 실제로 적혀 있던 말'을 그대로 담는다.
    사용자에게는 "탈지분유가 들어 있어 우유 알레르기에 해당합니다"처럼
    원문과 매핑 결과를 같이 보여줘야 납득이 된다.
    """

    verdict: AllergenVerdict
    matched_terms: list[str] = field(default_factory=list)  # ["탈지분유", "카제인"]
    matched_allergens: list[str] = field(default_factory=list)  # ["우유"]
    reason: str = ""
    panel: IngredientPanel | None = None


# --------------------------------------------------------------------------
# ③④ 소지품 재인식 · 개인 기억  (담당: C)
# --------------------------------------------------------------------------


@dataclass
class MemoryHit:
    """벡터 검색으로 찾아낸 과거 기록 하나."""

    memory_id: str
    kind: str  # "belonging" / "observation" / "profile"
    text: str  # 사람이 읽을 수 있는 설명
    score: float  # 유사도 0.0 ~ 1.0
    observed_at: float | None = None
    image_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 최종 응답
# --------------------------------------------------------------------------


@dataclass
class AssistantResponse:
    """사용자에게 나가는 최종 답변.

    latency_ms 는 대시보드의 단계별 지연 그래프에 그대로 들어간다.
    팀장 책임 지표(응답 지연 p50/p95)의 원천 데이터이므로 단계별로 기록한다.
    """

    text: str  # TTS 로 읽어줄 문장
    intent: Intent
    used_frame: Frame | None = None
    target: TargetResolution | None = None
    allergen: AllergenJudgement | None = None
    memories: list[MemoryHit] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)  # {"stt": 120.0, ...}
    needs_clarify: bool = False
