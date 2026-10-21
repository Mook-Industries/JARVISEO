"""다중 단서 융합 — '저거'를 최종 결정한다.  담당: 최홍묵

책임 지표: 대상 선택 정확도 · 되묻기 비율

cues.py 가 매긴 단서별 점수를 가중치로 합쳐 후보 순위를 정하고,
1위와 2위의 점수차(margin)가 기준보다 작으면 단정하지 않고 되묻는다.
착용형 비서에서 틀린 단정은 침묵보다 나쁘다. 사용자가 확인할 화면이 없다.

되묻기 대답("왼쪽", "컵")은 단서가 아니다. 사용자가 직접 고른 것이라 점수에 섞지 않고
후보를 걸러내는 데만 쓴다(10/15 결정 A안). 섞으면 ablation 의 v1 줄이 baseline 이 아니게 된다.

손 단서(v2)는 가리키는 손이 없으면 기권한다. 0점을 주는 게 아니라 그 사진에서 빼고
켜진 나머지 단서로 다시 나눈다(10/14 요구사항: 손이 없어도 대상을 골라야 한다).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

from jarviseo import config
from jarviseo.pointing.cues import score_center, score_point
from jarviseo.pointing.hands import HandPose, Ray, pointing_ray
from jarviseo.pointing.labels import korean_name
from jarviseo.types import (
    CueKind,
    CueScore,
    Detection,
    TargetCandidate,
    TargetResolution,
)

__all__ = ["IMPLEMENTED_CUES", "WEIGHTS", "resolve_target"]

# 단서별 가중치. 이 값이 곧 "무엇을 얼마나 믿는가"이고, ablation 의 비교 대상이다.
# 켜진 단서의 가중치 합으로 나누므로 몇 개를 켜든 점수는 0~1 이다.
WEIGHTS: dict[CueKind, float] = {
    CueKind.CENTER: 0.10,
    CueKind.POINT: 0.40,
    CueKind.GAZE: 0.15,
    CueKind.LANG: 0.25,
    CueKind.CTX: 0.10,
}

# 단서를 구현하면 여기와 _scorers 에 넣는다.
IMPLEMENTED_CUES = frozenset({CueKind.CENTER, CueKind.POINT})

# 후보 목록만 받아 단서 점수를 돌려주는 함수. 사진 크기 · 손 방향은 _scorers 가 미리 묶는다.
Scorer = Callable[[list[Detection]], list[CueScore]]


def resolve_target(
    candidates: list[Detection],
    hand: HandPose | None,
    gaze_vector: tuple[float, float] | None,
    text: str,
    frame_size: tuple[int, int],
    prev_target_label: str | None = None,
    replies: Sequence[str] = (),
    enabled_cues: set[CueKind] | None = None,
    margin_threshold: float = config.CLARIFY_MARGIN_THRESHOLD,
) -> TargetResolution:
    """사용자가 가리킨 물체를 결정한다.

    Args:
        candidates: 장면에서 찾은 후보 물체들.
        hand: 가리키는 손(hands.pick_pointing_hand 로 고른 것). None 이면 손 단서는 기권한다.
        gaze_vector: 시선 단서(v3) 자리. v1 은 쓰지 않는다.
        text: 처음 질문 원문. 되묻기 대답은 넣지 않는다(``replies``).
        frame_size: (width, height).
        prev_target_label: 대화 맥락 단서(v5) 자리. v1 은 쓰지 않는다.
        replies: 되묻기 대답들, 받은 순서대로. 후보를 걸러내는 데만 쓴다.
        enabled_cues: 켤 단서. **ablation 의 스위치.** None 이면 구현된 단서 전부.
            구현 안 된 단서가 있으면 ValueError — 조용히 빼면 v2 줄이 v1 숫자가 된다.
        margin_threshold: 1위-2위 점수차가 이 값보다 작으면 되묻는다.
            기본값은 ``config.CLARIFY_MARGIN_THRESHOLD`` 다(두 곳에 적으면 어긋난다).

    후보가 0개면 chosen=None, needs_clarify=True, 되물을 문장 없음(그래프가 "다시 가리켜
    주세요"로 끝낸다). 켜진 단서가 모두 기권해도(손 단서만 켰는데 손이 없음) 같다.
    후보가 1개면 margin 은 1.0(비교 상대가 없으므로 확신)이다.
    candidates 칸에는 걸러낸 뒤의 순위를 담는다. chosen 이 그 안에 있어야 DB 기록이 맞는다.
    """
    active = IMPLEMENTED_CUES if enabled_cues is None else frozenset(enabled_cues)
    if not active or active - IMPLEMENTED_CUES:
        allowed = ", ".join(k.value for k in IMPLEMENTED_CUES)
        raise ValueError(f"켤 단서가 없거나 구현 안 된 단서가 있다. 켤 수 있는 단서: {allowed}")
    ray = pointing_ray(hand) if hand is not None else None
    if ray is None:
        active -= {CueKind.POINT}  # 기권: 빼고 나머지 가중치로 다시 나눈다
    kinds = [k for k in CueKind if k in active]  # ablation 순서(v1~v5)로 기록한다

    if not candidates or not kinds:
        return TargetResolution([], chosen=None, margin=0.0, needs_clarify=True)

    scorers = _scorers(frame_size, ray)
    narrowed = _narrow_by_replies(candidates, replies, kinds, scorers, frame_size[0])
    ranked = _rank(narrowed, kinds, scorers)
    margin = 1.0 if len(ranked) == 1 else ranked[0].total_score - ranked[1].total_score
    if margin < margin_threshold:
        question = _build_clarify_question(ranked[0], ranked[1])
        return TargetResolution(ranked, None, margin, True, question)
    return TargetResolution(ranked, ranked[0], margin, False)


def _scorers(frame_size: tuple[int, int], ray: Ray | None) -> dict[CueKind, Scorer]:
    """이번 사진에서 쓸 단서별 점수 함수. 손이 없으면(ray None) 손 단서는 넣지 않는다."""
    scorers: dict[CueKind, Scorer] = {CueKind.CENTER: partial(score_center, frame_size=frame_size)}
    if ray is not None:
        scorers[CueKind.POINT] = partial(score_point, ray=ray)
    return scorers


def _rank(
    candidates: list[Detection], kinds: list[CueKind], scorers: dict[CueKind, Scorer]
) -> list[TargetCandidate]:
    """켜진 단서 점수를 가중 평균해 높은 순으로 줄 세운다. 동점이면 들어온 순서를 지킨다."""
    per_cue = {k: scorers[k](candidates) for k in kinds}
    total_weight = sum(WEIGHTS[k] for k in kinds)
    ranked = [
        TargetCandidate(
            detection=d,
            cue_scores=[per_cue[k][i] for k in kinds],
            total_score=sum(WEIGHTS[k] * per_cue[k][i].score for k in kinds) / total_weight,
        )
        for i, d in enumerate(candidates)
    ]
    return sorted(ranked, key=lambda c: c.total_score, reverse=True)


def _narrow_by_replies(
    candidates: list[Detection],
    replies: Sequence[str],
    kinds: list[CueKind],
    scorers: dict[CueKind, Scorer],
    width: int,
) -> list[Detection]:
    """되묻기 대답으로 후보를 걸러낸다. 아무것도 안 맞으면 그 대답은 없던 것으로 한다.

    이름("컵")은 남은 후보 전부에서 찾는다. 사용자가 이름을 댔으면 그것이 답이다.
    위치("왼쪽")는 그 대답을 부른 질문의 두 후보(그때의 1·2위) 안에서만 고른다.
    후보 전체에서 고르면, 컵 셋 중 가운데·오른쪽을 물었는데 맨 왼쪽 컵이 뽑힌다.
    같은 후보면 순위가 늘 같으므로, 대답마다 순위를 다시 매기면 그때 물은 두 후보가 나온다.
    """
    for reply in replies:
        said = reply.replace(" ", "")
        candidates = _by_name(candidates, said) or candidates
        if len(candidates) > 1:
            asked = [c.detection for c in _rank(candidates, kinds, scorers)[:2]]
            candidates = _by_position(asked, said, width) or candidates
    return candidates


def _by_name(candidates: list[Detection], said: str) -> list[Detection]:
    """대답에 든 후보 이름과 같은 후보만. "곰인형"처럼 여러 이름이 걸리면 가장 긴 이름."""
    names = {korean_name(d.label).replace(" ", "") for d in candidates}
    hits = [name for name in names if name in said]
    if not hits:
        return []
    best = max(hits, key=len)
    return [d for d in candidates if korean_name(d.label).replace(" ", "") == best]


def _by_position(candidates: list[Detection], said: str, width: int) -> list[Detection]:
    """대답에 위치 말이 있으면 그 자리의 후보 하나. 없으면 빈 리스트."""
    if "왼쪽" in said:
        return [min(candidates, key=lambda d: d.bbox.center[0])]
    if "오른쪽" in said:
        return [max(candidates, key=lambda d: d.bbox.center[0])]
    if "가운데" in said:
        return [min(candidates, key=lambda d: abs(d.bbox.center[0] - width / 2))]
    return []


def _build_clarify_question(top: TargetCandidate, second: TargetCandidate) -> str:
    """되물을 문장. 같은 종류면 위치로("왼쪽 컵이요, 오른쪽 컵이요?"), 다르면 이름으로 묻는다.

    1·2위만 묻는다. 셋을 다 읽으면 문장이 길어져 대답이 오히려 흐려진다.
    """
    first, other = korean_name(top.detection.label), korean_name(second.detection.label)
    if first == other:
        return f"왼쪽 {first}{_yo(first)}, 오른쪽 {first}{_yo(first)}?"
    return f"{first}{_yo(first)}, {other}{_yo(other)}?"


def _yo(name: str) -> str:
    """받침이 있으면 "이요"(컵이요), 없으면 "요"(사과요). 한글이 아니면 "요"."""
    last = name[-1]
    if "가" <= last <= "힣" and (ord(last) - ord("가")) % 28:
        return "이요"
    return "요"
