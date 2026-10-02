"""다중 단서 융합 — '저거'를 최종 결정한다.  담당: 최홍묵

책임 지표: 대상 선택 정확도

이 파일이 프로젝트의 기술적 중심이다.
cues.py 가 매긴 단서별 점수를 합쳐서 후보 순위를 정하고,
1위와 2위의 점수차가 충분하지 않으면 단정하지 않고 되묻는다.

되묻기가 왜 중요한가
--------------------
착용형 비서에서 틀린 단정은 침묵보다 나쁘다. 사용자가 확인할 화면이 없기
때문에, 잘못 답해도 그게 잘못인지 알 방법이 없다.
그리고 되묻는 동작 자체가 데모에서 가장 잘 보이는 차별점이다.

지금은 가중치를 사람이 정한다(WEIGHTS). 나중에 이 가중치를 로지스틱 회귀로
학습시키면 정형 데이터 기반 머신러닝이 파이프라인에 들어온다.
그때 학습 데이터는 지시 장면 500장 라벨링 결과를 그대로 쓴다.
"""

from __future__ import annotations

from jarviseo import config
from jarviseo.types import (
    CueKind,
    Detection,
    TargetCandidate,
    TargetResolution,
)

__all__ = ["WEIGHTS", "resolve_target"]

# 단서별 가중치. 합이 1.0 이 되도록 유지한다.
# 이 값이 곧 "무엇을 얼마나 믿는가"이고, ablation 의 비교 대상이다.
WEIGHTS: dict[CueKind, float] = {
    CueKind.CENTER: 0.10,
    CueKind.POINT: 0.40,
    CueKind.GAZE: 0.15,
    CueKind.LANG: 0.25,
    CueKind.CTX: 0.10,
}


def resolve_target(
    candidates: list[Detection],
    fingertip: Detection | None,
    gaze_vector: tuple[float, float] | None,
    text: str,
    frame_size: tuple[int, int],
    prev_target_label: str | None = None,
    enabled_cues: set[CueKind] | None = None,
    margin_threshold: float = config.CLARIFY_MARGIN_THRESHOLD,
) -> TargetResolution:
    """사용자가 가리킨 물체를 결정한다.

    Args:
        candidates: 장면에서 찾은 후보 물체들.
        fingertip: 손끝 검출 결과. 손이 안 보이면 None.
        gaze_vector: 시선 방향. 못 구했으면 None.
        text: 사용자 발화 원문.
        frame_size: (width, height). 상대 위치 계산에 쓴다.
        prev_target_label: 직전 턴에서 확정한 대상의 라벨. 없으면 None.
            CTX 단서("그거 말고 옆에 거")의 근거다.
        enabled_cues: 켤 단서 집합. **ablation 실험의 스위치가 이것이다.**
            None 이면 전부 켠다. {CueKind.CENTER} 하나만 주면 baseline 이 된다.
        margin_threshold: 1위-2위 점수차가 이 값보다 작으면 되묻는다.
            기본값은 ``config.CLARIFY_MARGIN_THRESHOLD`` 다. 숫자를 여기에 따로
            적지 않는 이유는, 두 곳에 있으면 한쪽만 고쳐져 어긋나기 때문이다.

    Returns:
        후보 순위, 선택 결과, 되물을지 여부.

    후보가 0개면 chosen=None, needs_clarify=True 로 반환한다.
    후보가 1개면 margin 은 1.0 으로 둔다(비교 상대가 없으므로 확신).
    """
    raise NotImplementedError


def _build_clarify_question(top: TargetCandidate, second: TargetCandidate) -> str:
    """되물을 문장을 만든다.

    두 후보를 구별할 수 있는 특징을 골라야 한다.
    같은 종류면 위치로("왼쪽 컵이요, 오른쪽 컵이요?"),
    다른 종류면 이름으로("컵이요, 병이요?") 묻는다.
    """
    raise NotImplementedError
