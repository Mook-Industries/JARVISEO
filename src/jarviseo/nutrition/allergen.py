"""알레르기 판정.  담당: 권용현

책임 지표: 알레르기 판정 recall

확정된 설계 결정: **정밀도보다 재현율을 우선한다.**
놓친 알레르겐과 과잉 경고는 피해가 비대칭이다. 경고를 한 번 더 하는 것보다
땅콩을 놓치는 쪽이 훨씬 위험하다. 애매하면 SAFE 라고 하지 말고 UNCERTAIN 을 낸다.

이 파일의 정확도를 결정하는 것은 모델이 아니라 동의어 사전이다
-----------------------------------------------------------------
'우유'를 찾는데 성분표에는 '탈지분유', '카제인', '유청'이라고 적혀 있다.
사전에 이 대응이 없으면 못 잡는다. 알려진 함정 중 하나이고,
사전 구축에 충분한 시간을 배정해야 한다.

사전은 코드가 아니라 데이터로 관리한다: data/datasets/allergen_synonyms.json
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from jarviseo.types import AllergenJudgement, IngredientPanel

__all__ = ["load_synonyms", "judge_allergens", "KNOWN_ALLERGENS", "normalize_term"]

# 식약처 표시 대상 알레르기 유발물질을 출발점으로 삼는다.
# 실제 사전은 JSON 파일로 관리하고, 이 목록은 키 이름의 기준일 뿐이다.
KNOWN_ALLERGENS = [
    "우유",
    "알류",
    "메밀",
    "땅콩",
    "대두",
    "밀",
    "고등어",
    "게",
    "새우",
    "돼지고기",
    "복숭아",
    "토마토",
    "아황산류",
    "호두",
    "닭고기",
    "쇠고기",
    "오징어",
    "조개류",
    "잣",
]


def normalize_term(text: str) -> str:
    """성분 표기를 비교 가능한 모양으로 맞춘다. 사전과 성분표 양쪽에 같은 함수를 쓴다.

    - NFC 정규화: 맥에서 만든 파일은 한글이 자모 단위(NFD)로 저장될 수 있다.
      눈으로는 같은 "우유"인데 문자열 비교는 틀리게 나온다.
    - 공백 제거: 성분표는 "탈지 분유", "탈지분유"를 섞어 쓴다.
    """
    return "".join(unicodedata.normalize("NFC", text).split())


def load_synonyms(path: Path) -> dict[str, list[str]]:
    """동의어 사전을 읽는다.

    형태: {"우유": ["우유", "탈지분유", "전지분유", "카제인", "유청", ...]}
    ``_`` 로 시작하는 키(``_meta`` 등)는 메모용이라 건너뛴다.

    파일을 열 때 encoding="utf-8" 을 반드시 명시한다.
    윈도우 기본값은 cp949 라서 생략하면 팀원 노트북에서만 깨진다.

    Returns:
        표준명 → 표기 목록. 표기는 ``normalize_term`` 을 거친 값이고,
        각 목록의 첫 항목은 표준명 자신이다(사전에 빠뜨려도 넣어 준다).

    Raises:
        ValueError: 사전이 잘못됐을 때. 판정 근거가 되는 데이터라 조용히 넘기지 않는다.
            - KNOWN_ALLERGENS 에 없는 키 (오타 방지)
            - 표기 목록이 문자열 배열이 아님
            - 한 표기가 두 알레르겐에 동시에 들어 있음 (판정 결과가 모호해짐)
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 최상위가 객체(dict)가 아닙니다")

    known = {normalize_term(a) for a in KNOWN_ALLERGENS}
    owner: dict[str, str] = {}  # 표기 → 처음 등록한 알레르겐 (중복 검사용)
    result: dict[str, list[str]] = {}

    for key, aliases in raw.items():
        if key.startswith("_"):
            continue
        name = normalize_term(key)
        if name not in known:
            raise ValueError(f"{path}: 알 수 없는 알레르겐 '{key}' (KNOWN_ALLERGENS 확인)")
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise ValueError(f"{path}: '{key}' 의 값은 문자열 배열이어야 합니다")

        terms: list[str] = []
        for term in (name, *aliases):
            term = normalize_term(term)
            if not term or term in terms:
                continue
            if term in owner and owner[term] != name:
                raise ValueError(f"{path}: '{term}' 이 '{owner[term]}' 와 '{name}' 에 중복됩니다")
            owner[term] = name
            terms.append(term)
        result[name] = terms

    return result


def judge_allergens(
    panel: IngredientPanel,
    user_allergens: list[str],
    synonyms: dict[str, list[str]],
) -> AllergenJudgement:
    """성분표와 사용자 알레르기를 대조한다.

    Args:
        panel: OCR 까지 끝난 성분표.
        user_allergens: 사용자가 등록한 알레르기. 예: ["땅콩", "우유"]
        synonyms: 동의어 사전.

    Returns:
        판정 결과. matched_terms 에는 성분표에 실제로 적혀 있던 말을
        그대로 담는다("탈지분유"). matched_allergens 에는 그것이 해당하는
        알레르겐을 담는다("우유"). 둘 다 있어야 사용자가 납득할 설명이 된다.

    UNCERTAIN 을 내야 하는 경우
    ---------------------------
    - OCR confidence 가 낮아 글자를 제대로 못 읽었을 때
    - 성분표를 찾긴 했는데 원재료명 항목이 안 보일 때
    - "혼합제제", "기타가공품" 처럼 내용을 알 수 없는 표기가 있을 때
    """
    raise NotImplementedError
