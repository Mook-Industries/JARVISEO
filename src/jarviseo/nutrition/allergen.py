"""알레르기 판정.  담당: 권용현

책임 지표: 알레르기 판정 recall

확정된 설계 결정: **정밀도보다 재현율을 우선한다.**
놓친 알레르겐과 과잉 경고는 피해가 비대칭이다. 경고를 한 번 더 하는 것보다
땅콩을 놓치는 쪽이 훨씬 위험하다. 애매하면 미검출이라고 하지 말고 UNDETERMINED 를 낸다.
판정 값: 함유(CONTAINS) / 혼입 가능(MAY_CONTAIN) / 미검출(NOT_DETECTED) / 확인 불가(UNDETERMINED)

이 파일의 정확도를 결정하는 것은 모델이 아니라 동의어 사전이다
-----------------------------------------------------------------
'우유'를 찾는데 성분표에는 '탈지분유', '카제인', '유청'이라고 적혀 있다.
사전에 이 대응이 없으면 못 잡는다. 알려진 함정 중 하나이고,
사전 구축에 충분한 시간을 배정해야 한다.

사전은 코드가 아니라 데이터로 관리한다: data/datasets/allergen_synonyms.json
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from jarviseo.types import AllergenJudgement, AllergenVerdict, ProductInfo

__all__ = [
    "KNOWN_ALLERGENS",
    "find_allergens",
    "judge_allergens",
    "judge_product",
    "load_synonyms",
    "normalize_term",
]

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


# 내용을 알 수 없는 원료 표기. 괄호로 내용이 적혀 있으면 그 안을 읽으면 되므로
# 괄호가 따라오지 않을 때만 판정 불가 근거로 본다.
_OPAQUE = re.compile(r"(?:혼합제제|기타가공품|복합원재료)(?!\s*\()")


def find_allergens(text: str, synonyms: dict[str, list[str]]) -> dict[str, list[str]]:
    """글자에서 알레르겐 표기를 찾는다. {알레르겐: [찾은 표기, ...]}

    성분 목록을 쪼갠 결과가 아니라 글자 전체에서 찾는다. OCR 이 쉼표를 빼먹어
    "해바라기유양파분말새우분말"처럼 붙어 나와도 '새우'를 찾기 위해서다.
    비교 전에 공백을 지운다(normalize_term). "탈지 분유"와 "탈지분유"를 같게 본다.

    긴 표기부터 맞추고, 이미 맞춘 글자 자리는 다시 쓰지 않는다.
    그래야 "밀크"가 우유로 잡히고 '밀'로는 잡히지 않는다(땅콩버터→버터, 메밀→밀 도 같다).
    사전의 모든 알레르겐 표기를 대상으로 한다. 사용자 알레르기만 보면
    긴 표기가 다른 알레르겐 것일 때 짧은 표기가 남아서 오탐이 난다.

    한 글자 표기(밀·게·굴·잣·콩)는 다른 낱말 속에서도 잡힐 수 있다(예: 굴비→굴).
    재현율 우선이라 그대로 둔다. 과잉 경고 사례가 모이면 사전에서 긴 표기로 막는다.
    """
    compact = normalize_term(text)
    if not compact:
        return {}
    terms = sorted(
        ((t, a) for a, ts in synonyms.items() for t in ts), key=lambda x: len(x[0]), reverse=True
    )
    taken = [False] * len(compact)
    found: dict[str, list[str]] = {}
    for term, allergen in terms:
        start = compact.find(term)
        while start != -1:
            end = start + len(term)
            if not any(taken[start:end]):
                taken[start:end] = [True] * len(term)
                hits = found.setdefault(allergen, [])
                if term not in hits:
                    hits.append(term)
            start = compact.find(term, start + 1)
    return found


def judge_allergens(
    user_allergens: list[str],
    synonyms: dict[str, list[str]],
    *,
    ingredients: str = "",
    notice: str = "",
    cross: str = "",
    readable: bool = True,
    reason: str = "",
) -> AllergenJudgement:
    """성분 표시와 사용자 알레르기를 대조한다. 바코드 조회 결과와 OCR 결과를 같은 함수로 판정한다.

    Args:
        user_allergens: 사용자가 등록한 알레르기. 예: ["땅콩", "우유"]
            사전에 없는 이름(예: "키위")도 그 이름 그대로 찾는다.
        synonyms: ``load_synonyms`` 결과.
        ingredients: 원재료명 글자. 여기서 찾으면 함유.
        notice: 알레르기 표시 글자("우유, 대두 함유"). 여기서 찾아도 함유.
        cross: 같은 제조시설 문장. 여기서만 찾으면 혼입 가능.
        readable: False 면 글자를 믿을 수 없다(OCR 신뢰도 미달 등).
        reason: readable=False 일 때 남길 사유. 예: "low_ocr_conf"

    Returns:
        matched_terms 에는 성분표에 적혀 있던 표기("탈지분유"), matched_allergens 에는
        그것이 해당하는 알레르겐("우유")을 담는다. 혼입 가능은 may_contain_* 에 따로 담는다.

    판정 우선순위: 함유 > 판독 불가 > 혼입 가능 > 그 밖의 확인 불가 > 미검출
    - 함유가 하나라도 있으면 글자를 다 못 읽었어도 함유다(경고는 확실하므로).
    - 판독 불가면 혼입 문장을 찾았어도 확인 불가다. 원재료에 없다고 말할 근거가 없다.
    - 확인 불가: 등록 알레르기 없음 / 판독 불가 / 원재료 글자 없음 / 내용 모를 원료 표기
    - 미검출은 '안전'이 아니다. 확인한 표시 범위에서 못 찾았다는 뜻이다.
    """
    # "계란"처럼 사전의 다른 표기로 등록했어도 표준명("알류")으로 맞춘다
    canonical = {t: a for a, ts in synonyms.items() for t in ts}
    wanted: list[str] = []
    for a in user_allergens:
        name = normalize_term(a)
        name = canonical.get(name, name)
        if name and name not in wanted:
            wanted.append(name)
    if not wanted:
        return AllergenJudgement(verdict=AllergenVerdict.UNDETERMINED, reason="no_user_allergens")

    table = dict(synonyms)
    for name in wanted:
        table.setdefault(name, [name])  # 사전에 없는 알레르기(예: 키위)는 이름 그대로 찾는다

    contains: dict[str, list[str]] = {}
    for text in (ingredients, notice):
        for allergen, hits in find_allergens(text, table).items():
            bucket = contains.setdefault(allergen, [])
            bucket += [h for h in hits if h not in bucket]
    cross_hits = find_allergens(cross, table)

    hit = [a for a in wanted if a in contains]
    may = [a for a in wanted if a in cross_hits and a not in contains]
    result = AllergenJudgement(
        verdict=AllergenVerdict.NOT_DETECTED,
        matched_allergens=hit,
        matched_terms=[t for a in hit for t in contains[a]],
        may_contain_allergens=may,
        may_contain_terms=[t for a in may for t in cross_hits[a]],
    )

    if hit:
        result.verdict = AllergenVerdict.CONTAINS
    elif not readable:
        # 혼입 문장만 읽혔어도 '혼입 가능'으로 끝내지 않는다. 그러면 "원재료에는 없어요"라고
        # 말하게 되는데, 원재료를 제대로 못 읽었으니 근거가 없다. 혼입 정보는 결과에 남긴다.
        result.verdict, result.reason = AllergenVerdict.UNDETERMINED, reason or "unreadable"
    elif may:
        result.verdict = AllergenVerdict.MAY_CONTAIN
    elif not normalize_term(ingredients + notice):
        result.verdict, result.reason = AllergenVerdict.UNDETERMINED, "no_ingredients"
    elif _OPAQUE.search(ingredients):
        result.verdict, result.reason = AllergenVerdict.UNDETERMINED, "opaque_ingredient"
    return result


def judge_product(
    info: ProductInfo, user_allergens: list[str], synonyms: dict[str, list[str]]
) -> AllergenJudgement:
    """바코드 조회 결과(캐시·API)를 판정한다. 공공 데이터라 판독 불가는 없다."""
    result = judge_allergens(
        user_allergens,
        synonyms,
        ingredients=info.raw_ingredients,
        notice=info.allergen_notice,
        cross=info.cross_contamination,
    )
    result.product, result.source = info, info.source
    return result
