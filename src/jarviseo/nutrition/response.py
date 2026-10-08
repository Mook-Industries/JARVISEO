"""판정 결과 → 사용자에게 읽어 줄 안내 문장.  담당: 권용현

LLM 을 쓰지 않는다. 판정 결과를 규칙(템플릿)으로 문장으로 바꾼다.
VLM 이 문장을 지어내면 판정과 다른 말을 할 수 있어서, 성분 경로는 이 문장을 그대로 TTS 로 보낸다.

원칙
----
- 짧게, 핵심 먼저. TTS 로 듣는 문장이다. "무엇이 들어 있는지"를 먼저 말한다.
- '안전하다', '먹어도 된다'고 말하지 않는다. 미검출은 확인한 범위에서 못 찾았다는 뜻이다.
- 근거를 말한다. 성분표 표기가 알레르겐 이름과 다르면 표기도 말한다(탈지분유 → 우유).
- 진단이 아니다. 표시된 성분을 읽어 전할 뿐이라 "드시지 마세요" 같은 지시는 하지 않는다.
"""

from __future__ import annotations

from jarviseo.types import AllergenJudgement, AllergenVerdict

__all__ = ["RETAKE_CLOSER", "RETAKE_WHOLE", "build_message", "josa"]

# 재처리 후에도 판독이 안 될 때 다시 비춰 달라고 하는 안내 (기획 05-workflow 5.4)
RETAKE_CLOSER = "성분표를 더 가까이 가리켜 주세요."
RETAKE_WHOLE = "성분표 전체가 보이도록 가리켜 주세요."

# 근거 표기는 이만큼만 읽는다. 길면 듣다가 놓친다.
_MAX_TERMS = 3

_CHECK_PACKAGE = "포장지를 직접 확인해 주세요."
_UNDETERMINED = {
    "no_user_allergens": (
        "등록된 알레르기가 없어서 확인할 수 없어요. 설정에서 알레르기를 등록해 주세요."
    ),
    "low_ocr_conf": f"성분표 글자를 정확히 읽지 못했어요. {_CHECK_PACKAGE}",
    "unreadable": f"성분표 글자를 정확히 읽지 못했어요. {_CHECK_PACKAGE}",
    "no_ingredients": "원재료명을 찾지 못했어요. 성분표 전체가 보이게 비춰 주세요.",
    "opaque_ingredient": f"성분을 알 수 없는 원료가 들어 있어 확인할 수 없어요. {_CHECK_PACKAGE}",
}
_UNDETERMINED_DEFAULT = f"성분을 확인하지 못했어요. {_CHECK_PACKAGE}"


def _final_consonant(word: str) -> int | None:
    """마지막 한글 음절의 받침 번호. 받침이 없으면 0, 한글로 끝나지 않으면 None."""
    for ch in reversed(word.strip()):
        if "가" <= ch <= "힣":
            return (ord(ch) - ord("가")) % 28
        if ch.isalnum():
            return None  # 영문·숫자로 끝나면 받침을 모른다
    return None


def josa(word: str, pair: str) -> str:
    """받침에 맞는 조사를 붙인다. pair: "이/가", "은/는", "을/를", "과/와", "으로/로".

    "으로/로"는 받침이 ㄹ(8번)이면 '로'다: 밀로, 탈지분유로, 전란액으로.
    한글로 끝나지 않으면 받침 없는 쪽을 쓴다.
    """
    with_final, without = pair.split("/")
    final = _final_consonant(word)
    if pair == "으로/로":
        use = with_final if final not in (None, 0, 8) else without
    else:
        use = with_final if final else without
    return word + use


def _evidence(terms: list[str], allergens: list[str]) -> list[str]:
    """알레르겐 이름과 다른 표기만 근거로 말한다. "우유"를 "우유"로 적혀 있다고 할 필요는 없다."""
    return [t for t in terms if t not in allergens][:_MAX_TERMS]


def build_message(j: AllergenJudgement, product_name: str = "") -> str:
    """판정 결과를 TTS 로 읽을 문장으로 만든다."""
    where = f"{product_name}에 " if product_name else ""

    if j.verdict is AllergenVerdict.CONTAINS:
        names = ", ".join(j.matched_allergens)
        parts = [f"주의하세요. {where}{josa(names, '이/가')} 들어 있어요."]
        if ev := _evidence(j.matched_terms, j.matched_allergens):
            parts.append(f"성분표에 {josa(', '.join(ev), '으로/로')} 적혀 있어요.")
        if j.may_contain_allergens:
            may = ", ".join(j.may_contain_allergens)
            parts.append(f"{josa(may, '은/는')} 같은 시설에서 써서 섞여 들어갔을 수 있어요.")
        return " ".join(parts)

    if j.verdict is AllergenVerdict.MAY_CONTAIN:
        may = ", ".join(j.may_contain_allergens)
        return (
            f"{where}원재료에는 등록하신 알레르기 성분이 없어요. "
            f"다만 {josa(may, '을/를')} 쓰는 시설에서 만들어 섞여 들어갔을 수 있어요."
        )

    if j.verdict is AllergenVerdict.NOT_DETECTED:
        return (
            "확인한 성분표에서 등록하신 알레르기 성분을 찾지 못했어요. "
            "포장지 표시도 함께 확인해 주세요."
        )

    return _UNDETERMINED.get(j.reason, _UNDETERMINED_DEFAULT)
