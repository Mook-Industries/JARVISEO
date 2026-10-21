"""판정 결과 → 사용자에게 읽어 줄 안내 문장.  담당: 권용현

LLM 을 쓰지 않는다. 판정 결과를 규칙(템플릿)으로 문장으로 바꾼다.
VLM 이 문장을 지어내면 판정과 다른 말을 할 수 있어서, 성분 경로는 이 문장을 그대로 TTS 로 보낸다.

원칙
----
- 짧게, 핵심 먼저. TTS 로 듣는 문장이다. "무엇이 들어 있는지"를 먼저 말한다.
- '안전하다', '먹어도 된다'고 말하지 않는다. 미검출은 확인한 범위에서 못 찾았다는 뜻이다.
- 함유·혼입 경고는 "해당 제품에는"으로 시작하는 한 문장.
  길면 군말을 줄이고, 알레르기 이름은 다 말한다.
  알레르기는 사용자가 등록한 이름으로 말한다(계란으로 등록했으면 "알류"가 아니라 "계란").
- 진단이 아니다. 표시된 성분을 읽어 전할 뿐이라 "드시지 마세요" 같은 지시는 하지 않는다.
"""

from __future__ import annotations

from jarviseo.types import AllergenJudgement, AllergenVerdict

__all__ = [
    "OCCLUDED_NOTE",
    "RETAKE_CLOSER",
    "RETAKE_GLARE",
    "RETAKE_STEADY",
    "RETAKE_UNCOVER",
    "MAX_WARNING_CHARS",
    "RETAKE_OTHER_SIDE",
    "RETAKE_WHOLE",
    "SMALL_TEXT_NOTE",
    "build_message",
    "josa",
    "warning_sentence",
]

# 재처리 후에도 판독이 안 될 때 다시 비춰 달라고 하는 안내 (기획 05-workflow 5.4)
RETAKE_CLOSER = "성분표를 더 가까이 가리켜 주세요."
RETAKE_WHOLE = "성분표 전체가 보이도록 가리켜 주세요."
# 글자는 꽤 읽었는데 원재료명이 없을 때. 영양정보 · 주의사항 면을 보여 준 경우가 많다(크런키 실측).
RETAKE_OTHER_SIDE = "원재료명이 안 보여요. 원재료명이 적힌 면을 보여 주세요."
# 성분표 일부가 가려졌거나 잘렸을 때 (nutrition/occlusion.py).
# 찾은 경고가 있으면 경고 뒤에 OCCLUDED_NOTE 를 붙이고,
# 없으면 RETAKE_UNCOVER 로 다시 보여 달라고 한다.
# 가려진 자리에 알레르기가 있을 수 있어서 '못 찾았어요'라고 말하지 않는다.
OCCLUDED_NOTE = "성분표 일부가 가려져서 다 확인하지는 못했어요."
RETAKE_UNCOVER = "성분표 일부가 가려져서 잘 인식하지 못했어요. 가리지 않게 다시 보여 주세요."
# 성분표 글씨가 작게 찍혔을 때 (service.MIN_TEXT_PX). 찾은 경고 뒤에 붙인다.
# 못 찾았으면 RETAKE_CLOSER 로 더 가까이 보여 달라고 한다.
SMALL_TEXT_NOTE = "글씨가 작게 보여서 빠진 성분이 있을 수 있어요."
# OCR 전 품질 검사에서 걸렸을 때 (nutrition/quality.py). 반사는 보정으로 못 살려서
# 촬영 방법을 바꿔 달라고 한다 (멘토 피드백 2026-10-14).
RETAKE_STEADY = "성분표를 가까이, 흔들리지 않게 다시 보여 주세요."
RETAKE_GLARE = "빛이 반사되지 않게 포장을 조금 돌려서 다시 보여 주세요."

# 함유·혼입 경고 문장이 이보다 길면 군말을 줄인 짧은 틀로 말한다. 알레르기 이름은 줄이지 않는다.
MAX_WARNING_CHARS = 50
_HEAD = "해당 제품에는"  # 함유·혼입 경고의 첫머리

_CHECK_PACKAGE = "포장지를 직접 확인해 주세요."
_UNDETERMINED = {
    "no_user_allergens": (
        "등록된 알레르기가 없어서 확인할 수 없어요. 설정에서 알레르기를 등록해 주세요."
    ),
    "low_ocr_conf": f"성분표 글자를 정확히 읽지 못했어요. {_CHECK_PACKAGE}",
    "unreadable": f"성분표 글자를 정확히 읽지 못했어요. {_CHECK_PACKAGE}",
    "no_ingredients": "원재료명을 찾지 못했어요. 성분표 전체가 보이게 비춰 주세요.",
    "occluded": f"성분표 일부가 가려져서 잘 인식하지 못했어요. {_CHECK_PACKAGE}",
    "small_text": f"글씨가 작아서 성분표를 정확히 읽지 못했어요. {_CHECK_PACKAGE}",
    "opaque_ingredient": f"성분을 알 수 없는 원료가 들어 있어 확인할 수 없어요. {_CHECK_PACKAGE}",
    "blurry": f"사진이 흐려서 성분표를 읽지 못했어요. {_CHECK_PACKAGE}",
    "glare": f"빛이 반사돼서 성분표를 읽지 못했어요. {_CHECK_PACKAGE}",
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


def _warning(contains: str, may: str, short: bool) -> str:
    if contains and may:
        tail = "섞였을 수 있어요." if short else "같은 시설에서 써서 섞여 들어갔을 수 있어요."
        return f"{_HEAD} {josa(contains, '이/가')} 들어 있고, {josa(may, '은/는')} {tail}"
    if contains:
        return f"{_HEAD} {josa(contains, '이/가')} 들어 있어요."
    if short:
        return f"{_HEAD} {josa(may, '이/가')} 섞였을 수 있어요."
    return f"{_HEAD} 같은 시설에서 쓰는 {josa(may, '이/가')} 섞여 들어갔을 수 있어요."


def warning_sentence(
    contains: list[str], may: list[str], max_chars: int = MAX_WARNING_CHARS
) -> str:
    """함유·혼입 경고 한 문장. 알레르기 이름은 하나도 빼지 않는다.

    문장이 길면 이름이 아니라 군말을 줄인다("같은 시설에서 써서 섞여 들어갔을" → "섞였을").
        짧으면: "해당 제품에는 밀이 들어 있고, 새우는 같은 시설에서 써서 섞여 들어갔을 수 있어요."
        길면:   "해당 제품에는 밀, 대두가 들어 있고, 새우, 땅콩은 섞였을 수 있어요."
    이름이 많아 짧은 틀로도 넘으면 그대로 다 말한다. 빠뜨린 알레르기는 사용자를 위험하게 한다.
    """
    full = _warning(", ".join(contains), ", ".join(may), short=False)
    if len(full) <= max_chars:
        return full
    return _warning(", ".join(contains), ", ".join(may), short=True)


def build_message(
    j: AllergenJudgement, product_name: str = "", labels: dict[str, str] | None = None
) -> str:
    """판정 결과를 TTS 로 읽을 문장으로 만든다.

    함유·혼입 가능은 "해당 제품에는"으로 시작하는 한 문장으로 말한다(``warning_sentence``).
    경고는 듣자마자 알아야 해서 문장을 나누지 않는다. 길면 군말만 줄이고 알레르기 이름은 다 말한다.
    제품명은 말하지 않는다. 사용자가 가리킨 그 제품이고, 긴 제품명은 경고를 늦게 들리게 한다.
    성분표 표기("탈지분유")도 문장에 넣지 않는다. 판정 결과(matched_terms)에는 남는다.

    Args:
        labels: 표준 이름 → 사용자가 등록한 이름(``allergen.user_labels``). "알류" 대신 "계란".
        product_name: 호출하는 쪽과 맞추려고 남겨 둔다. 문장에는 쓰지 않는다.
    """
    labels = labels or {}
    contains = [labels.get(a, a) for a in j.matched_allergens]
    may = [labels.get(a, a) for a in j.may_contain_allergens]

    if j.verdict is AllergenVerdict.CONTAINS:
        return warning_sentence(contains, may)

    # 혼입 가능은 원재료에는 없다는 뜻이다. '들어 있다'고 하지 않고 섞였을 수 있다고만 한다.
    if j.verdict is AllergenVerdict.MAY_CONTAIN:
        return warning_sentence([], may)

    # 미검출은 "읽힌 범위에서 그 표기를 찾지 못했다"는 뜻일 뿐이다. 없다고 단정하지 않는다.
    # 라벨이 바뀌었거나 잘못 읽었을 수 있어서 포장 표기를 직접 보라고 덧붙인다 (멘토 피드백).
    if j.verdict is AllergenVerdict.NOT_DETECTED:
        return (
            "읽은 성분표에서는 등록하신 알레르기 성분을 찾지 못했어요. "
            "표시가 바뀌었거나 잘못 읽었을 수 있으니 포장지를 직접 확인해 주세요."
        )

    return _UNDETERMINED.get(j.reason, _UNDETERMINED_DEFAULT)
