"""성분표 글자 읽기.  담당: 권용현

책임 지표: OCR 인식률

파이프라인: ROI 크롭 -> 업스케일 -> 전처리 -> OCR -> 구간 분리 -> 성분 목록

업스케일이 왜 필요한가
----------------------
성분표 글자는 약 2mm 다. 30cm 거리에서 1920x1080 으로 찍어도 글자 높이가
10~15픽셀 정도밖에 안 나온다. OCR 이 안정적으로 읽으려면 보통 30픽셀 이상이
필요하다. 그래서 잘라낸 뒤 2~4배로 키워서 넣는다.

(0.3MP 카메라를 기각한 이유가 이것이다. 640x480 이면 글자가 4픽셀이라
아무리 키워도 없는 정보가 생기지 않는다.)

구간을 나누는 이유
------------------
성분표에는 세 종류의 알레르기 근거가 섞여 있다.
- 원재료명      "밀가루(미국산), 탈지분유, ..."           → 함유
- 알레르기 표시 "우유, 대두, 밀 함유"                     → 함유
- 같은 제조시설 "이 제품은 땅콩을 사용한 제품과 같은 ..."  → 혼입 가능
섞어 버리면 "혼입 가능"을 "함유"로 잘못 말하게 된다.

학습은 하지 않는다. LLM 도 쓰지 않는다. 구간 분리와 파싱은 규칙(정규식)이다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from jarviseo import config
from jarviseo.nutrition.allergen import KNOWN_ALLERGENS
from jarviseo.types import BBox, OCRLine

__all__ = [
    "OcrReader",
    "PanelSections",
    "crop_and_upscale",
    "join_lines",
    "mean_confidence",
    "parse_ingredient_text",
    "parse_ingredients",
    "read_panel",
    "read_upright",
    "split_sections",
    "split_terms",
]

# 원재료명 / 원재료명 및 함량 / 원재료 및 함량 (OCR 이 글자 사이에 공백을 넣는 경우 포함)
# 성분표를 자를 때 첫 글자가 잘려 "재료명"만 남는 일이 있다. '명'까지 있으면 머리말로 본다.
# 잘린 채로 OCR 이 '재'를 '대'로 읽는 일도 있다("대료명", 실측 label_01 · PaddleOCR).
_INGREDIENT_HEADER = re.compile(
    r"(?:원\s*재\s*료\s*명?|[재대]\s*료\s*명)(?:\s*및\s*(?:그\s*)?함\s*량)?\s*[:：]?"
)
# "알레르기 유발물질", "알레르기유발물질 표시"
# EasyOCR 이 "르"를 "로"처럼 잘못 읽어서 셋째 글자는 아무 한 글자나 받는다.
# "유발물질"만 있어도 머리말로 본다.
_ALLERGEN_HEADER = re.compile(
    r"(?:알\s*레\s*\S\s*기\s*(?:유\s*발\s*물\s*질)?|유\s*발\s*물\s*질)(?:\s*표\s*시)?\s*[:：]?"
)
# "같은 제조시설", "동일한 제조시설", "같은 시설" … "혼입"
# EasyOCR 이 "같은"을 "끝은"처럼 잘못 읽는 일이 있어서 "제조시설"만 있어도 혼입 문장으로 본다.
_CROSS = re.compile(r"(?:같은|동일한?)\s*시\s*설|제\s*조\s*시\s*설|혼\s*입")
# "우유, 대두 함유" — 뒤에 량/될/할 이 붙으면 함유량·함유될 수 있음 등 다른 뜻이다.
# 앞쪽은 넉넉히 잡고(괄호 포함) 실제 알레르겐 목록은 _allergen_tail 로 잘라낸다.
_CONTAINS = re.compile(r"([가-힣A-Za-z0-9,\s·/()]+?)\s*함\s*유(?![량될할])")
# "~ 함유" 알레르기 표시에 쓰이는 이름. 법정 표시명에 흔한 다른 표기를 더했다.
_NOTICE_WORDS = {
    *KNOWN_ALLERGENS, "계란", "달걀", "난류", "소고기", "굴", "전복", "홍합", "아황산", "조개",
}  # fmt: skip
# 원재료 구간이 끝나는 다른 항목들. 첫 글자가 잘린 "목보고번호", "장재질"도 받는다.
_OTHER_HEADER = re.compile(
    r"내\s*용\s*량|영\s*양\s*(?:정\s*보|성\s*분)|보\s*관\s*방\s*법|소\s*비\s*기\s*한|유\s*통\s*기\s*한"
    r"|품?\s*목\s*보\s*고\s*번\s*호|제\s*조\s*원|판\s*매\s*원|반\s*품|교\s*환|부\s*정\s*불\s*량"
    r"|포?\s*장\s*재\s*질|직\s*사\s*광\s*선|업\s*소\s*명"
)
# 원산지 표기. "아황산", "구연산", "젖산"처럼 '산'으로 끝나는 성분이 있어서
# '~산' 전체가 아니라 나라 이름만 원산지로 본다.
_ORIGIN = re.compile(
    r"^(?:국내|국|외국|수입|미국|호주|캐나다|중국|베트남|태국|인도네시아|말레이시아|필리핀"
    r"|뉴질랜드|독일|프랑스|이탈리아|스페인|네덜란드|벨기에|덴마크|브라질|아르헨티나|칠레"
    r"|러시아|우크라이나|인도|일본|영국|폴란드|헝가리|터키|멕시코|페루|싱가포르|대만|아일랜드"
    r"|노르웨이|남아공|핀란드|스위스|오스트리아|이스라엘|가나|코트디부아르|에콰도르)산$"
)
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s*%")
_LABELS = {"원산지", "함량", "원재료", "원재료명"}


# --------------------------------------------------------------------------
# 1. 글자 → 구간 분리 → 성분 목록
# --------------------------------------------------------------------------


@dataclass
class PanelSections:
    """성분표 원문을 세 구간으로 나눈 결과. 못 찾은 구간은 빈 문자열이다."""

    # 원재료명 구간. 머리말을 못 찾으면 다른 구간에 안 들어간 글자 전체(보조 분석용)
    ingredients: str = ""
    allergen_notice: str = ""  # "우유, 대두 함유" 같은 알레르기 표시
    cross_contamination: str = ""  # "같은 제조시설에서 ~ 사용" 문장
    has_ingredient_header: bool = False  # "원재료명"을 찾았나. 못 찾으면 판정 불가로 본다


def _normalize(text: str) -> str:
    """OCR 이 섞어 내는 전각·변형 기호를 하나로 맞춘다."""
    text = unicodedata.normalize("NFC", text)
    for src, dst in {"（": "(", "）": ")", "[": "(", "]": ")", "{": "(", "}": ")",
                     "，": ",", "、": ",", "；": ",", ";": ",", "：": ":"}.items():  # fmt: skip
        text = text.replace(src, dst)
    # OCR 이 쉼표를 마침표로 읽는 일이 많다("대두,땅콩.밀.복숭아"). 한글 사이에 띄어쓰기 없이
    # 낀 마침표는 문장 끝이 아니므로 쉼표로 바꾼다. 안 바꾸면 혼입 목록이 거기서 끊긴다.
    text = re.sub(r"(?<=[가-힣])\.(?=[가-힣])", ",", text)
    return re.sub(r"\s+", " ", text).strip()


def join_lines(lines: list[OCRLine]) -> str:
    """OCR 줄을 위→아래 순서로 이어 붙인다.

    공백으로 잇는다. 한 단어가 줄바꿈에 걸려 "탈지분 유"가 되어도
    판정할 때 공백을 지우고 비교하므로(allergen.normalize_term) 문제없다.
    """
    return _normalize(" ".join(line.text for line in lines))


# 혼입 문구는 보통 "이 제품은 ~"으로 시작한다. 문장 시작을 찾는 단서로 쓴다.
_SENTENCE_START = re.compile(r"(?:이|본|해당)\s*제\s*품")
_SENTENCE_END = ".※•"
# 항목이 바뀌는 자리: 문장 부호, 품목보고번호·바코드 같은 긴 숫자, 이미 지운 구간(공백 2칸 이상).
# _normalize 가 공백을 한 칸으로 줄이므로 2칸 이상은 _blank 로 지운 자리뿐이다.
_BOUNDARY = re.compile(r"[.※•]|\d{6,}|\s{2,}")
# 혼입 문장 시작을 못 찾으면 이만큼만 거슬러 올라간다 (원재료까지 삼키지 않게)
_MAX_LOOKBACK = 40
# "우유, 대두, 땅콩 혼입가능"처럼 이름만 늘어놓은 목록의 한 항목 최대 길이 (괄호 내용 제외)
_MAX_LIST_ITEM = 12


def _sentence_end(text: str, pos: int) -> int:
    ends = [i for i in (text.find(ch, pos) for ch in _SENTENCE_END) if i != -1]
    return min(ends) if ends else len(text)


def _top_chunks(text: str) -> list[tuple[int, int]]:
    """괄호 바깥 쉼표로 나눈 조각의 (시작, 끝) 위치. "조개류(굴, 홍합)"는 한 조각이다."""
    spans, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            spans.append((start, i))
            start = i + 1
    spans.append((start, len(text)))
    return spans


def _core(chunk: str) -> str:
    """괄호 안 내용을 뺀 이름. 짝이 안 맞는 괄호 글자도 지운다."""
    return re.sub(r"\([^)]*\)?|[()]", " ", chunk).strip()


def _is_name_list(text: str) -> bool:
    """알레르겐 이름만 쉼표로 늘어놓은 목록인가. "우유, 대두, 조개류(굴, 홍합 포함)" → True"""
    cores = [_core(text[s:e]) for s, e in _top_chunks(text)]
    cores = [c for c in cores if c]
    return len(cores) >= 2 and all(
        len(c.replace(" ", "")) <= _MAX_LIST_ITEM and not re.search(r"\d", c) for c in cores
    )


def _cross_sentence(text: str, pos: int, floor: int = 0) -> tuple[int, int, bool]:
    """혼입 문구가 들어 있는 문장의 (시작, 끝, 시작이 확실한가).

    마침표가 없는 성분표가 많아서 마침표만 보면 원재료까지 한 문장으로 잡힌다.
    가장 가까운 항목 경계(•·※·마침표·긴 숫자·지운 구간) 뒤에서
    1) "이 제품은"을 찾으면 거기서부터,
    2) 없어도 그 사이가 이름 목록("우유, 대두, 땅콩 혼입가능")이면 경계부터 확실한 시작으로 본다.
    3) 둘 다 아니면 예전처럼 닫는 괄호까지 보고, 그래도 멀면 40자까지만 거슬러 올라간다.
       이때는 시작이 불확실하다.
    경계를 못 찾아 floor(원재료명 머리말 끝)까지 간 경우에는 2)를 쓰지 않는다.
    원재료 목록도 이름 목록처럼 보여서, 원재료를 통째로 '혼입 가능'으로 낮출 수 있기 때문이다.
    """
    bounds = [b.end() for b in _BOUNDARY.finditer(text, floor, pos)]
    base = bounds[-1] if bounds else floor
    phrases = list(_SENTENCE_START.finditer(text, base, pos))
    end = _sentence_end(text, pos)
    if phrases:
        return phrases[-1].start(), end, True
    if bounds and _is_name_list(text[base:pos]):
        return base, end, True
    base = max(base, text.rfind(")", 0, pos) + 1)
    return max(base, pos - _MAX_LOOKBACK), end, False


def _open_paren(text: str) -> int:
    """닫히지 않은 마지막 여는 괄호의 위치. 없으면 -1."""
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            stack.pop()
    return stack[-1] if stack else -1


def _allergen_tail(text: str) -> int:
    """'~ 함유' 앞 글자에서 알레르겐 표시가 시작하는 위치. 없으면 len(text).

    정규식은 앞쪽 원재료와 "원재료명" 머리말까지 붙여 잡는다. 그대로 지우면
    머리말이 사라져 판정이 '확인 불가'로 간다. 그래서
    - "합성향료 (우유 대두 밀 함유)" 처럼 닫히지 않은 괄호 안이 모두 표시 이름이면
      괄호 안 전체를 쓰고 (OCR 이 쉼표를 빼먹어도 받기 위해서),
    - 아니면 뒤에서부터 알레르기 표시 이름(우유·대두·계란 …)인 낱말만 남긴다.
      표시 이름이 아닌 낱말에서 멈춘다("정제 미강유 우유" → "우유", "건청경채 닭고기 쇠고기" →
      "닭고기 쇠고기").
    """
    base = _open_paren(text) + 1  # 닫히지 않은 괄호가 없으면 0
    if base:
        words = [w for w in re.split(r"[,\s]+", _core(text[base:])) if w]
        if words and all(w in _NOTICE_WORDS for w in words):
            return base
    sub = text[base:]
    start = len(sub)
    for s, e in reversed(_top_chunks(sub)):
        words = _core(sub[s:e]).split()
        # OCR 이 쉼표를 빼먹어 "건청경채 닭고기 쇠고기"처럼 붙어 오므로 낱말 단위로 거슬러 간다.
        # 표시 이름이 아닌 낱말("건청경채", "정제 미강유")을 만나면 거기서 멈추고 원재료로 남긴다.
        kept = 0
        for w in reversed(words):
            if w not in _NOTICE_WORDS:
                break
            kept += 1
        if not kept:
            break
        chunk = sub[s:e]
        pos = len(chunk)
        for w in reversed(words[-kept:]):
            pos = chunk.rfind(w, 0, pos)
        start = s + pos
        if kept < len(words):
            break
    return base + start


_WORD = re.compile(r"[^,\s·/()]+")


def _name_lists(text: str) -> list[tuple[int, int]]:
    """'혼입' 글자 없이 알레르겐 이름만 늘어놓은 목록의 (시작, 끝) 위치들.  (#76)

    항목 경계(•·※·마침표·긴 숫자)로 나눈 덩어리마다, 알레르기 표시 이름(우유·대두·밀 …)인
    낱말이 2개 이상이고 그 사이에 쉼표가 있으면 첫 이름부터 마지막 이름까지를 목록으로 본다.
    OCR 이 '혼입 가능' 글자를 못 읽은 혼입 목록을 잡으려는 것이다. 사이에 낀 깨진 낱말
    ("달길", "땅몸")도 함께 넘겨서, 판정 쪽 동의어·유사 매칭이 볼 수 있게 한다.
    """
    spans = []
    cuts = [0, *(b.end() for b in _BOUNDARY.finditer(text)), len(text)]
    for lo, hi in zip(cuts, cuts[1:], strict=False):
        names = [w for w in _WORD.finditer(text, lo, hi) if w.group() in _NOTICE_WORDS]
        if len(names) >= 2 and "," in text[names[0].start() : names[-1].end()]:
            spans.append((names[0].start(), names[-1].end()))
    return spans


def _blank(text: str, s: int, e: int) -> str:
    """이미 쓴 구간을 공백으로 지운다. 길이를 유지해야 다른 위치값이 어긋나지 않는다."""
    return text[:s] + " " * (e - s) + text[e:]


def split_sections(text: str) -> PanelSections:
    """성분표 원문을 원재료명 / 알레르기 표시 / 같은 제조시설 구간으로 나눈다."""
    text = _normalize(text)
    out = PanelSections()
    masked = text  # 이미 쓴 구간은 지워서 다른 구간에 다시 잡히지 않게 한다

    # 1) 알레르기 표시: "알레르기 유발물질: ~" 이 있으면 그 문장, 없으면 "~ 함유" 괄호
    header = _ALLERGEN_HEADER.search(masked)
    if header:
        e = _sentence_end(masked, header.end())
        out.allergen_notice = masked[header.end() : e].strip(" .:")
        masked = _blank(masked, header.start(), e)
    else:
        notices = []
        for c in list(_CONTAINS.finditer(masked)):
            # 원재료 목록과 머리말까지 함께 지우지 않도록 알레르겐 목록 부분만 쓴다
            s = c.start(1) + _allergen_tail(c.group(1))
            if s < c.end(1):
                notices.append(masked[s : c.end(1)].strip(" ,"))
                masked = _blank(masked, s, c.end())
        out.allergen_notice = ", ".join(notices)

    # 2) 같은 제조시설 (혼입 가능)
    head = _INGREDIENT_HEADER.search(masked)
    m = _CROSS.search(masked)
    if m:
        floor = head.end() if head and head.end() <= m.start() else 0
        s, e, sure = _cross_sentence(masked, m.start(), floor)
        out.cross_contamination = masked[s:e].strip(" .")
        # 문장 시작이 불확실하면 혼입 문구 앞쪽은 원재료 구간에 남긴다.
        # 경계가 애매한 성분을 '혼입 가능'으로 낮춰 말하는 것보다 '함유'로 남기는 쪽이 안전하다.
        masked = _blank(masked, s if sure else m.start(), e)

    # 3) 원재료명: 머리말 뒤부터 다른 항목 머리말 앞까지
    if head:
        out.has_ingredient_header = True
        rest = masked[head.end() :]
        stop = _OTHER_HEADER.search(rest)
        if stop:
            # 다른 항목 뒤에 남은 알레르겐 이름 목록은 '혼입 가능'으로 본다(#76).
            # 어느 구간에도 안 넣으면 통째로 놓친다. 놓치는 것보다 낮춰 말하는 쪽이 안전하다.
            tail = rest[stop.start() :]
            lists = [tail[s:e].strip(" ,") for s, e in _name_lists(tail)]
            out.cross_contamination = ", ".join(x for x in (out.cross_contamination, *lists) if x)
            rest = rest[: stop.start()]
    else:
        # 머리말을 못 찾으면 남은 글자 전체를 보조로 원재료 칸에 둔다 (멘토 피드백 2026-10-14).
        # has_ingredient_header 는 False 로 두므로 판정은 '확인 불가'로 가고,
        # 그 안에서 알레르겐을 찾으면 '함유'로 경고만 한다. 버리면 그 경고까지 놓친다.
        rest = masked
    out.ingredients = re.sub(r"\s+", " ", rest).strip(" .,•")
    return out


def _split_top(text: str) -> list[str]:
    """괄호 바깥의 쉼표에서만 자른다. "초콜릿(설탕, 코코아버터), 밀가루" → 2개."""
    parts, depth, buf = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)  # OCR 이 여는 괄호를 놓쳐도 음수가 되지 않게
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _outside_and_inside(item: str) -> tuple[str, list[str]]:
    """ "밀가루(밀:미국산)" → ("밀가루", ["밀:미국산"]). 닫히지 않은 괄호도 안쪽으로 본다."""
    outside, inside, depth, buf = [], [], 0, []
    for ch in item:
        if ch == "(":
            if depth == 0:
                buf = []
            else:
                buf.append(ch)
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
            if depth == 0:
                inside.append("".join(buf))
            else:
                buf.append(ch)
        elif depth > 0:
            buf.append(ch)
        else:
            outside.append(ch)
    if depth > 0 and buf:
        inside.append("".join(buf))
    return "".join(outside).strip(), inside


def _clean(term: str) -> str:
    term = _PERCENT.sub("", term)
    term = re.sub(r"\s*(?:등\s*)?함\s*유$", "", term.strip(" .:·-※"))  # "대두 등 함유" → 대두
    term = term.strip(" .:·-※()")
    if not term or term in _LABELS or _ORIGIN.match(term.replace(" ", "")):
        return ""
    return term


def parse_ingredient_text(text: str) -> list[str]:
    """원재료명 구간 → 성분 목록. 순서를 지키고 중복은 뺀다.

    괄호 안은 버리지 않는다. "초콜릿(설탕, 전지분유)"의 전지분유처럼 알레르겐이
    괄호 안에 숨어 있는 경우가 많다(재현율 우선). 원산지·함량(%)만 버린다.
    """
    result: list[str] = []

    def walk(chunk: str) -> None:
        for item in _split_top(chunk):
            name, inners = _outside_and_inside(item)
            for part in name.split(":"):
                if (term := _clean(part)) and term not in result:
                    result.append(term)
            for inner in inners:
                walk(inner)

    walk(_normalize(text))
    return result


def split_terms(text: str) -> list[str]:
    """알레르기 표시처럼 짧은 나열을 자른다. "우유, 대두·밀/땅콩" → 4개.

    EasyOCR 은 쉼표를 자주 빼먹어서 "우유 대두 밀"처럼 나온다. 알레르기 표시는
    한 단어짜리 이름의 나열이라 공백으로도 자른다. (원재료명은 "탈지 분유"처럼
    한 성분이 띄어 쓰이기도 해서 공백으로 자르지 않는다.)
    """
    text = re.sub(r"\s*(?:등\s*)?함\s*유\s*$", "", _normalize(text))
    terms = (_clean(t) for t in re.split(r"[,·/\s]+", text))
    return [t for t in terms if t and t != "등"]


def parse_ingredients(lines: list[OCRLine]) -> list[str]:
    """읽어낸 줄에서 원재료명 구간의 성분명만 뽑아낸다.

    알레르기 표시·같은 제조시설 문구가 필요하면 ``split_sections`` 를 쓴다.
    실제 오인식 패턴(예: '칼슘'의 'ㄹ' 깨짐)은 촬영본을 돌려 보고 나서 채운다.
    """
    return parse_ingredient_text(split_sections(join_lines(lines)).ingredients)


# --------------------------------------------------------------------------
# 2. 이미지 → 글자
# --------------------------------------------------------------------------

# 확대 후 긴 변 상한. 3배로 키우면 1080p 크롭이 5000px 을 넘어 CPU OCR 이 수십 초 걸린다.
_MAX_SIDE = 2560

_reader = None  # easyocr.Reader. 모델 로드가 수 초 걸려 프로세스당 한 번만 만든다.


class OcrReader(Protocol):
    """easyocr.Reader 중 우리가 쓰는 부분. 테스트에서 가짜를 끼우려고 둔다."""

    def readtext(self, image: np.ndarray, **kwargs) -> list: ...


def _local_reader() -> OcrReader:
    import easyocr  # torch 를 끌고 와서 무겁다. 쓸 때만 불러온다(CI 에는 설치하지 않음)

    return easyocr.Reader(["ko", "en"], gpu=False, verbose=False)


def _get_reader() -> OcrReader:
    """OCR 엔진. Colab 서버 주소가 설정돼 있으면 그쪽(GPU), 아니면 이 PC 의 EasyOCR(CPU)."""
    global _reader
    if _reader is None:
        if config.OCR_REMOTE_URL:
            from jarviseo.nutrition.ocr_remote import RemoteOcrReader

            _reader = RemoteOcrReader(
                config.OCR_REMOTE_URL, config.OCR_REMOTE_TOKEN, fallback=_local_reader
            )
        else:
            _reader = _local_reader()
    return _reader


def crop_and_upscale(image: np.ndarray, bbox: BBox, scale: float = 3.0) -> np.ndarray:
    """성분표 영역을 잘라내고 키운다.

    보간 방식은 cv2.INTER_CUBIC 을 쓴다. 기본값인 INTER_LINEAR 는
    글자 경계가 뭉개져서 OCR 에 불리하다.
    좌표가 이미지 밖으로 나가면 안쪽으로 자른다. 긴 변은 _MAX_SIDE 를 넘지 않게 배율을 줄인다.

    Raises:
        ValueError: 잘라낸 영역이 비었을 때.
    """
    h, w = image.shape[:2]
    x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
    x2, y2 = min(w, int(round(bbox.x2))), min(h, int(round(bbox.y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"빈 영역입니다: {bbox} (이미지 {w}x{h})")
    crop = image[y1:y2, x1:x2]
    scale = min(scale, _MAX_SIDE / max(crop.shape[:2]))
    if scale <= 1.0:
        return crop.copy()
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _to_bbox(points) -> BBox:
    pts = np.asarray(points, dtype=float)
    return BBox(
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


# 제목 칸 위로 거슬러 올라가 볼 내용 줄 수. 원재료가 이보다 길면 위쪽 줄은 놓친다.
_MAX_ROWS_ABOVE = 3
# 제목이 같은 줄 내용보다 이만큼(제목 높이 대비) 위아래로 어긋나야 '가운데 놓인 제목'으로 본다.
# 보통 표는 제목과 내용 첫 줄이 나란해서 어긋남이 0 에 가깝다. label_05 실측 0.4.
_HEADER_SHIFT = 0.25

_Row = list[tuple[BBox, str, float]]


def _left(row: _Row) -> float:
    return min(b.x1 for b, _, _ in row)


def _top(row: _Row) -> float:
    return min(b.y1 for b, _, _ in row)


def _bottom(row: _Row) -> float:
    return max(b.y2 for b, _, _ in row)


def _move_centered_header(rows: list[_Row]) -> None:
    """표의 "원재료명" 칸이 내용 여러 줄의 세로 가운데에 있으면, 제목을 내용 첫 줄 앞으로 옮긴다.

    표는 왼쪽에 제목 칸, 오른쪽에 내용 칸이다. 내용이 여러 줄이면 제목이 그 가운데에 오는
    포장이 있다(실측 label_05 새우깡). 그대로 위→아래로 이으면 제목보다 위에 있는 내용 줄
    ("쉬림프크래커[소맥분, 옥수수전분, 새우 …")이 제목 앞에 붙어서 원재료 구간에서 빠진다.

    1) 제목이 같은 줄 내용과 세로로 어긋나 있을 때만 본다. 나란하면 보통 표라 그대로 둔다.
       (내용이 홀수 줄이라 제목이 가운데 줄과 딱 나란한 표는 못 잡는다.)
    2) 오른쪽 칸에만 글자가 있는 줄(제목 칸 오른쪽에서 시작)을 제목 줄 위아래로 모으고,
       위쪽 줄을 몇 개 넣어야 제목이 내용 덩어리의 세로 가운데에 가장 가까운지 본다.
    다른 칸 내용을 끌어오면 과잉 경고가 될 수 있지만, 원재료를 놓치는 것보다는 낫다.
    """
    for i, row in enumerate(rows):
        first = min(row, key=lambda b: b[0].x1)  # 제목 칸은 줄의 맨 왼쪽
        if not _INGREDIENT_HEADER.fullmatch(first[1].strip()):
            continue
        hb, unit = first[0], first[0].height
        others = [b for b, _, _ in row if b is not hb]
        if others:
            cy = sum(b.center[1] for b in others) / len(others)
            if abs(hb.center[1] - cy) < unit * _HEADER_SHIFT:
                return  # 제목과 내용 첫 줄이 나란한 보통 표
        edge = hb.x2 - unit * 0.3  # 이보다 오른쪽에서 시작하는 줄은 내용 칸에만 글자가 있다
        above = 0
        while above < _MAX_ROWS_ABOVE and i - above >= 1 and _left(rows[i - above - 1]) >= edge:
            above += 1
        below = i
        while below + 1 < len(rows) and _left(rows[below + 1]) >= edge:
            below += 1
        bottom = _bottom(rows[below]) if below > i else _bottom(row)
        offsets = [
            abs(hb.center[1] - (_top(rows[i - n] if n else row) + bottom) / 2)
            for n in range(above + 1)
        ]
        best = min(range(above + 1), key=offsets.__getitem__)
        if best:
            row.remove(first)
            rows[i - best].append(first)  # 줄 안 순서는 뒤에서 왼쪽→오른쪽으로 다시 정렬한다
            if not row:
                rows.pop(i)
        return  # 원재료명 제목은 하나뿐이다


def _group_lines(boxes: list[tuple[BBox, str, float]]) -> list[OCRLine]:
    """EasyOCR 은 글자 덩어리 단위로 준다. 세로 중심이 가까운 덩어리를 한 줄로 묶는다.

    같은 줄 판단: 덩어리의 세로 중심이 줄의 세로 중심에서 글자 높이의 절반 이내.
    줄 안에서는 왼쪽→오른쪽, 줄끼리는 위→아래로 정렬한다.
    """
    rows: list[list[tuple[BBox, str, float]]] = []
    for item in sorted(boxes, key=lambda b: b[0].center[1]):
        box = item[0]
        if rows:
            last = rows[-1]
            cy = sum(b.center[1] for b, _, _ in last) / len(last)
            height = sum(b.height for b, _, _ in last) / len(last)
            if abs(box.center[1] - cy) <= max(height, box.height) / 2:
                last.append(item)
                continue
        rows.append([item])
    _move_centered_header(rows)

    lines = []
    for row in rows:
        row.sort(key=lambda b: b[0].x1)
        text = " ".join(t for _, t, _ in row)
        # 긴 덩어리의 신뢰도가 더 많이 반영되게 글자 수로 가중 평균한다
        n = sum(max(1, len(t)) for _, t, _ in row)
        conf = sum(c * max(1, len(t)) for _, t, c in row) / n
        box = BBox(
            min(b.x1 for b, _, _ in row), min(b.y1 for b, _, _ in row),
            max(b.x2 for b, _, _ in row), max(b.y2 for b, _, _ in row),
        )  # fmt: skip
        lines.append(OCRLine(text=text, confidence=float(conf), bbox=box))
    return lines


def read_panel(panel_image: np.ndarray, reader: OcrReader | None = None) -> list[OCRLine]:
    """확대한 성분표 이미지에서 글자를 줄 단위로 읽는다. 위→아래 순서.

    Args:
        panel_image: ``crop_and_upscale`` 결과 (BGR).
        reader: 없으면 EasyOCR 한국어+영어 모델을 쓴다. 첫 호출 때 모델을 내려받는다.
    """
    results = (reader or _get_reader()).readtext(panel_image, detail=1, paragraph=False)
    boxes = [(_to_bbox(pts), str(text).strip(), float(conf)) for pts, text, conf in results]
    return _group_lines([b for b in boxes if b[1]])


def mean_confidence(lines: list[OCRLine]) -> float:
    """글자 수로 가중한 평균 신뢰도. 줄이 없으면 0.0 (= 검증 실패).

    ``config.OCR_CONF_THRESHOLD`` 와 비교해 재처리·재촬영 여부를 정한다.
    """
    n = sum(len(line.text) for line in lines)
    if n == 0:
        return 0.0
    return sum(line.confidence * len(line.text) for line in lines) / n


# 회전 보정에서 시도하는 각도 → cv2.rotate 코드. 성분표는 캔·병 옆면에 90° 돌려 인쇄되는 일이 있다.
_ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
# 이만큼 높은 신뢰도로 이만큼 많이 읽었으면 방향은 맞다. 머리말을 못 찾은 이유는 다른 데 있다.
# 돌아간 성분표는 몇 글자만 낮은 신뢰도로 읽힌다(label_12 그대로: 0.40 × 79자).
_UPRIGHT_CONF = 0.8
_UPRIGHT_CHARS = 40


def _looks_upright(lines: list[OCRLine]) -> bool:
    """방향이 이미 맞아 보이는가. 그러면 돌려 읽기(OCR 두 번 더)를 건너뛴다."""
    return mean_confidence(lines) >= _UPRIGHT_CONF and len(join_lines(lines)) >= _UPRIGHT_CHARS


def _orientation_score(lines: list[OCRLine]) -> tuple[bool, float]:
    """방향이 맞는 정도. (원재료명 머리말을 찾았나, 신뢰도로 가중한 글자 수)

    방향이 틀리면 OCR 은 몇 글자만 낮은 신뢰도로 읽는다. 실측(평가 세트 label_12, PaddleOCR):
    그대로 0.40 × 79자, 시계 방향 90° 0.89 × 518자.
    """
    text = join_lines(lines)
    return split_sections(text).has_ingredient_header, mean_confidence(lines) * len(text)


def read_upright(
    panel_image: np.ndarray,
    reader: OcrReader | None = None,
    try_rotation: bool = True,
) -> tuple[list[OCRLine], int]:
    """성분표를 읽되, 글자가 돌아가 있으면 90°·270° 로 돌려서도 읽어 나은 쪽을 쓴다.

    그대로 읽어서 "원재료명" 머리말을 찾으면 바로 끝낸다(대부분의 사진, 추가 비용 없음).
    못 찾았어도 높은 신뢰도로 글자를 많이 읽었으면 방향은 맞은 것이라 돌리지 않는다.
    머리말이 잘렸거나 가려진 사진에서 OCR 을 두 번 더 부르던 것을 막는다(GPU 서버 실측 1~2초씩).
    (멘토 피드백 2026-10-14: 글줄이 기울어진 경우 회전 보정)

    Returns:
        (줄 목록, 돌린 각도). 각도는 시계 방향 0 / 90 / 270.
    """
    lines = read_panel(panel_image, reader=reader)
    best = (_orientation_score(lines), lines, 0)
    if best[0][0] or not try_rotation or _looks_upright(lines):
        return lines, 0
    for angle, code in _ROTATIONS.items():
        rotated = read_panel(cv2.rotate(panel_image, code), reader=reader)
        score = _orientation_score(rotated)
        if score > best[0]:
            best = (score, rotated, angle)
    return best[1], best[2]
