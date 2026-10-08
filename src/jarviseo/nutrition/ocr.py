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
    "split_sections",
    "split_terms",
]

# 원재료명 / 원재료명 및 함량 / 원재료 및 함량 (OCR 이 글자 사이에 공백을 넣는 경우 포함)
_INGREDIENT_HEADER = re.compile(r"원\s*재\s*료\s*명?(?:\s*및\s*(?:그\s*)?함\s*량)?\s*[:：]?")
# "알레르기 유발물질", "알레르기유발물질 표시"
# EasyOCR 이 "르"를 "로"처럼 잘못 읽어서 셋째 글자는 아무 한 글자나 받는다.
# "유발물질"만 있어도 머리말로 본다.
_ALLERGEN_HEADER = re.compile(
    r"(?:알\s*레\s*\S\s*기\s*(?:유\s*발\s*물\s*질)?|유\s*발\s*물\s*질)(?:\s*표\s*시)?\s*[:：]?"
)
# "같은 제조시설", "동일한 제조시설", "같은 시설" … "혼입"
# EasyOCR 이 "같은"을 "끝은"처럼 잘못 읽는 일이 있어서 "제조시설"만 있어도 혼입 문장으로 본다.
_CROSS = re.compile(r"(?:같은|동일한?)\s*시\s*설|제\s*조\s*시\s*설|혼\s*입")
# "우유, 대두 함유" — 뒤에 량/될/할 이 붙으면 함유량·함유될 수 있음 등 다른 뜻이다
_CONTAINS = re.compile(r"([가-힣A-Za-z0-9,\s·/]+?)\s*함\s*유(?![량될할])")
# 원재료 구간이 끝나는 다른 항목들
_OTHER_HEADER = re.compile(
    r"내\s*용\s*량|영\s*양\s*(?:정\s*보|성\s*분)|보\s*관\s*방\s*법|소\s*비\s*기\s*한|유\s*통\s*기\s*한"
    r"|품\s*목\s*보\s*고\s*번\s*호|제\s*조\s*원|판\s*매\s*원|반\s*품|교\s*환|부\s*정\s*불\s*량"
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

    ingredients: str = ""  # 원재료명 구간
    allergen_notice: str = ""  # "우유, 대두 함유" 같은 알레르기 표시
    cross_contamination: str = ""  # "같은 제조시설에서 ~ 사용" 문장
    has_ingredient_header: bool = False  # "원재료명"을 찾았나. 못 찾으면 판정 불가로 본다


def _normalize(text: str) -> str:
    """OCR 이 섞어 내는 전각·변형 기호를 하나로 맞춘다."""
    text = unicodedata.normalize("NFC", text)
    for src, dst in {"（": "(", "）": ")", "[": "(", "]": ")", "{": "(", "}": ")",
                     "，": ",", "、": ",", "；": ",", ";": ",", "：": ":"}.items():  # fmt: skip
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip()


def join_lines(lines: list[OCRLine]) -> str:
    """OCR 줄을 위→아래 순서로 이어 붙인다.

    공백으로 잇는다. 한 단어가 줄바꿈에 걸려 "탈지분 유"가 되어도
    판정할 때 공백을 지우고 비교하므로(allergen.normalize_term) 문제없다.
    """
    return _normalize(" ".join(line.text for line in lines))


# 혼입 문구는 보통 "이 제품은 ~"으로 시작한다. 문장 시작을 찾는 단서로 쓴다.
_SENTENCE_START = re.compile(r"(?:이|본|해당)\s*제\s*품")
_SENTENCE_END = ".※"
# 혼입 문장 시작을 못 찾으면 이만큼만 거슬러 올라간다 (원재료까지 삼키지 않게)
_MAX_LOOKBACK = 40


def _sentence_end(text: str, pos: int) -> int:
    ends = [i for i in (text.find(ch, pos) for ch in _SENTENCE_END) if i != -1]
    return min(ends) if ends else len(text)


def _cross_sentence(text: str, pos: int, floor: int = 0) -> tuple[int, int, bool]:
    """혼입 문구가 들어 있는 문장의 (시작, 끝, 시작이 확실한가).

    마침표가 없는 성분표가 많아서 마침표만 보면 원재료까지 한 문장으로 잡힌다.
    "이 제품은"을 먼저 찾고, 없으면 ※·마침표·닫는 괄호·floor 중 가까운 곳에서,
    그래도 멀면 40자까지만 거슬러 올라간다. 이때는 시작이 불확실하다.
    """
    base = max(floor, max(text.rfind(ch, 0, pos) for ch in _SENTENCE_END + ")") + 1)
    phrases = list(_SENTENCE_START.finditer(text, base, pos))
    if phrases:
        return phrases[-1].start(), _sentence_end(text, pos), True
    return max(base, pos - _MAX_LOOKBACK), _sentence_end(text, pos), False


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
        found = list(_CONTAINS.finditer(masked))
        out.allergen_notice = ", ".join(c.group(1).strip(" ,") for c in found)
        for c in found:
            masked = _blank(masked, c.start(), c.end())

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
        out.ingredients = (rest[: stop.start()] if stop else rest).strip(" .,")
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


def _get_reader() -> OcrReader:
    global _reader
    if _reader is None:
        import easyocr  # torch 를 끌고 와서 무겁다. 쓸 때만 불러온다(CI 에는 설치하지 않음)

        _reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
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
