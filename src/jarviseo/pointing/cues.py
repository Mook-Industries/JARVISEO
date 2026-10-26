"""단서별 점수 계산.  담당: 최홍묵

후보 물체 하나하나에 "이게 사용자가 가리킨 것일 가능성"을 단서별로 0.0~1.0 점수로 매긴다.
여기서 나온 점수를 resolver.py 가 가중치로 합친다. ablation 의 축이 이 파일이다.

v1 center(화면 가운데 · 크기) · v2 point(검지 방향) · v4 lang(질문 속 색 · 종류 · 위치 · 크기)
· v5 ctx(이어 묻기 — 직전 대상과 같은 종류)가 있다. 단서를 추가할 때는 types.CueKind 에 있는
이름으로 score_* 함수를 만들고, resolver 의 IMPLEMENTED_CUES 와 단서 → 함수 표에 등록한다.

이 파일의 함수는 딥러닝이 아니다. 좌표 · 낱말표 · 색 범위로 하는 계산이고, 신경망은 detector.py ·
hands.py 까지만 쓴다. 말 읽기에 LLM 을 부르지 않는 것은 라우터와 같은 이유다 — 말 끝에서 첫 소리까지
이미 5.7초라 API 왕복을 더 넣을 자리가 없다.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import cv2
import numpy as np

from jarviseo.pointing.hands import Ray, angle_to
from jarviseo.pointing.labels import KIND_WORDS
from jarviseo.types import BBox, CueKind, CueScore, Detection

__all__ = [
    "CENTER_WEIGHT",
    "COLOR_RANGES",
    "POINT_SIGMA_DEG",
    "LangQuery",
    "parse_lang",
    "refers_back",
    "score_center",
    "score_ctx",
    "score_lang",
    "score_point",
]

# 가운데 단서 안에서 위치를 얼마나 믿을지. 나머지(0.3)는 크기.
CENTER_WEIGHT = 0.7
# 손 방향 점수가 얼마나 빨리 떨어지는지. 15° 면 0.61, 30° 면 0.14.
# valid 에서 σ 10 · 15 · 20 이 같은 결과(13/62)라 가운데 값을 골랐다.
POINT_SIGMA_DEG = 15.0

# --- 말 단서(v4)의 낱말표 ------------------------------------------------------------
# 색 이름 → 영어. 관형형("노란") · 명사형("노랑") · "…색"을 다 적는다.
COLOR_WORDS: dict[str, str] = {
    **dict.fromkeys(("빨간", "빨강", "빨간색", "빨강색", "붉은"), "red"),
    **dict.fromkeys(("주황", "주황색"), "orange"),
    **dict.fromkeys(("노란", "노랑", "노란색", "노랑색"), "yellow"),
    **dict.fromkeys(("초록", "초록색", "녹색"), "green"),
    **dict.fromkeys(("파란", "파랑", "파란색", "파랑색", "하늘색", "남색"), "blue"),
    **dict.fromkeys(("보라", "보라색", "보랏빛"), "purple"),
    **dict.fromkeys(("분홍", "분홍색", "핑크", "핑크색"), "pink"),
    **dict.fromkeys(("흰", "흰색", "하얀", "하양", "하얀색", "하양색"), "white"),
    **dict.fromkeys(("검은", "검정", "검은색", "검정색", "까만", "까만색"), "black"),
    "회색": "gray",
}
POSITION_WORDS: dict[str, str] = {
    "왼쪽": "left",
    "오른쪽": "right",
    "가운데": "center",
    "중간": "center",
    "위": "top",
    "위쪽": "top",
    "아래": "bottom",
    "아래쪽": "bottom",
    "밑": "bottom",
}
SIZE_WORDS: dict[str, str] = {"큰": "big", "커다란": "big", "작은": "small"}
# 다른 물건을 기준으로 삼는 말. "컵 옆에 있는 거"의 컵은 답이 아니라 기준점이라,
# 종류로 읽으면 정반대 물건을 고른다 → 말 단서는 기권한다(10/22 결정).
RELATION_WORDS: dict[str, str] = dict.fromkeys(("옆", "앞", "뒤", "사이", "근처"), "relation")
# 낱말 끝에서 떼어 볼 조사. 긴 것부터 뗀다("에서"를 "서"로 남기지 않게).
PARTICLES = ("에서", "이랑", "으로", "하고", "은", "는", "이", "가", "을", "를", "에", "의", "도")
PARTICLES += ("만", "랑", "로")
# 색 범위(OpenCV HSV: 색상 0~179 · 채도 · 밝기 0~255). 색마다 (색상 낮음, 채도 낮음, 밝기 낮음,
# 색상 높음, 채도 높음, 밝기 높음) 구간들이다. 표준 색상환(빨강 0 · 180, 주황 15, 노랑 30, 초록 60,
# 파랑 110, 보라 140 근처)을 나눈 출발점이고, 평가셋(s04)을 보며 고치지 않는다 — 보면서 맞추면
# 말 단서 숫자가 실력보다 부풀려진다. 흰 · 검정 · 회색은 색상과 상관없이 채도 · 밝기로만 가른다.
COLOR_RANGES: dict[str, tuple[tuple[int, int, int, int, int, int], ...]] = {
    "red": ((0, 70, 50, 9, 255, 255), (170, 70, 50, 179, 255, 255)),
    "orange": ((10, 70, 50, 21, 255, 255),),
    "yellow": ((22, 70, 50, 34, 255, 255),),
    "green": ((35, 50, 40, 85, 255, 255),),
    "blue": ((86, 50, 40, 130, 255, 255),),
    "purple": ((131, 50, 40, 155, 255, 255),),
    # 자홍 쪽 색상 + 채도가 낮고 밝은 빨강(연분홍)
    "pink": ((156, 40, 80, 169, 255, 255), (0, 30, 150, 9, 69, 255), (170, 30, 150, 179, 69, 255)),
    "white": ((0, 0, 180, 179, 40, 255),),
    "black": ((0, 0, 0, 179, 255, 49),),
    "gray": ((0, 0, 50, 179, 40, 179),),
}
# 색은 박스 안쪽만 본다. 사방에서 이만큼씩 덜어 낸다. YOLO 박스 가장자리에는 책상 · 벽이 낀다.
COLOR_INSET = 0.2

# --- 이어 묻기 단서(v5)의 낱말표 -----------------------------------------------------
# 바로 전에 말한 물건을 되가리키는 말. "그거 씻어야 돼?"의 "그거". 낱말 단위로 보므로
# "그런데" · "그래서"는 걸리지 않는다. "이거" · "저거"는 눈앞의 물건이라 넣지 않는다.
BACK_WORDS: dict[str, str] = dict.fromkeys(
    ("그", "그거", "그것", "그건", "그게", "그걸", "걔", "아까", "방금"), "back"
)


def score_center(candidates: list[Detection], frame_size: tuple[int, int]) -> list[CueScore]:
    """화면 가운데에 가깝고 크게 찍힌 물체일수록 높은 점수. 후보와 같은 순서로 돌려준다.

    안경 카메라는 머리 방향을 따라가서, 보고 있는 물건이 화면 가운데에 온다.
    크기는 비슷한 거리의 두 물건 중 가까운 쪽을 가르는 보조다.
    다른 단서가 없을 때의 기본값이고, ablation 표의 첫 줄(baseline)이다.
    """
    width, height = frame_size
    half_w, half_h = width / 2, height / 2
    scores = []
    for d in candidates:
        cx, cy = d.bbox.center
        # 가로·세로를 각각 -1~1 로 맞춘 거리. 구석이 √2 라서 나눠 0~1 로 만든다.
        distance = math.hypot((cx - half_w) / half_w, (cy - half_h) / half_h) / math.sqrt(2)
        closeness = 1.0 - distance
        # 넓이 비율은 대개 0.01~0.1 이라 그대로 쓰면 0 근처에 몰린다. 제곱근(변의 비율)으로 편다.
        size = math.sqrt(d.bbox.area / (width * height))
        score = CENTER_WEIGHT * closeness + (1.0 - CENTER_WEIGHT) * size
        scores.append(
            CueScore(
                kind=CueKind.CENTER,
                score=min(1.0, max(0.0, score)),
                detail={"closeness": closeness, "size": size},
            )
        )
    return scores


def score_point(candidates: list[Detection], ray: Ray) -> list[CueScore]:
    """검지 끝에서 뻗은 선이 후보 중심을 향할수록 높은 점수. 후보와 같은 순서로 돌려준다.

    선과 '검지 끝 → 후보 중심' 사이 각도를 종 모양(가우시안)으로 바꾼다. 0° 면 1, 손 뒤쪽은 0 쪽.
    코사인은 30° 에서도 0.87 이라 후보끼리 차이가 안 난다. 선이 박스를 지나는지로 재면 책상의
    큰 박스(노트북 · 모니터)가 겹쳐 비긴다(valid 10/62, 중심 각도는 13/62).
    후보 중심이 검지 끝과 같은 자리면 손가락이 그 물건 위에 있는 것이라 1점이다.
    """
    scores = []
    for d in candidates:
        angle = angle_to(ray, d.bbox.center)
        score = 1.0 if angle is None else math.exp(-0.5 * (angle / POINT_SIGMA_DEG) ** 2)
        scores.append(CueScore(kind=CueKind.POINT, score=score, detail={"angle": angle}))
    return scores


@dataclass(frozen=True)
class LangQuery:
    """질문에서 읽은 말 단서. 알아들은 것만 채운다."""

    kinds: frozenset[str] = frozenset()  # 검출 이름들("이 컵" → cup)
    colors: tuple[str, ...] = ()  # 영어 색 이름, 말한 순서
    position: str | None = None  # left · right · center · top · bottom
    size: str | None = None  # big · small
    relation: bool = False  # 다른 물건을 기준으로 삼는 말("컵 옆에")

    @property
    def empty(self) -> bool:
        return not (self.kinds or self.colors or self.position or self.size)


def _lookup[T](token: str, table: dict[str, T]) -> T | None:
    """낱말을 표에서 찾는다. 그대로 없으면 끝의 조사를 한 번 떼어 다시 찾는다.

    그대로 먼저 보는 이유: "작은" · "검은"의 "은"이 조사와 같아서, 먼저 떼면 "작" · "검"이 된다.
    """
    if token in table:
        return table[token]
    for particle in PARTICLES:
        if len(token) > len(particle) and token.endswith(particle):
            return table.get(token[: -len(particle)])
    return None


def parse_lang(text: str) -> LangQuery:
    """질문 원문에서 색 · 종류 · 위치 · 크기 말과 관계 말을 읽는다.

    글자 포함이 아니라 낱말(띄어쓰기) 단위로 본다. "책상"에 "책"이, "컵라면"에 "컵"이 들어 있어서
    글자로 찾으면 "책상 위 컵"이 책이 된다. 위 · 아래는 바로 앞 낱말이 종류 이름이면 관계
    ("컵 위에 있는 거"), 아니면 위치("위에 있는 병")다.
    """
    kinds: set[str] = set()
    colors: list[str] = []
    position = size = None
    relation = previous_kind = False
    for token in re.findall(r"[^\s?!.,~]+", text):
        if (labels := _lookup(token, KIND_WORDS)) is not None:
            kinds |= labels
            previous_kind = True
            continue
        if (color := _lookup(token, COLOR_WORDS)) is not None:
            if color not in colors:
                colors.append(color)
        elif _lookup(token, RELATION_WORDS) is not None:
            relation = True
        elif (place := _lookup(token, POSITION_WORDS)) is not None:
            if previous_kind and place in ("top", "bottom"):
                relation = True
            else:
                position = place
        elif (big := _lookup(token, SIZE_WORDS)) is not None:
            size = big
        previous_kind = False
    return LangQuery(frozenset(kinds), tuple(colors), position, size, relation)


def _color_share(hsv: np.ndarray, bbox: BBox, color: str) -> float:
    """박스 안쪽 픽셀 중 그 색 범위에 드는 비율(0~1). 안쪽이 비면 0."""
    height, width = hsv.shape[:2]
    dx, dy = bbox.width * COLOR_INSET, bbox.height * COLOR_INSET
    x1, y1 = max(0, round(bbox.x1 + dx)), max(0, round(bbox.y1 + dy))
    x2, y2 = min(width, round(bbox.x2 - dx)), min(height, round(bbox.y2 - dy))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = hsv[y1:y2, x1:x2]
    mask = np.zeros(region.shape[:2], np.uint8)
    for h1, s1, v1, h2, s2, v2 in COLOR_RANGES[color]:
        mask |= cv2.inRange(region, (h1, s1, v1), (h2, s2, v2))
    return float(np.count_nonzero(mask)) / mask.size


def _spread(values: list[float], high_wins: bool) -> list[float]:
    """값을 무리 안에서 0~1 로 편다. 무리가 하나뿐이거나 값이 다 같으면 모두 1."""
    low, top = min(values), max(values)
    if top == low:
        return [1.0] * len(values)
    return [((v - low) if high_wins else (top - v)) / (top - low) for v in values]


def _middle(values: list[float]) -> list[float]:
    """무리의 가운데(최솟값과 최댓값의 중간)에 가까울수록 1."""
    low, top = min(values), max(values)
    if top == low:
        return [1.0] * len(values)
    mid, half = (low + top) / 2, (top - low) / 2
    return [1.0 - abs(v - mid) / half for v in values]


def _relative(pool: list[Detection], word: str) -> list[float]:
    """위치 · 크기 낱말을 무리 안의 순위 점수로. "왼쪽" = 무리에서 가장 왼쪽이 1."""
    xs = [d.bbox.center[0] for d in pool]
    ys = [d.bbox.center[1] for d in pool]
    areas = [d.bbox.area for d in pool]
    by_word = {
        "left": lambda: _spread(xs, high_wins=False),
        "right": lambda: _spread(xs, high_wins=True),
        "center": lambda: _middle(xs),
        "top": lambda: _spread(ys, high_wins=False),
        "bottom": lambda: _spread(ys, high_wins=True),
        "big": lambda: _spread(areas, high_wins=True),
        "small": lambda: _spread(areas, high_wins=False),
    }
    return by_word[word]()


def score_lang(
    candidates: list[Detection],
    query: LangQuery,
    frame_size: tuple[int, int],
    image: np.ndarray | None = None,
) -> list[CueScore] | None:
    """질문 속 말이 맞는 후보일수록 높은 점수. 후보와 같은 순서로 돌려주고, 기권이면 None.

    속성(종류 · 색 · 위치 · 크기)마다 0~1 을 매겨 곱한다 — "왼쪽 컵"은 컵 중 왼쪽이어야 한다.
    평균이면 왼쪽 병과 오른쪽 컵이 비긴다. 위치 · 크기는 종류가 맞는 후보들(없으면 전체) 안에서
    순위를 매긴다. 색은 BGR 사진의 박스 안쪽을 HSV 로 바꿔 그 색 범위에 드는 픽셀 비율이다.

    기권(None): 관계 말이 있을 때, 쓸 수 있는 말이 없을 때(색만 말했는데 사진이 없음 포함),
    모든 후보 점수가 같을 때. 0점 대신 기권하는 이유: 모두에게 같은 점수를 주면 총점 차이(margin)가
    줄어 되묻기만 는다(10/22 결정). frame_size 는 다른 단서와 같은 모양으로 받는다.
    """
    use_color = bool(query.colors) and image is not None
    if query.relation or not candidates:
        return None
    if not (query.kinds or use_color or query.position or query.size):
        return None
    parts: list[dict[str, float]] = [{} for _ in candidates]
    if query.kinds:
        for part, d in zip(parts, candidates, strict=True):
            part["kind"] = 1.0 if d.label in query.kinds else 0.0
    if use_color:
        # OpenCV 사진은 BGR 이다. RGB 로 보고 바꾸면 파랑 · 빨강이 뒤바뀐다.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for part, d in zip(parts, candidates, strict=True):
            part["color"] = max(_color_share(hsv, d.bbox, c) for c in query.colors)
    pool = [i for i, d in enumerate(candidates) if d.label in query.kinds]
    if not pool:
        pool = list(range(len(candidates)))
    for key, word in (("position", query.position), ("size", query.size)):
        if word is None:
            continue
        ranked = dict(zip(pool, _relative([candidates[i] for i in pool], word), strict=True))
        for i, part in enumerate(parts):
            part[key] = ranked.get(i, 0.0)
    scores = [math.prod(part.values()) for part in parts]
    if max(scores) - min(scores) < 1e-9:
        return None
    return [
        CueScore(kind=CueKind.LANG, score=score, detail=part)
        for score, part in zip(scores, parts, strict=True)
    ]


def refers_back(text: str) -> bool:
    """질문에 바로 전 물건을 되가리키는 말(그거 · 아까 · 방금 …)이 있나. 낱말 단위로 본다."""
    return any(_lookup(token, BACK_WORDS) for token in re.findall(r"[^\s?!.,~]+", text))


def score_ctx(
    candidates: list[Detection], prev_label: str | None, text: str
) -> list[CueScore] | None:
    """이어 묻기에서 바로 전 대상과 같은 종류의 후보에 1, 아니면 0. 후보 순서대로, 기권이면 None.

    이름만 본다. 안경 카메라는 턴 사이에 고개를 따라 움직여서 앞 사진의 박스 자리는 뜻이 없다.
    되가리키는 말이 있을 때만 쓴다 — "이 병 뭐야?"처럼 새 물건을 물을 때 앞에서 물은 컵이 끼어들면
    틀린 물건을 밀어 준다. 기권(None): 직전 대상이 없을 때, 되가리키는 말이 없을 때, 후보를 못
    가를 때(맞는 후보가 없거나 모두 같은 이름). 0점 대신 기권하는 이유는 말 단서와 같다.
    """
    if prev_label is None or not candidates or not refers_back(text):
        return None
    scores = [1.0 if d.label == prev_label else 0.0 for d in candidates]
    if max(scores) == min(scores):
        return None
    return [
        CueScore(kind=CueKind.CTX, score=score, detail={"prev_label": prev_label})
        for score in scores
    ]
