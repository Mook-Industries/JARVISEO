"""검출 라벨의 한글 이름 (COCO 80종 + 학습한 product) 과 종류 낱말표.  담당: 최홍묵

되묻기 문장("왼쪽 컵이요, 오른쪽 컵이요?")과 대답을 받아쓸 때 주는 STT 힌트에 같이 쓴다.
TTS 로 읽고 사용자가 그대로 대답할 말이라 짧은 일상어로 골랐다.

boat 는 "배", dog 는 "개"가 아니다. 대답에서 이름을 찾을 때 "배낭"·"두 개" 안에 걸린다.
person 은 검출기가 후보에서 빼므로 넣지 않는다.

KIND_WORDS 는 사람이 부르는 종류 이름 → 검출 이름이다. 말 단서(질문 속 "이 컵") · 되묻기 대답 ·
라우터가 이 표 하나를 같이 쓴다. 따로 두면 "텀블러"를 한쪽만 알아듣는다(모의 루프 턴 35).
"""

from __future__ import annotations

__all__ = ["KIND_ALIASES", "KIND_WORDS", "KOREAN_NAMES", "korean_name"]

KOREAN_NAMES: dict[str, str] = {
    "bicycle": "자전거",
    "car": "자동차",
    "motorcycle": "오토바이",
    "airplane": "비행기",
    "bus": "버스",
    "train": "기차",
    "truck": "트럭",
    "boat": "보트",
    "traffic light": "신호등",
    "fire hydrant": "소화전",
    "stop sign": "정지 표지판",
    "parking meter": "주차 요금기",
    "bench": "벤치",
    "bird": "새",
    "cat": "고양이",
    "dog": "강아지",
    "horse": "말",
    "sheep": "양",
    "cow": "소",
    "elephant": "코끼리",
    "bear": "곰",
    "zebra": "얼룩말",
    "giraffe": "기린",
    "backpack": "배낭",
    "umbrella": "우산",
    "handbag": "핸드백",
    "tie": "넥타이",
    "suitcase": "여행 가방",
    "frisbee": "원반",
    "skis": "스키",
    "snowboard": "스노보드",
    "sports ball": "공",
    "kite": "연",
    "baseball bat": "야구 방망이",
    "baseball glove": "야구 글러브",
    "skateboard": "스케이트보드",
    "surfboard": "서핑보드",
    "tennis racket": "테니스 라켓",
    "bottle": "병",
    "wine glass": "와인잔",
    "cup": "컵",
    "fork": "포크",
    "knife": "칼",
    "spoon": "숟가락",
    "bowl": "그릇",
    "banana": "바나나",
    "apple": "사과",
    "sandwich": "샌드위치",
    "orange": "오렌지",
    "broccoli": "브로콜리",
    "carrot": "당근",
    "hot dog": "핫도그",
    "pizza": "피자",
    "donut": "도넛",
    "cake": "케이크",
    "chair": "의자",
    "couch": "소파",
    "potted plant": "화분",
    "bed": "침대",
    "dining table": "식탁",
    "toilet": "변기",
    "tv": "텔레비전",
    "laptop": "노트북",
    "mouse": "마우스",
    "remote": "리모컨",
    "keyboard": "키보드",
    "cell phone": "휴대폰",
    "microwave": "전자레인지",
    "oven": "오븐",
    "toaster": "토스터",
    "sink": "싱크대",
    "refrigerator": "냉장고",
    "book": "책",
    "clock": "시계",
    "vase": "꽃병",
    "scissors": "가위",
    "teddy bear": "곰 인형",
    "hair drier": "드라이기",
    "toothbrush": "칫솔",
    # COCO 에는 없고 학습한 가중치에만 있는 종류(FINETUNE_CLASSES, #167). 포장 식품 전체를 부른다.
    "product": "식품",
}

# 한글 이름(KOREAN_NAMES) 말고도 사람이 흔히 부르는 이름. 학습 종류 규칙(data/README.md)과 같게
# 모양으로 묶는다 — 텀블러는 병 모양이라 bottle, 햇반은 bowl, 캔 · 봉지 · 컵라면은 product.
# 사전학습 검출기는 캔을 cup · bottle 로 부르기도 하지만, 그 실수에 맞춰 넓히지 않는다(v4 결정 ②).
KIND_ALIASES: dict[str, frozenset[str]] = {
    "텀블러": frozenset({"bottle"}),
    "물병": frozenset({"bottle"}),
    "머그": frozenset({"cup"}),
    "종이컵": frozenset({"cup"}),
    "햇반": frozenset({"bowl"}),
    "폰": frozenset({"cell phone"}),
    "핸드폰": frozenset({"cell phone"}),
    "스마트폰": frozenset({"cell phone"}),
    "모니터": frozenset({"tv"}),
    "티비": frozenset({"tv"}),
    "공책": frozenset({"book"}),
    "캔": frozenset({"product"}),
    "봉지": frozenset({"product"}),
    "과자": frozenset({"product"}),
    "컵라면": frozenset({"product"}),
    "라면": frozenset({"product"}),
    "파우치": frozenset({"product"}),
    "통조림": frozenset({"product"}),
}

# 종류 낱말(띄어쓰기 없이) → 검출 이름들. 여러 낱말 이름("곰 인형")은 붙여 쓴 꼴로 둔다.
KIND_WORDS: dict[str, frozenset[str]] = {
    name.replace(" ", ""): frozenset({label}) for label, name in KOREAN_NAMES.items()
} | KIND_ALIASES


def korean_name(label: str) -> str:
    """한글 이름을 돌려준다. 표에 없으면 라벨을 그대로 돌려준다."""
    return KOREAN_NAMES.get(label, label)
