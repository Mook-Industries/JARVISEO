"""키워드 라우터가 성분 → 기억 → 가리킴 → 나머지 순으로 의도를 정하는지 확인한다.
손이 보여도 GENERAL 로 둘 장면 · 찾기 · 영역 질문을 가려내는지도 본다(#217).
"왼쪽 컵 뭐야?"처럼 고르는 말도 가리킴으로 보내는지 본다(#166).

새 사례는 가리킴 평가셋을 찍으며 적은 실제 발화(#147 · 말 · 맥락 세션 s04)에서 뽑았다.
"""

import pytest

from jarviseo.types import Intent
from jarviseo.workflow.router import asks_whole_scene, classify_intent


@pytest.mark.parametrize("text", ["이거 먹어도 돼?", "성분 알려줘", "이거 뭐 들어갔어?"])
def test_ingredient_words_win_over_demonstratives(text):
    """성분을 묻는 말은 지시어가 있어도 INGREDIENT 다."""
    assert classify_intent(text) is Intent.INGREDIENT


@pytest.mark.parametrize(
    "text",
    [
        "아까 본 그거 뭐였지?",  # 지시어가 섞여도 지난 일을 묻는 말
        "내 열쇠 어디 뒀지?",
        "내 열쇠 어디뒀지?",  # STT 가 붙여 쓴 경우
        "안경집 어디 뒀더라?",
        "차 키 어디 놨지?",
        "핸드폰 어디서 충전하고 있었지?",
        "가방 어디 있었더라?",
        "어제 본 우산 무슨 색이었지?",
        "전에 본 컵",
    ],
)
def test_past_questions_are_recall(text):
    """지난 일을 묻는 말은 RECALL 이다."""
    assert classify_intent(text) is Intent.RECALL


@pytest.mark.parametrize("text", ["저거 뭐야?", "이게 뭐예요", "그거 얼마야?", "저것 좀 봐"])
def test_demonstratives_are_pointing(text):
    """지시어가 있으면 POINTING 이다."""
    assert classify_intent(text) is Intent.POINTING


@pytest.mark.parametrize(
    "text",
    [
        "오늘 날씨 뭐야?",  # "뭐야"만으로는 가리킴이 아니다
        "내 가방 어디 있어?",  # 지금 찾는 말(10/7 결정)
        "충전하는 데 얼마나 걸려?",
        "",
    ],
)
def test_everything_else_is_general(text):
    """그 밖의 말은 GENERAL 이다."""
    assert classify_intent(text) is Intent.GENERAL


# --- #147 지시어 보강: 촬영 발화에서 놓쳤던 말 ------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "이건 뭐야?",
        "이건뭐야?",  # STT 가 붙여 쓴 경우
        "쟤는 뭐야?",
        "이걸 뭐라고 불러?",
        "내가 가리키는 거 뭐야?",
        "저기 오렌지 주스 얼마야?",
    ],
)
def test_more_demonstratives_are_pointing(text):
    """이건 · 쟤 · 이걸 · 가리키 · 저기도 물건 하나를 집는 말이다."""
    assert classify_intent(text) is Intent.POINTING


@pytest.mark.parametrize(
    "text",
    [
        "왼쪽꺼 뭐야?",
        "왼쪽 꺼 뭐야?",
        "오른쪽꺼 뭐야?",
        "컵라면 위에 있는거 뭐야?",
        "옆에 있는 거 뭐야?",
    ],
)
def test_position_then_thing_is_pointing(text):
    """위치 말 바로 뒤에 거 · 꺼 · 것이 붙으면 물건 하나를 집는다."""
    assert classify_intent(text) is Intent.POINTING


@pytest.mark.parametrize("text", ["뭐야?", "뭐야", "뭐지?", "뭐예요?"])
def test_bare_question_is_pointing(text):
    """앞에 아무 말도 없는 "뭐야?"는 보고 있는 물건을 묻는다."""
    assert classify_intent(text) is Intent.POINTING


@pytest.mark.parametrize(
    "text",
    [
        "알러지 체크 해줘",
        "나 이거 알러지 있어?",  # 지시어가 있어도 성분이 먼저
        "이 과자에 우유 들어 있어?",
        "나트륨이 얼마나 들어 있어?",  # 영양성분도 성분 길(알레르기 질문이 VLM 으로 새지 않게)
    ],
)
def test_allergy_words_are_ingredient(text):
    """알러지 · 들어 있어도 성분 질문이다."""
    assert classify_intent(text) is Intent.INGREDIENT


@pytest.mark.parametrize(
    "text",
    [
        "저기요 오늘 날씨 어때?",  # 부르는 말
        "그건 그렇고 내일 비 와?",  # 말 잇기
        "얘기 좀 해줘",
        "이 근처 맛집 알려줘",
        "앞으로 할 거 알려줘",  # 위치 말 뒤에 바로 '거'가 붙지 않음
        "위험한 거 있어?",
        "오늘 비 올 거야?",
        "길이 재는 법 알려줘",
        "어떤게 치폴레 소스지?",  # 찾기 질문은 VLM 이 전체 사진에서 읽는다
        "둘중 어떤게 시오다래 소스야?",
        "왼쪽에는 뭐가 있지?",  # 영역 질문
        "뭐가 보여?",  # 장면 질문
        "가장 큰 물건이 뭐야?",
    ],
)
def test_lookalikes_stay_general(text):
    """새 규칙과 비슷해 보여도 물건 하나를 집지 않는 말은 GENERAL 이다."""
    assert classify_intent(text) is Intent.GENERAL


# --- 손 신호에서 빼는 질문 (#217) -----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "어떤게 치폴레 소스지?",  # 찾기
        "둘중 어떤게 시오다래 소스야?",
        "가장 큰 물건이 뭐야?",  # 비교
        "이 중에 제일 싼 거?",
        "뭐가 보여?",  # 장면
        "지금 앞에 뭐가 있어?",
        "여기 어디야?",
        "왼쪽에는 뭐가 있지?",  # 영역
        "오른쪽 칸에는 뭐가 있지?",
        "위엔 뭐 있어?",
    ],
)
def test_whole_scene_questions_are_flagged(text):
    """찾기 · 비교 · 장면 · 영역 질문은 사진 전체가 필요하다. 손이 보여도 크롭하지 않는다."""
    assert asks_whole_scene(text)


@pytest.mark.parametrize(
    "text",
    [
        "이 약 하루에 몇 번 먹어야 돼?",  # 손 신호가 노리는 물건 질문
        "자비서, 이 글자 읽어 줘.",
        "바나프레소 컵에 뭐라고 적혀 있어?",
        "갈아만든 배 캔이 몇 밀리리터야?",
        "이 근처 맛집 알려줘",  # 손이 없으면 GENERAL 그대로라 여기서 막을 필요 없다
        "오늘 날씨 어때?",
    ],
)
def test_single_object_questions_are_not_flagged(text):
    """물건 하나를 묻거나 사진과 상관없는 질문은 걸리지 않는다. 손이 있을 때만 가리킴으로 간다."""
    assert not asks_whole_scene(text)


# --- 고르는 말 (#166) -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "이 병 뭐야?",  # 이 · 그 · 저 + 종류
        "이 텀블러 뭐야?",  # 별칭(bottle)
        "그 컵 씻어야 돼?",
        "흰색 컵 뭐야?",  # 색 · 위치 · 크기 + 종류
        "왼쪽 컵 뭐야?",
        "큰 병 뭐야?",
        "작은 거 뭐야?",  # 속성 + 거
        "위에 있는 병 뭐야?",  # 위치 + 있는 + 종류
        "보라색 뭐야?",  # 고르는 말만 + 단독 질문
        "컵 뭐야?",
    ],
)
def test_picking_words_are_pointing(text):
    """고르는 말로 물건 하나를 집는 질문은 가리킴이다(s04 에서 뽑은 꼴).

    말 단서가 그 말을 읽는다.
    """
    assert classify_intent(text) is Intent.POINTING


@pytest.mark.parametrize(
    "text",
    [
        "책 추천해 줘",  # 종류 낱말이 있어도 물건을 집지 않는다
        "컵라면 맛있게 끓이는 법 알려줘",
        "작은 방 정리하는 법 알려줘",  # 크기 말 뒤가 종류 · 거가 아님
        "오늘 큰 비 와?",
        "왼쪽으로 가야 돼?",
        "이 근처 맛집 알려줘",  # 근처는 관계 말
        "이 병원 몇 시까지 해?",  # 띄어쓰기를 지우면 "이병"에 걸린다
        "가장 큰 물건이 뭐야?",  # 비교 질문
        "오른쪽 칸에는 뭐가 있지?",  # 영역 질문
    ],
)
def test_words_that_do_not_pick_one_stay_general(text):
    """고르는 말이 섞여도 물건 하나를 집지 않는 질문은 GENERAL 이다."""
    assert classify_intent(text) is Intent.GENERAL


def test_joined_words_are_not_read():
    """붙여 쓴 "왼쪽컵"은 못 잡는다. 말 단서(parse_lang)도 낱말로 읽어서 둘이 어긋나지 않는다."""
    assert classify_intent("왼쪽컵 뭐야?") is Intent.GENERAL


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("노란 봉지 먹어도 돼?", Intent.INGREDIENT),
        ("아까 본 노란 컵 뭐였지?", Intent.RECALL),
    ],
)
def test_ingredient_and_recall_still_come_first(text, intent):
    """고르는 말이 있어도 성분 · 기억 낱말이 먼저다."""
    assert classify_intent(text) is intent


@pytest.mark.parametrize(
    "text",
    [
        "방금 물어봤던거 뭐지?",
        "방금 뭐라고 했지?",
        "방금 그거 뭐였지?",
        "마지막으로 물어본 게 뭐였지?",
    ],
)
def test_recent_dialog_questions_are_recall(text):
    """지난 대화 자체를 물으면 기억으로 보낸다. 전에는 GENERAL · POINTING 으로 빠졌다."""
    assert classify_intent(text) is Intent.RECALL


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("방금 들어온 거 뭐야?", Intent.GENERAL),
        ("방금 질문 하나 할게", Intent.GENERAL),
        ("그거 뭐야?", Intent.POINTING),
    ],
)
def test_recent_words_alone_are_not_recall(text, intent):
    """방금만 있거나(대화를 묻지 않음) 때 없이 그거만 있으면 기억이 아니다."""
    assert classify_intent(text) is intent
