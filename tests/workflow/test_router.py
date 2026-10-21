"""키워드 라우터가 성분 → 기억 → 가리킴 → 나머지 순으로 의도를 정하는지 확인한다.

새 사례는 가리킴 평가셋을 찍으며 적은 실제 발화(#147)에서 뽑았다.
"""

import pytest

from jarviseo.types import Intent
from jarviseo.workflow.router import classify_intent


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
