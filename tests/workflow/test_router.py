"""키워드 라우터가 성분 → 기억 → 가리킴 → 나머지 순으로 의도를 정하는지 확인한다."""

import pytest

from jarviseo.types import Intent
from jarviseo.workflow.router import classify_intent


@pytest.mark.parametrize("text", ["이거 먹어도 돼?", "성분 알려줘", "이거 뭐 들어갔어?"])
def test_성분을_묻는_말은_지시어가_있어도_ingredient(text):
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
def test_지난_일을_묻는_말은_recall(text):
    assert classify_intent(text) is Intent.RECALL


@pytest.mark.parametrize("text", ["저거 뭐야?", "이게 뭐예요", "그거 얼마야?", "저것 좀 봐"])
def test_지시어가_있으면_pointing(text):
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
def test_그_밖의_말은_general(text):
    assert classify_intent(text) is Intent.GENERAL
