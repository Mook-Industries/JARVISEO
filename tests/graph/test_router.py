"""규칙 라우터가 기억을 묻는 질문만 RECALL 로 보내는지 확인한다."""

import pytest

from jarviseo.graph.router import route
from jarviseo.types import Intent


@pytest.mark.parametrize(
    "text",
    [
        "아까 본 가방 어디 있었지?",
        "조금 전에 본 책 제목이 뭐였지?",
        "어제 본 거 뭐였더라?",
        "내 열쇠 어디 뒀지?",
        "지난번에 먹은 과자 이름 기억나?",
    ],
)
def test_지난_일을_묻는_말은_recall(text):
    assert route(text) is Intent.RECALL


@pytest.mark.parametrize(
    "text",
    [
        "이거 뭐야?",
        "저거 뭐야?",
        "이거 뭐 들어갔어?",  # 성분 노드가 붙기 전까지는 GENERAL
        "내 가방 어디 있어?",  # 지금 보이는 것을 찾는 말
        "충전하는 데 얼마나 걸려?",
        "오늘 날씨 어때?",
    ],
)
def test_그_밖의_말은_general(text):
    assert route(text) is Intent.GENERAL
