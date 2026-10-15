"""질문 한 번이 그래프를 처음부터 끝까지 도는지 확인한다.

STT·TTS·VLM·임베딩은 Fake 를 끼워서 API 키 없이 CI 에서 돈다.
"""

import time

import numpy as np
import pytest

from jarviseo.capture import FrameBuffer
from jarviseo.graph import Parts, build_graph
from jarviseo.graph.pipeline import REPROMPT, SORRY
from jarviseo.memory import (
    DialogMemory,
    FakeEmbedder,
    MemoryStore,
    Observation,
    SessionTurn,
    TurnInference,
    TurnVoice,
    VectorMemory,
)
from jarviseo.types import Frame, Intent
from jarviseo.vlm import FakeVLMClient
from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech, ResponseCollector
from jarviseo.voice.collector import FILLER


@pytest.fixture
def file_store(tmp_path):
    """대화 기억은 다른 스레드에서 저장된다. 인메모리 SQLite 는 스레드마다 달라 파일로 둔다."""
    s = MemoryStore(database_url=f"sqlite:///{(tmp_path / 'graph.db').as_posix()}")
    s.init_schema()
    yield s
    s.engine.dispose()


def camera():
    """지금 찍은 사진 한 장이 든 버퍼."""
    frames = FrameBuffer()
    image = np.zeros((48, 64, 3), np.uint8)
    frames.push(Frame(image=image, timestamp=time.monotonic(), source_id="test"))
    return frames


def turn_input(store):
    """턴 정보(세션·사용자). 그래프를 부르는 쪽이 넣는 값이다."""
    user_id = store.ensure_user()
    return {"session_id": store.start_session(user_id), "user_id": user_id}


def test_일반_질문은_사진과_함께_묻고_답을_읽고_기록한다(file_store):
    tts = FakeTextToSpeech()
    vlm = FakeVLMClient(["빨간 컵이에요."])
    memory = VectorMemory(file_store, embedder=FakeEmbedder())
    dialog = DialogMemory(memory)
    frames = camera()
    graph = build_graph(
        Parts(
            stt=FakeSpeechToText(["자비서, 이거 뭐야?"]),
            tts=tts,
            vlm=vlm,
            frames=frames,
            store=file_store,
            memory=memory,
            dialog=dialog,
        )
    )

    result = graph.invoke(turn_input(file_store) | {"trigger_type": "WAKEWORD"})
    dialog.close()  # 뒤에서 하던 대화 기억 저장을 마친다

    assert result["intent"] is Intent.GENERAL
    assert tts.spoken == ["빨간 컵이에요."]
    ((frame, question, memories),) = vlm.asked
    assert frame is frames.pick_best()  # 말을 시작할 때 고른 사진을 보냈다
    assert question == "이거 뭐야?"  # 호출어는 빼고
    assert memories is None  # 기억을 안 찾은 질문
    with file_store.session() as db:
        turn = db.get(SessionTurn, result["turn_id"])
        assert (turn.question_text, turn.answer_text) == ("이거 뭐야?", "빨간 컵이에요.")
        assert db.get(TurnVoice, turn.turn_id).stt_raw_text == "자비서, 이거 뭐야?"
        assert db.get(TurnInference, turn.turn_id).llm_ms == 0  # VLM 지연이 열에 들어갔다
        # 질문·대답이 기억에 들어갔다 (VLM 을 한 번 더 부르지 않는다)
        assert db.query(Observation).one().description == "질문: 이거 뭐야?\n대답: 빨간 컵이에요."
    assert len(vlm.asked) == 1


def test_기억을_묻는_질문은_기억을_찾아_vlm_에_넘긴다(file_store):
    vlm = FakeVLMClient(["컵이에요.", "책상 위에 있었어요."])
    memory = VectorMemory(file_store, embedder=FakeEmbedder())
    parts = Parts(
        stt=FakeSpeechToText(["이거 뭐야?", "아까 본 검은 백팩 어디 있었지?"]),
        tts=FakeTextToSpeech(),
        vlm=vlm,
        store=file_store,
        memory=memory,
        recall_enabled=True,
    )
    graph = build_graph(parts)
    first = turn_input(file_store)
    turn_id = graph.invoke(first)["turn_id"]
    memory.remember_observation(first["user_id"], turn_id, "책상 위에 검은 백팩이 있다")

    result = graph.invoke(first)

    assert result["intent"] is Intent.RECALL
    (hit,) = vlm.asked[-1][2]
    assert hit.text == "책상 위에 검은 백팩이 있다"
    assert result["response"].memories == [hit]


def test_기억_기능이_꺼져_있으면_기억을_묻는_질문도_general():
    vlm = FakeVLMClient()
    stt = FakeSpeechToText(["아까 본 거 뭐였지?"])
    graph = build_graph(Parts(stt=stt, tts=FakeTextToSpeech(), vlm=vlm, recall_enabled=False))

    assert graph.invoke({"user_id": 1})["intent"] is Intent.GENERAL
    assert vlm.asked[0][2] is None


def test_질문이_없으면_아무것도_읽지_않고_기록도_남기지_않는다(file_store):
    tts = FakeTextToSpeech()
    graph = build_graph(
        Parts(stt=FakeSpeechToText([]), tts=tts, vlm=FakeVLMClient(), store=file_store)
    )

    result = graph.invoke(turn_input(file_store))

    assert "turn_id" not in result
    assert tts.spoken == []
    with file_store.session() as db:
        assert db.query(SessionTurn).count() == 0


def test_호출어만_부르면_되묻고_다시_듣는다():
    tts = FakeTextToSpeech()
    stt = FakeSpeechToText(["자비서.", "이거 뭐야?"])
    graph = build_graph(Parts(stt=stt, tts=tts, vlm=FakeVLMClient(["컵이에요."])))

    result = graph.invoke({"wake_at": time.monotonic()})

    assert tts.spoken == [REPROMPT, "컵이에요."]
    assert result["utterance"].text == "이거 뭐야?"


def test_vlm_이_실패하면_사과_문장을_읽는다():
    tts = FakeTextToSpeech()
    graph = build_graph(Parts(stt=FakeSpeechToText(), tts=tts, vlm=FakeVLMClient([None])))

    graph.invoke({})

    assert tts.spoken == [SORRY]


def test_답이_늦으면_잠깐만요를_먼저_읽고_기록에_남긴다(file_store):
    tts = FakeTextToSpeech()
    parts = Parts(
        stt=FakeSpeechToText(),
        tts=tts,
        vlm=FakeVLMClient(["컵이에요."], delay=0.3),
        store=file_store,
        collector=ResponseCollector(tts, after_sec=0.05),
    )

    result = build_graph(parts).invoke(turn_input(file_store))

    assert tts.spoken == [FILLER, "컵이에요."]
    with file_store.session() as db:
        assert db.get(TurnVoice, result["turn_id"]).is_filler_sent is True


def test_답을_기다리는_동안_다시_부르면_답을_버리고_끊긴_턴으로_남긴다(file_store):
    tts = FakeTextToSpeech()
    parts = Parts(
        stt=FakeSpeechToText(),
        tts=tts,
        vlm=FakeVLMClient(["늦은 답"], delay=0.3),
        store=file_store,
        collector=ResponseCollector(tts, called_since=lambda t: True),  # 바로 다시 불렀다
    )

    result = build_graph(parts).invoke(turn_input(file_store))

    assert result["interrupted"] is True  # 다음 턴은 BARGE_IN
    assert tts.spoken == []
    with file_store.session() as db:
        assert db.get(TurnVoice, result["turn_id"]).is_interrupted is True
        assert db.get(SessionTurn, result["turn_id"]).answer_text == ""
