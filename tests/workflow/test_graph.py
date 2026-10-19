"""질문 한 번이 그래프를 처음부터 끝까지 도는지 확인한다.

STT·TTS·VLM·임베딩은 Fake 를 끼워서 API 키 없이 CI 에서 돈다.
"""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from jarviseo.capture import FrameBuffer
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
from jarviseo.vlm import FakeVLMClient, VLMClient
from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech, ResponseCollector
from jarviseo.voice.collector import FILLER
from jarviseo.workflow import GraphDeps, build_graph
from jarviseo.workflow.messages import RECAPTURE_GIVE_UP, REPROMPT, SORRY


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


def test_general_question_asks_with_photo_speaks_and_logs(file_store):
    """일반 질문은 사진과 함께 묻고 답을 읽고 기록한다."""
    tts = FakeTextToSpeech()
    vlm = FakeVLMClient(["빨간 컵이에요."])
    memory = VectorMemory(file_store, embedder=FakeEmbedder())
    dialog = DialogMemory(memory)
    frames = camera()
    graph = build_graph(
        GraphDeps(
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
    assert {"route", "stt", "vlm", "tts", "total"} <= set(result["latency_ms"])
    with file_store.session() as db:
        turn = db.get(SessionTurn, result["turn_id"])
        assert (turn.question_text, turn.answer_text) == ("이거 뭐야?", "빨간 컵이에요.")
        assert db.get(TurnVoice, turn.turn_id).stt_raw_text == "자비서, 이거 뭐야?"
        assert db.get(TurnInference, turn.turn_id).llm_ms == 0  # VLM 지연이 열에 들어갔다
        # 질문·대답이 기억에 들어갔다 (VLM 을 한 번 더 부르지 않는다)
        assert db.query(Observation).one().description == "질문: 이거 뭐야?\n대답: 빨간 컵이에요."


def test_calls_vlm_once_per_turn(file_store):
    """대답 한 번만 부른다. 기억에 넣으려고 장면 설명을 또 받지 않는다(#66)."""
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(output_text="검은 백팩이에요.")

    vlm = VLMClient(client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    dialog = DialogMemory(VectorMemory(file_store, embedder=FakeEmbedder()))
    deps = GraphDeps(
        stt=FakeSpeechToText(["이거 뭐야?", "저건 뭐야?"]),
        tts=FakeTextToSpeech(),
        vlm=vlm,
        frames=camera(),
        store=file_store,
        memory=dialog.memory,
        dialog=dialog,
    )
    graph = build_graph(deps)
    turn = turn_input(file_store)

    graph.invoke(turn)
    graph.invoke(turn)
    dialog.close()  # 뒤에서 하는 저장까지 끝난 뒤에 센다

    assert len(requests) == 2
    with file_store.session() as db:
        assert db.query(Observation).count() == 2


def test_remembers_dialog_and_passes_it_as_evidence_later(file_store):
    """지난 대화를 기억해 두었다가 나중에 묻으면 근거로 넘긴다."""
    vlm = FakeVLMClient(["검은 백팩이에요.", "검은 백팩이었어요."])
    # 가짜 임베딩(글자 두 개 묶음) 점수는 실제와 달라서 문턱을 끈다
    dialog = DialogMemory(VectorMemory(file_store, embedder=FakeEmbedder(), min_score=0.0))
    deps = GraphDeps(
        stt=FakeSpeechToText(["이거 뭐야?", "아까 본 백팩 뭐였지?"]),
        tts=FakeTextToSpeech(),
        vlm=vlm,
        store=file_store,
        memory=dialog.memory,
        dialog=dialog,
        recall_enabled=True,
    )
    graph = build_graph(deps)
    turn = turn_input(file_store)

    graph.invoke(turn)
    dialog.close()  # 첫 턴의 대화가 기억에 다 들어간 뒤 묻는다
    result = graph.invoke(turn)

    assert result["intent"] is Intent.RECALL
    (hit,) = vlm.asked[-1][2]
    assert hit.text == "질문: 이거 뭐야?\n대답: 검은 백팩이에요."


@pytest.mark.parametrize(
    ("question", "vlm", "called_since"),
    [
        ("아까 본 거 뭐였지?", FakeVLMClient(["기억에 없어요."]), None),  # 기억을 물은 턴
        ("이거 뭐야?", FakeVLMClient([None]), None),  # VLM 실패 → 사과 문장
        ("이거 뭐야?", FakeVLMClient(["늦은 답"], delay=0.3), lambda t: True),  # 다시 불러 버린 답
    ],
    ids=["recall", "sorry", "dropped"],
)
def test_turn_without_content_is_not_remembered(file_store, question, vlm, called_since):
    """기억할 내용이 없는 턴은 기억에 넣지 않는다."""
    tts = FakeTextToSpeech()
    dialog = DialogMemory(VectorMemory(file_store, embedder=FakeEmbedder()))
    collector = ResponseCollector(tts, called_since=called_since) if called_since else None
    deps = GraphDeps(
        stt=FakeSpeechToText([question]),
        tts=tts,
        vlm=vlm,
        store=file_store,
        memory=dialog.memory,
        dialog=dialog,
        collector=collector,
        recall_enabled=True,
    )

    result = build_graph(deps).invoke(turn_input(file_store))
    dialog.close()

    assert "turn_id" in result  # 턴 기록은 남는다
    with file_store.session() as db:
        assert db.query(Observation).count() == 0


def test_canned_notice_turn_is_not_remembered(file_store):
    """미리 정한 안내만 읽은 턴은 기억에 넣지 않는다."""
    # 사진이 없어 재촬영을 포기했다. "화면이 흐려서…"는 기억할 내용이 아니다.
    tts = FakeTextToSpeech()
    dialog = DialogMemory(VectorMemory(file_store, embedder=FakeEmbedder()))
    deps = GraphDeps(
        stt=FakeSpeechToText(["이거 뭐야?"]),
        tts=tts,
        vlm=FakeVLMClient(),
        frames=FrameBuffer(),
        store=file_store,
        memory=dialog.memory,
        dialog=dialog,
    )

    result = build_graph(deps).invoke(turn_input(file_store))
    dialog.close()

    assert tts.spoken == [RECAPTURE_GIVE_UP]
    assert "turn_id" in result
    with file_store.session() as db:
        assert db.query(Observation).count() == 0


def test_recall_question_passes_memories_to_vlm(file_store):
    """기억을 묻는 질문은 기억을 찾아 VLM 에 넘긴다."""
    vlm = FakeVLMClient(["컵이에요.", "책상 위에 있었어요."])
    memory = VectorMemory(file_store, embedder=FakeEmbedder())
    deps = GraphDeps(
        stt=FakeSpeechToText(["이거 뭐야?", "아까 본 검은 백팩 어디 있었지?"]),
        tts=FakeTextToSpeech(),
        vlm=vlm,
        store=file_store,
        memory=memory,
        recall_enabled=True,
    )
    graph = build_graph(deps)
    first = turn_input(file_store)
    turn_id = graph.invoke(first)["turn_id"]
    memory.remember_observation(first["user_id"], turn_id, "책상 위에 검은 백팩이 있다")

    result = graph.invoke(first)

    assert result["intent"] is Intent.RECALL
    (hit,) = vlm.asked[-1][2]
    assert hit.text == "책상 위에 검은 백팩이 있다"
    assert result["response"].memories == [hit]


def test_recall_disabled_routes_to_general():
    """기억 기능이 꺼져 있으면 기억을 묻는 질문도 GENERAL 이다."""
    vlm = FakeVLMClient()
    stt = FakeSpeechToText(["아까 본 거 뭐였지?"])
    graph = build_graph(GraphDeps(stt=stt, tts=FakeTextToSpeech(), vlm=vlm, recall_enabled=False))

    assert graph.invoke({"user_id": 1})["intent"] is Intent.GENERAL
    assert vlm.asked[0][2] is None


def test_no_memory_store_asks_with_empty_memories():
    """기억 저장소가 없으면 빈 기억으로 묻는다."""
    vlm = FakeVLMClient()
    deps = GraphDeps(
        stt=FakeSpeechToText(["아까 본 거 뭐였지?"]),
        tts=FakeTextToSpeech(),
        vlm=vlm,
        recall_enabled=True,
    )

    build_graph(deps).invoke({"user_id": 1})

    assert vlm.asked[0][2] == []  # 찾았는데 없던 것으로 넘겨야 VLM 이 지어내지 않는다


@pytest.mark.parametrize("texts", [[], ["음."]])
def test_no_question_speaks_and_logs_nothing(file_store, texts):
    """질문이 없으면 아무것도 읽지 않고 기록도 남기지 않는다."""
    tts = FakeTextToSpeech()
    graph = build_graph(
        GraphDeps(stt=FakeSpeechToText(texts), tts=tts, vlm=FakeVLMClient(), store=file_store)
    )

    result = graph.invoke(turn_input(file_store))

    assert "turn_id" not in result
    assert tts.spoken == []
    with file_store.session() as db:
        assert db.query(SessionTurn).count() == 0


def test_wake_word_only_asks_and_listens_again():
    """호출어만 부르면 되묻고 다시 듣는다."""
    tts = FakeTextToSpeech()
    stt = FakeSpeechToText(["자비서.", "이거 뭐야?"])
    graph = build_graph(GraphDeps(stt=stt, tts=tts, vlm=FakeVLMClient(["컵이에요."])))

    result = graph.invoke({"wake_at": time.monotonic()})

    assert tts.spoken == [REPROMPT, "컵이에요."]
    assert result["utterance"].text == "이거 뭐야?"


def test_vlm_failure_reads_apology():
    """VLM 이 실패하면 사과 문장을 읽는다."""
    tts = FakeTextToSpeech()
    graph = build_graph(GraphDeps(stt=FakeSpeechToText(), tts=tts, vlm=FakeVLMClient([None])))

    graph.invoke({})

    assert tts.spoken == [SORRY]


def test_slow_answer_reads_filler_and_logs_it(file_store):
    """답이 늦으면 "잠깐만요"를 먼저 읽고 기록에 남긴다."""
    tts = FakeTextToSpeech()
    deps = GraphDeps(
        stt=FakeSpeechToText(),
        tts=tts,
        vlm=FakeVLMClient(["컵이에요."], delay=0.3),
        store=file_store,
        collector=ResponseCollector(tts, after_sec=0.05),
    )

    result = build_graph(deps).invoke(turn_input(file_store))

    assert tts.spoken == [FILLER, "컵이에요."]
    with file_store.session() as db:
        assert db.get(TurnVoice, result["turn_id"]).is_filler_sent is True


def test_call_while_waiting_drops_answer_and_logs_interrupted(file_store):
    """답을 기다리는 동안 다시 부르면 답을 버리고 끊긴 턴으로 남긴다."""
    tts = FakeTextToSpeech()
    deps = GraphDeps(
        stt=FakeSpeechToText(),
        tts=tts,
        vlm=FakeVLMClient(["늦은 답"], delay=0.3),
        store=file_store,
        collector=ResponseCollector(tts, called_since=lambda t: True),  # 바로 다시 불렀다
    )

    result = build_graph(deps).invoke(turn_input(file_store))

    assert result["interrupted"] is True  # 다음 턴은 BARGE_IN
    assert tts.spoken == []
    with file_store.session() as db:
        assert db.get(TurnVoice, result["turn_id"]).is_interrupted is True
        assert db.get(SessionTurn, result["turn_id"]).answer_text == ""
