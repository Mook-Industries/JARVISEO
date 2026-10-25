"""질문 한 번이 그래프를 처음부터 끝까지 도는지 확인한다.

STT·TTS·VLM·임베딩은 Fake 를 끼워서 API 키 없이 CI 에서 돈다.
"""

import threading
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
from jarviseo.workflow.messages import RECAPTURE_GIVE_UP, REPROMPT, SORRY, VLM_TOO_SLOW


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


def test_reprompt_plays_start_chime_before_listening_again():
    """호출어만 부르면 "네, 말씀하세요"를 읽은 뒤, 다시 듣기 전에 시작음을 낸다."""
    tts = FakeTextToSpeech()
    chimed: list[list[str]] = []
    deps = GraphDeps(
        stt=FakeSpeechToText(["자비서.", "이거 뭐야?"]),
        tts=tts,
        vlm=FakeVLMClient(["컵이에요."]),
        start_chime=lambda: chimed.append(list(tts.spoken)),
    )

    build_graph(deps).invoke({"wake_at": time.monotonic()})

    assert chimed == [[REPROMPT]]  # 되묻는 말 뒤, 대답 전에 한 번


def test_question_after_wake_word_has_no_extra_start_chime():
    """호출어에 이어 바로 질문하면 그래프는 시작음을 안 낸다(호출 때 assistant.py 가 냈다)."""
    chimed: list[str] = []
    deps = GraphDeps(
        stt=FakeSpeechToText(["자비서, 이거 뭐야?"]),
        tts=FakeTextToSpeech(),
        vlm=FakeVLMClient(["컵이에요."]),
        start_chime=lambda: chimed.append("start"),
    )

    build_graph(deps).invoke({"wake_at": time.monotonic()})

    assert chimed == []


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


# --- VLM 답 상한 -----------------------------------------------------------------


class _HeldVLM(FakeVLMClient):
    """release 가 설정될 때까지 답하지 않는 VLM. 답했으면 answered 가 설정된다.

    상한이 지나도 답이 안 온 상태를 sleep 길이(시간)에 기대지 않고 만든다. 걸린 시간으로 재면
    그래프 만들기 · DB 기록까지 같이 재져서, 느린 CI 러너에서 가끔 기준을 넘었다.
    """

    def __init__(self, answers) -> None:
        super().__init__(answers)
        self.release = threading.Event()
        self.answered = threading.Event()

    def ask(self, frame, question, memories=None):
        self.release.wait(5)  # 그래프가 답을 기다리면 여기서 5초 뒤에야 답한다
        try:
            return super().ask(frame, question, memories)
        finally:
            self.answered.set()


def test_slow_vlm_reads_notice_instead_of_waiting(file_store):
    """VLM 이 상한 안에 답하지 않으면 안내 문장을 읽고 끝낸다. 지연 · 기억은 남기지 않는다."""
    tts = FakeTextToSpeech()
    dialog = DialogMemory(VectorMemory(file_store, embedder=FakeEmbedder()))
    vlm = _HeldVLM(["늦은 답"])
    deps = GraphDeps(
        stt=FakeSpeechToText(["이거 뭐야?"]),
        tts=tts,
        vlm=vlm,
        store=file_store,
        memory=dialog.memory,
        dialog=dialog,
        vlm_deadline_sec=0.1,
    )

    result = build_graph(deps).invoke(turn_input(file_store))
    answered_before_end = vlm.answered.is_set()
    vlm.release.set()  # 늦은 답이 이제야 온다
    assert vlm.answered.wait(2)
    dialog.close()

    assert tts.spoken == [VLM_TOO_SLOW]
    assert not answered_before_end  # 답이 오기 전에 끝냈다(늦은 답을 기다리지 않았다)
    assert "vlm" not in result["latency_ms"]  # 왕복을 모르는 요청의 시간은 남기지 않는다
    with file_store.session() as db:
        assert db.query(Observation).count() == 0  # 늦게 온 답도 기억에 안 들어간다


def test_answer_within_deadline_is_read():
    """상한 안에 온 답은 지금처럼 읽고 지연도 남긴다."""
    tts = FakeTextToSpeech()
    deps = GraphDeps(
        stt=FakeSpeechToText(["이거 뭐야?"]),
        tts=tts,
        vlm=FakeVLMClient(["컵이에요."], delay=0.05),
        vlm_deadline_sec=1.0,
    )

    result = build_graph(deps).invoke({})

    assert tts.spoken == ["컵이에요."]
    assert "vlm" in result["latency_ms"]


def test_filler_then_notice_when_answer_is_too_slow():
    """필러("잠깐만요")를 읽은 뒤에도 상한까지 답이 없으면 안내 문장으로 끝낸다."""
    tts = FakeTextToSpeech()
    deps = GraphDeps(
        stt=FakeSpeechToText(["이거 뭐야?"]),
        tts=tts,
        vlm=FakeVLMClient(["늦은 답"], delay=0.5),
        collector=ResponseCollector(tts, after_sec=0.05),
        vlm_deadline_sec=0.2,
    )

    result = build_graph(deps).invoke({})

    assert tts.spoken == [FILLER, VLM_TOO_SLOW]
    assert result["filler_sent"] is True
    assert "늦은 답" not in tts.prefetched  # 버린 답은 미리 합성하지 않는다


class FakeWakeCheck:
    """호출어 2단계 확인 대신. 정해 둔 받아쓰기 결과로 통과·거절한다(None 은 확인 못 함 = 통과)."""

    def __init__(self, text):
        self.text, self.started = text, []

    def start(self, t):
        self.started.append(t)
        rejected = self.text is not None and "자비서" not in self.text
        return SimpleNamespace(
            text=self.text, rejected=lambda: rejected, passed=lambda: not rejected
        )


class LateWakeVerdict:
    """질문 듣기보다 늦게 끝나는 확인. WakeVerdict 처럼 passed() 가 기다리기 전에는 글자가 없다."""

    def __init__(self, text):
        self._text, self._done = text, False

    @property
    def text(self):
        return self._text if self._done else None

    def rejected(self):
        return self.text is not None and "자비서" not in self.text

    def passed(self):
        self._done = True  # 결과가 나올 때까지 기다렸다
        return not self.rejected()


@pytest.mark.parametrize("texts", [["오늘 서울은 맑겠습니다."], []])
def test_rejected_wake_speaks_nothing_and_logs_nothing(file_store, texts):
    """2단계 확인에서 호출어가 아니면 되묻기·대답 없이 끝나고 기록도 남기지 않는다."""
    tts = FakeTextToSpeech()
    vlm = FakeVLMClient(["맑아요."])
    check = FakeWakeCheck("오늘 서울은")
    graph = build_graph(
        GraphDeps(stt=FakeSpeechToText(texts), tts=tts, vlm=vlm, store=file_store, wake_check=check)
    )

    wake_at = time.monotonic()
    result = graph.invoke(turn_input(file_store) | {"wake_at": wake_at})

    assert result["wake_rejected"] is True
    assert result["wake_check_text"] == "오늘 서울은"
    assert check.started == [wake_at]
    assert tts.spoken == []  # REPROMPT 도 읽지 않는다
    assert vlm.asked == []
    with file_store.session() as db:
        assert db.query(SessionTurn).count() == 0


@pytest.mark.parametrize("text", ["자비서.", None])
def test_confirmed_wake_answers_as_before(text):
    """호출어가 확인됐거나 확인하지 못했으면(받아쓰기 실패) 지금처럼 질문을 받는다."""
    tts = FakeTextToSpeech()
    graph = build_graph(
        GraphDeps(
            stt=FakeSpeechToText(["이거 뭐야?"]),
            tts=tts,
            vlm=FakeVLMClient(["컵이에요."]),
            wake_check=FakeWakeCheck(text),
        )
    )

    result = graph.invoke({"wake_at": time.monotonic()})

    assert "wake_rejected" not in result
    assert result["wake_check_text"] == text  # 통과한 턴에도 무엇을 받아썼는지 남는다
    assert tts.spoken == ["컵이에요."]


def test_confirmed_wake_without_question_keeps_check_text():
    """호출어만 부르고 질문이 없던 턴도 확인에서 받아쓴 글자를 남긴다."""
    tts = FakeTextToSpeech()
    graph = build_graph(
        GraphDeps(
            stt=FakeSpeechToText([]),
            tts=tts,
            vlm=FakeVLMClient(),
            wake_check=FakeWakeCheck("자비서."),
        )
    )

    result = graph.invoke({"wake_at": time.monotonic()})

    assert result["wake_check_text"] == "자비서."
    assert tts.spoken == [REPROMPT]


@pytest.mark.parametrize("text, rejected", [("오늘 서울은", True), ("자비서.", False)])
def test_late_check_still_records_check_text(text, rejected):
    """확인 받아쓰기가 질문 듣기보다 늦게 끝나도, 기다려서 받은 글자를 남긴다.

    전에는 기다리기 전에 글자를 읽어서 거른 호출이 'None'으로 찍혔다(TV 실측 session 14).
    """
    tts = FakeTextToSpeech()
    graph = build_graph(
        GraphDeps(
            stt=FakeSpeechToText(["이거 뭐야?"]),
            tts=tts,
            vlm=FakeVLMClient(["컵이에요."]),
            wake_check=SimpleNamespace(start=lambda t: LateWakeVerdict(text)),
        )
    )

    result = graph.invoke({"wake_at": time.monotonic()})

    assert result["wake_check_text"] == text
    assert result.get("wake_rejected", False) is rejected
    assert tts.spoken == ([] if rejected else ["컵이에요."])


def test_wake_check_skipped_without_wake_time():
    """호출어 감지 시각이 없는 턴(테스트·직접 부름)은 확인하지 않는다."""
    check = FakeWakeCheck("오늘 서울은")
    graph = build_graph(
        GraphDeps(
            stt=FakeSpeechToText(["이거 뭐야?"]),
            tts=FakeTextToSpeech(),
            vlm=FakeVLMClient(["컵이에요."]),
            wake_check=check,
        )
    )

    result = graph.invoke({})

    assert check.started == []
    assert result["response"].text == "컵이에요."
