"""끼어드는 때마다 턴이 어떻게 나뉘고 total 이 얼마인지 재현한다 (#105).  담당: 문태현

실제 그래프(build_graph)·BargeIn·대상 고르기(resolve_target)·SQLite 기록을 그대로 쓰고,
STT·TTS·VLM 만 시간이 흐르는 가짜로 바꾼다. 턴을 도는 방식은 scripts/assistant.py 의 run() 과 같다.
마이크·카메라·API 키 없이 돈다. 시간은 실제 초라서 다 돌면 30초쯤 걸린다.

장면 (처음 질문은 모두 "자비서, 이거 뭐야?")
    answer        대답을 읽는 중에 "자비서, 이건 뭐야?"
    clarify       컵 둘 중 어느 것인지 되묻는 중에 다시 부름
    clarify_wait  되묻기를 다 읽고 대답을 기다리는 중에 다시 부름
    clarify_edge  되묻는 말 끝에 걸쳐 부름. 감지가 재생이 끝난 뒤(0.2초)에 들어온다
    clarify_only  끼어들기 없이 되묻기만 (대답 없음, 비교용)

턴마다 DB 에 남은 trigger_type · 질문 · 대답 · 끊김 · 되묻기 · total_ms 를 찍고,
끼어든 턴은 "다시 부르고 다음 턴이 시작되기까지"를 같이 찍는다.
고치기 전(#105)에는 clarify·clarify_wait 에서 새 질문이 되묻기 대답으로 들어가 따로 턴이 안 생겼고,
그 시간이 앞 턴 total 에 붙었다. 대답을 기다리는 중 다시 부르면 그 말을 다 받아쓴 뒤에야(실제로는
"자비서"만 부르고 기다리면 5초를 다 채운 뒤에야) 다음 턴이 시작됐다.

실행:
    python scripts/check_barge_turns.py                 # 장면 넷 다
    python scripts/check_barge_turns.py clarify answer  # 고른 장면만
"""

import argparse
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from jarviseo import config
from jarviseo.capture import FrameBuffer
from jarviseo.memory import MemoryStore, SessionTurn
from jarviseo.pointing import resolve_target
from jarviseo.types import BBox, Detection, Frame, Utterance
from jarviseo.vlm import FakeVLMClient
from jarviseo.voice import BargeIn
from jarviseo.voice.transcript import MIN_LETTERS, clean, is_question
from jarviseo.workflow import GraphDeps, build_graph

FIRST_QUESTION = "자비서, 이거 뭐야?"
NEW_QUESTION = "자비서, 이건 뭐야?"
ANSWER = "하얀 컵이에요. 손잡이가 달려 있어요."
CUP = Detection("cup", 0.9, BBox(400, 500, 700, 900))
TWO_CUPS = [CUP, Detection("cup", 0.9, BBox(1200, 500, 1500, 900))]

# 사람이 끼어드는 때: (이 글자가 든 문장을 읽기 시작하고, 몇 초 뒤).
# 새 질문의 "자비서"가 그때 끝난다.
SCENES = {
    "answer": ("대답 읽는 중 끼어들기", [CUP], ("하얀 컵", 1.0)),
    "clarify": ("되묻기 읽는 중 끼어들기", TWO_CUPS, ("왼쪽 컵이요", 0.6)),
    "clarify_wait": ("되묻기 대답 기다리는 중 다시 부름", TWO_CUPS, ("왼쪽 컵이요", 2.5)),
    # 되묻는 말("왼쪽 컵이요, 오른쪽 컵이요?" 16글자 = 1.28초)이 끝나고 0.2초 뒤에 감지
    "clarify_edge": ("되묻는 말 끝에 걸쳐 부름 (늦은 감지)", TWO_CUPS, ("왼쪽 컵이요", 1.48)),
    "clarify_only": ("끼어들기 없이 되묻기만", TWO_CUPS, None),
}

STT_SEC = 0.4  # 받아쓰기 왕복
SILENCE_SEC = 0.7  # 말이 끝나고 VAD 가 끝으로 보기까지(record_speech 의 silence_sec)
TTS_FIRST_SEC = 0.3  # 첫 소리까지
TTS_CHAR_SEC = 0.08  # 한 글자 읽는 시간
VLM_SEC = 1.0
STARTS: list[float] = []  # 끼어든 턴마다 다시 부르고 다음 턴이 시작되기까지(초)


class Clock:
    def __init__(self) -> None:
        self.t0 = time.monotonic()

    def log(self, text: str) -> None:
        print(f"    {time.monotonic() - self.t0:6.2f}s  {text}")


class TimedSpeechToText:
    """사람이 말한 문장(시작, 끝, 글자)을 시각 순서로 들고 있다가 listen() 때 꺼낸다.

    마이크 줄(MicStream)처럼 지나간 말도 since 로 다시 읽을 수 있다. since 가 없으면 지금 이어지고
    있거나 timeout 안에 시작하는 말을 받는다. 받아쓴 글자는 실제처럼 호출어를 지운다.
    실제 listen() 처럼 since 로 다시 읽은 말이 "자비서"뿐이면 그 뒤 timeout 안의 말을 한 번 더
    기다리고, stop() 이 True 가 되면 말하는 중이어도 바로 멈춘다.
    """

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.said: list[tuple[float, float, str]] = []
        self.last_latency_ms: float | None = None
        self.last_raw_text: str | None = None

    def say(self, start: float, end: float, text: str) -> None:
        self.said.append((start, end, text))

    def listen(
        self,
        on_speech_start=None,
        timeout=5.0,
        since=None,
        min_letters=MIN_LETTERS,
        hints=(),
        stop=None,
    ):
        self.last_latency_ms = self.last_raw_text = None
        began = time.monotonic()
        after = began if since is None else since - config.WAKE_LOOKBACK_SEC
        deadline = after + timeout  # 이때까지 말이 시작돼야 한다(record_speech 의 timeout)

        def stopped() -> bool:
            if stop is not None and stop():
                self.clock.log("듣기 멈춤 (다시 부름)")
                return True
            return False

        while True:
            if stopped():
                return None
            heard = next(((s, e, x) for s, e, x in self.said if e > after), None)
            if heard is not None and since is not None and not clean(heard[2]):
                after, deadline = heard[1], heard[1] + SILENCE_SEC + timeout  # "자비서"뿐
                continue
            if heard is not None and heard[0] <= deadline:
                break
            if time.monotonic() > deadline:
                self.clock.log(f"듣기: {timeout:g}초 동안 말 없음")
                return None
            time.sleep(0.01)
        start, end, raw = heard
        if on_speech_start is not None:
            on_speech_start(start)
        while time.monotonic() < end:
            if stopped():
                return None
            time.sleep(0.01)
        time.sleep(STT_SEC)
        self.last_latency_ms, self.last_raw_text = STT_SEC * 1000, raw
        self.clock.log(f"받아씀 {raw!r}")
        text = clean(raw)
        return Utterance(text, start, end) if is_question(text, min_letters) else None


class TimedTextToSpeech:
    """첫 소리까지 TTS_FIRST_SEC, 한 글자에 TTS_CHAR_SEC 읽는다. stop() 이면 바로 멈춘다."""

    def __init__(self, clock: Clock, on_start) -> None:
        self.clock = clock
        self.on_start = on_start  # 읽기 시작할 때 사람이 끼어들 때를 건다
        self.last_latency_ms: float | None = None
        self.last_interrupted = False
        self._stop = threading.Event()

    def speak(self, text: str) -> float | None:
        self._stop.clear()
        self.last_interrupted = False
        time.sleep(TTS_FIRST_SEC)
        self.last_latency_ms = TTS_FIRST_SEC * 1000
        self.clock.log(f"읽음 {text!r}")
        self.on_start(text)
        if self._stop.wait(TTS_CHAR_SEC * len(text)):
            self.last_interrupted = True
            self.clock.log("읽기 끊김")
        return self.last_latency_ms

    def stop(self) -> None:
        self._stop.set()


def run_scene(name: str, store: MemoryStore, user_id: int) -> None:
    title, detections, cut_in = SCENES[name]
    print(f"\n[{title}]")
    clock = Clock()
    stt = TimedSpeechToText(clock)
    barge: BargeIn
    called = threading.Event()  # 한 번만 끼어든다

    def on_start(text: str) -> None:
        if cut_in is None or cut_in[0] not in text or called.is_set():
            return
        called.set()

        def call_again() -> None:
            time.sleep(cut_in[1])
            t = time.monotonic()
            clock.log(f"사용자: {NEW_QUESTION!r}")
            stt.say(t - 0.5, t + 0.8, NEW_QUESTION)  # "자비서"가 t 에 끝나고 질문이 이어진다
            barge.on_detect(t, 0.9)

        threading.Thread(target=call_again, daemon=True).start()

    tts = TimedTextToSpeech(clock, on_start)
    barge = BargeIn(tts)
    frames = FrameBuffer()
    stop = threading.Event()
    threading.Thread(target=_camera, args=(frames, stop), daemon=True).start()
    graph = build_graph(
        GraphDeps(
            stt=stt,
            tts=tts,
            vlm=FakeVLMClient([ANSWER], delay=VLM_SEC),
            frames=frames,
            detector=_Detector(detections),
            resolve=resolve_target,
            store=store,
            called_since=barge.called_since,
        )
    )
    session_id = store.start_session(user_id, title=name)

    # 처음 질문: "자비서"가 0.2초에 끝나고 "이거 뭐야?"가 1.0초에 끝난다
    stt.say(clock.t0 - 0.3, clock.t0 + 1.0, FIRST_QUESTION)
    barge.on_detect(clock.t0 + 0.2, 0.9)
    try:
        _run_turns(graph, barge, session_id, user_id)
    finally:
        stop.set()
    _show(store, session_id, called_again=cut_in is not None)


def _run_turns(graph, barge: BargeIn, session_id: int, user_id: int) -> None:
    """scripts/assistant.py 의 run() 과 같다. 쌓인 호출이 없으면 끝난다."""
    trigger = "WAKEWORD"
    while (wake := barge.wait(timeout=1.0)) is not None:
        if trigger == "BARGE_IN":
            waited = time.monotonic() - wake.t
            print(f"    -- 다시 부르고 {waited:.2f}초 뒤 다음 턴 시작")
            STARTS.append(waited)
        turn = {"session_id": session_id, "user_id": user_id, "wake_at": wake.t}
        state = graph.invoke({**turn, "trigger_type": trigger})
        trigger = "BARGE_IN" if state.get("interrupted") else "WAKEWORD"
        if trigger == "WAKEWORD":
            barge.clear()


def _show(store: MemoryStore, session_id: int, called_again: bool) -> None:
    with store.session() as db:
        turns = (
            db.query(SessionTurn)
            .filter_by(session_id=session_id)
            .order_by(SessionTurn.turn_id)
            .all()
        )
        for n, turn in enumerate(turns, 1):
            total = "-" if turn.total_ms is None else f"{turn.total_ms / 1000:.2f}초"
            print(
                f"  {n}턴 {turn.trigger_type:<8} 질문 {turn.question_text!r} → 대답 "
                f"{turn.answer_text!r}  끊김 {_yes(turn.voice.is_interrupted)} "
                f"되묻기 {_yes(turn.inference.is_reask)}  total {total}"
            )
        if STARTS:
            print(f"  다시 부르고 다음 턴이 시작되기까지 {STARTS[-1]:.2f}초")
            STARTS.clear()
        if called_again:
            new = clean(NEW_QUESTION)
            answered = any(t.question_text == new and t.answer_text for t in turns)
            print(f"  새 질문 {new!r} 에 대답: {_yes(answered)}")


def _yes(value: bool | None) -> str:
    return "O" if value else "X"


class _Detector:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def detect(self, image: np.ndarray) -> list[Detection]:
        return list(self.detections)


def _camera(frames: FrameBuffer, stop: threading.Event) -> None:
    # 체커보드는 선명도가 높아 다시 고르지 않는다. 박스(BBox)가 들어가게 1080p 로 둔다.
    tile = np.indices((1080, 1920)).sum(axis=0) // 40 % 2
    image = np.repeat((tile * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    while not stop.is_set():
        frames.push(Frame(image=image, timestamp=time.monotonic(), source_id="check_barge_turns"))
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenes", nargs="*", help=f"장면 {', '.join(SCENES)}. 없으면 넷 다")
    args = parser.parse_args()
    unknown = [name for name in args.scenes if name not in SCENES]
    if unknown:
        parser.error(f"모르는 장면: {', '.join(unknown)}")

    with tempfile.TemporaryDirectory() as folder:
        store = MemoryStore(database_url=f"sqlite:///{(Path(folder) / 'barge.db').as_posix()}")
        store.init_schema()
        user_id = store.ensure_user()
        for name in args.scenes or SCENES:
            run_scene(name, store, user_id)
        store.engine.dispose()


if __name__ == "__main__":
    main()
