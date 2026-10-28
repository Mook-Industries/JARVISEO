"""대시보드 브라우저의 마이크 · 스피커로 안경과 같은 팀 음성 부품을 돌린다 (#218).  담당: 문태현

서버에 마이크 · 스피커가 없을 때(클라우드 · 휴대폰) 쓴다. 예전에는 브라우저가 Chrome 받아쓰기 ·
브라우저 목소리로 따로 돌아서, 노트북과 받아쓰기 · 목소리 · "잠깐만요" · 되묻기가 달랐다.
이제 브라우저는 "자비서"만 듣고(wakeword.js), 소리 · 장면은 서버가 노트북과 같은 코드로 다룬다.

    브라우저                                 서버 (연결 하나 = BrowserVoice 하나)
    "자비서" 감지(호출어 모델)  ── wake ──▶  BargeIn.on_detect (끼어들기도 같다)
    직전 4초 + 턴 동안 마이크   ── 소리 ──▶  RemoteMic → MicStream
    턴 동안 카메라 장면         ── 장면 ──▶  FrameBuffer (말 시작 근처에서 가장 선명한 장면)
                                              2단계 확인 voice/wakecheck.py
                                              받아쓰기 voice/stt.py (OpenAI, 1.5초 앞 읽기)
                                              "잠깐만요" collector.py · 읽기 tts.py (OpenAI 목소리)
    받은 순서대로 소리 내기     ◀── PCM ───   RemoteOutput (대답 = 채널 0, 시작음 · 종료음 = 1)
    화면 바꾸기                 ◀── 알림 ──   wake · thinking · result · idle …

게이트(재생 중 마이크 막기)는 연결마다 따로 둔다. 검출기 · 손 모델 · VLM · 기억은 웹 질문
(assistant_web.py)과 같은 것을 같이 쓰고, OCR · 검출은 성분 페이지와 같은 잠금 안에서 돈다.

보내는 모양(WebSocket /api/voice/ws, main.py)
    브라우저 → 서버  바이트: [종류 1][pos float64][내용]  1 = 마이크(16kHz int16), 2 = 장면 JPEG
                     pos 는 마이크를 연 뒤 몇 번째 샘플인지(장면은 찍은 순간까지 센 수)
                     글자(JSON): wake{id, pos, score, button, camera} · ask{question, image} · stop
                                 · voice{voice, speed}  (환경설정 보이스 선택 · 말하기 속도 배수)
    서버 → 브라우저  바이트: [채널 1][PCM 24kHz int16],  글자(JSON): 위의 알림, flush{ch}
"""

from __future__ import annotations

import logging
import queue
import struct
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from jarviseo import config
from jarviseo.api.assistant_web import WHOLE_LABEL, _RetryWholeFrame, save_photos
from jarviseo.api.ingredient_turn import IngredientFollowUp
from jarviseo.capture import FrameBuffer
from jarviseo.types import BBox, Detection, Frame, Intent

__all__ = ["AUDIO", "BrowserVoice", "FRAME", "HEADER"]

log = logging.getLogger(__name__)

PHOTO_DIR = Path("data/captures/web")
HEADER = struct.Struct("<Bd")  # 종류, pos
AUDIO, FRAME = 1, 2
TTS_CHANNEL, CHIME_CHANNEL = 0, 1
# 호출 직전 소리가 마이크 버퍼에 다 들어올 때까지 기다리는 최대 시간(초)
CATCH_UP_SEC = 0.5


class _Turn:
    """지금 턴에서 들은 질문과 고른 장면. 성분 질문 보완(assistant_web.py 와 같다)에 쓴다."""

    def __init__(self) -> None:
        self.reset()

    def reset(self, follow_up_of: str | None = None) -> None:
        self.question: str | None = None
        self.frame: np.ndarray | None = None
        self.checker: _RetryWholeFrame | None = None
        # 되물은 뒤 이어 듣는 턴이면 앞 질문. "다시 확인해줘"는 앞 질문을 붙여야 성분 질문이다
        self.follow_up_of = follow_up_of

    @property
    def ingredient(self) -> bool:
        from jarviseo.workflow.follow_up import follow_up_question
        from jarviseo.workflow.router import classify_intent

        if self.question is None:
            return False
        text = self.question
        if self.follow_up_of is not None:
            text = follow_up_question(self.follow_up_of, text)
        return classify_intent(text) is Intent.INGREDIENT


class _HeardSTT:
    """받아쓴 첫 질문을 턴에 적는다(되묻기 대답은 질문이 아니다). 나머지는 그대로 넘긴다."""

    def __init__(self, inner, turn: _Turn) -> None:
        self.inner, self.turn = inner, turn

    def listen(self, *args, **kwargs):
        utterance = self.inner.listen(*args, **kwargs)
        if utterance is not None and self.turn.question is None:
            self.turn.question = utterance.text
        return utterance

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


class _TurnFrames(FrameBuffer):
    """고른 장면을 턴에 적는다. 성분 판정이 크롭을 못 읽으면 이 장면 전체를 다시 읽는다."""

    def __init__(self, maxlen: int, turn: _Turn) -> None:
        super().__init__(maxlen)
        self.turn = turn

    def pick_best(self, around: float | None = None, window: float = 0.5) -> Frame | None:
        frame = super().pick_best(around, window)
        if frame is not None:
            self.turn.frame = frame.image
        return frame


class _Locked:
    """검출기 · 손 모델을 웹 질문 · 성분 페이지와 같은 잠금 안에서 부른다."""

    def __init__(self, inner, lock: threading.Lock) -> None:
        self.inner, self.lock = inner, lock

    def detect(self, image: np.ndarray):
        with self.lock:
            return self.inner.detect(image)


class _TurnDetector(_Locked):
    """성분 질문인데 물체를 하나도 못 찾으면 사진 전체를 대상으로 준다(assistant_web 과 같다)."""

    def __init__(self, inner, lock: threading.Lock, turn: _Turn) -> None:
        super().__init__(inner, lock)
        self.turn = turn

    def detect(self, image: np.ndarray) -> list[Detection]:
        found = super().detect(image)
        if found or not self.turn.ingredient:
            return found
        h, w = image.shape[:2]
        return [Detection(label=WHOLE_LABEL, confidence=1.0, bbox=BBox(0, 0, w, h))]


class _TurnChecker:
    """크롭에서 성분표를 못 읽으면 고른 장면 전체로 한 번 더 읽는다(assistant_web 과 같다)."""

    def __init__(self, inner, lock: threading.Lock, turn: _Turn) -> None:
        self.inner, self.lock, self.turn = inner, lock, turn
        self.synonyms = getattr(inner, "synonyms", {})

    def check(self, image: np.ndarray, allergens: list[str], *args, **kwargs):
        whole = self.turn.frame if self.turn.frame is not None else image
        self.turn.checker = checker = _RetryWholeFrame(self.inner, whole)
        with self.lock:
            return checker.check(image, allergens, *args, **kwargs)


class _WakeCheck:
    """2단계 확인(voice/wakecheck.py). 화면의 호출 버튼으로 부른 것은 확인하지 않는다."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.buttons: set[float] = set()

    def start(self, t: float):
        if t in self.buttons:
            self.buttons.discard(t)
            from jarviseo.voice.wakecheck import WakeVerdict

            done: Future[str | None] = Future()
            done.set_result(None)  # 받아쓰지 않음 = 통과
            return WakeVerdict(done, time.monotonic())
        return self.inner.start(t)


class BrowserVoice:
    """WebSocket 연결 하나. start() 로 부품을 만들고 close() 로 끝낸다.

    on_* 는 WebSocket 을 받는 쪽에서 부른다. send(bytes | dict) 는 어느 스레드에서 불러도
    브라우저로 순서대로 가야 한다(main.py 가 보내는 줄 하나에 넣는다).
    """

    def __init__(
        self,
        store,
        checker,
        user_id: int,
        lock: threading.Lock,
        assistant,
        send: Callable[[bytes | dict], None],
        to_result: Callable[[dict, float, BrowserVoice], dict],
        openai_client=None,
    ) -> None:
        """assistant 는 웹 질문의 WebAssistant(무거운 부품을 같이 쓴다).
        to_result(state, t0, voice) 는 끝난 상태를 /api/ask 응답 모양으로 바꾼다(main.py).
        openai_client 는 테스트에서 가짜 받아쓰기 · 읽기 API 를 끼울 때만 넘긴다.
        """
        self.store, self.checker, self.user_id = store, checker, user_id
        self.lock, self.assistant = lock, assistant
        self.send, self.to_result = send, to_result
        self.client = openai_client
        self.pointing = "v1"
        self.retried_whole = False  # 지난 턴에 크롭을 못 읽어 장면 전체로 다시 읽었나
        self.judged_image: np.ndarray | None = None  # 지난 턴 성분 판정에 쓴 사진
        self.turn = _Turn()
        self.session_id: int | None = None
        self._asks: queue.SimpleQueue[tuple[str, np.ndarray | None]] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._typed = False  # 입력창 질문을 돌리는 중(성분표를 보여 달라고 한 뒤 듣지 않는다)
        self._wake_lock = threading.Lock()
        self._wake_id = 0  # 받은 호출 번호 중 가장 큰 것. idle 에 실어 보낸다
        self._camera: dict[float, bool] = {}  # 호출 시각 → 그 호출에 장면을 보내는지
        self._thread: threading.Thread | None = None
        self._p: dict = {}

    # ---- 켜고 끄기 ----

    def start(self) -> None:
        """부품을 만들고 듣기 시작한다. OpenAI 키가 없으면 RuntimeError."""
        if self.client is None and not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 가 없어 받아쓰기 · 읽기를 못 한다")
        from jarviseo.voice import (
            BargeIn,
            Chime,
            ResponseCollector,
            SpeechToText,
            TextToSpeech,
            WakeCheck,
        )
        from jarviseo.voice.gate import MicGate
        from jarviseo.voice.mic import MicStream
        from jarviseo.voice.remote import RemoteMic, RemoteOutput
        from jarviseo.workflow import GraphDeps, build_graph

        with self.lock:  # 웹 질문과 같이 쓰는 무거운 부품(처음이면 여기서 만든다)
            shared = self.assistant.shared_parts()
        p = self._p
        p["remote"] = remote = RemoteMic()
        p["mic"] = mic = MicStream(source=remote.blocks()).__enter__()
        gate = MicGate()  # 연결마다 따로. 다른 브라우저가 읽는 동안 이 마이크를 막지 않게
        p["tts"] = tts = TextToSpeech(
            output=RemoteOutput(self.send, TTS_CHANNEL), gate=gate, client=self.client
        )
        p["chime"] = chime = Chime(gate=gate, output=RemoteOutput(self.send, CHIME_CHANNEL))
        p["barge"] = barge = BargeIn(tts, gate)
        p["collector"] = collector = ResponseCollector(tts, called_since=barge.called_since)

        def speech_end() -> None:
            chime.end()  # 질문 말이 끝나면 종료음(assistant.py 와 같다)
            self.send({"type": "thinking"})

        stt = SpeechToText(mic=mic, on_speech_end=speech_end, gate=gate, client=self.client)
        check = None
        if config.WAKE_CHECK:
            p["check"] = check = _WakeCheck(WakeCheck(mic, client=self.client))
        p["frames"] = frames = _TurnFrames(config.FRAME_BUFFER_SIZE, self.turn)
        dialog = shared["dialog"]
        hands = shared["hands"]
        self.pointing = "v2" if hands is not None else "v1"
        # "여기" 장면 크롭(IngredientFollowUp)에는 장면 전체 박스를 만들지 않는 원래 탐지기를 준다
        raw_detector = _Locked(shared["detector"], self.lock)
        locked_hands = _Locked(hands, self.lock) if hands is not None else None
        self.deps = GraphDeps(
            stt=_HeardSTT(stt, self.turn),
            tts=tts,
            vlm=shared["vlm"],
            frames=frames,
            detector=_TurnDetector(shared["detector"], self.lock, self.turn),
            hands=locked_hands,
            resolve=shared["resolve"],
            # 성분표를 못 읽으면 같은 턴 안에서 보여 달라고 하고 다시 읽는다(api/ingredient_turn.py)
            checker=IngredientFollowUp(
                _TurnChecker(self.checker, self.lock, self.turn),
                tts,
                stt,
                frames,
                start_chime=chime.start,
                on_listen=lambda: self.send({"type": "wake", "barge_in": False, "followup": True}),
                active=lambda: not self._typed,
                detector=raw_detector,
                hands=locked_hands,
                resolve=shared["resolve"],
            )
            if self.checker
            else None,
            store=self.store,
            memory=dialog.memory if dialog is not None else None,
            dialog=dialog,
            collector=collector,
            called_since=barge.called_since,
            wake_check=check,
            start_chime=chime.start,
            recall_enabled=config.ENABLE_RECALL and dialog is not None,
        )
        self.graph = build_graph(self.deps)
        # 카메라가 꺼져 있고 올린 사진도 없으면 사진 없이 묻는다(웹 질문과 같다)
        self.graph_blind = build_graph(replace(self.deps, frames=None))
        self.session_id = self.store.start_session(self.user_id, title="web-voice")
        self._thread = threading.Thread(target=self._run, name="browser-voice", daemon=True)
        self._thread.start()
        self.send({"type": "ready", "pointing": self.pointing, "recall": self.deps.recall_enabled})
        log.info("브라우저 음성 연결 켜짐 (가리킴 %s)", self.pointing)

    def close(self) -> None:
        """연결이 끊겼다. 읽던 것을 멈추고 듣던 질문도 끝낸다."""
        self._stop.set()
        p = self._p
        if "tts" in p:
            p["tts"].stop()
        if "remote" in p:
            p["remote"].close()  # 듣던 listen() 은 마이크가 닫혀 끝난다
        if "mic" in p:
            p["mic"].close()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if "collector" in p:
            p["collector"].close()
        if self.session_id is not None:
            self.store.end_session(self.session_id)
        self._p = {}

    # ---- 브라우저에서 온 것 ----

    def on_binary(self, data: bytes) -> None:
        """마이크 소리나 카메라 장면 하나."""
        if len(data) < HEADER.size or not self._p:
            return
        kind, pos = HEADER.unpack_from(data)
        body = data[HEADER.size :]
        if kind == AUDIO and len(body) % 2 == 0:
            self._p["remote"].push(int(pos), np.frombuffer(body, "<i2"))
        elif kind == FRAME:
            image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                t = self._p["remote"].time_of(pos)
                self._p["frames"].push(Frame(image=image, timestamp=t, source_id="web"))

    def on_message(self, msg: dict) -> None:
        kind = msg.get("type")
        p = self._p
        if not p:
            return
        if kind == "wake":
            # 호출어 감지 · 호출 버튼 = "자비서". 읽는 중이면 끊는다(끼어들기)
            t = p["remote"].time_of(float(msg.get("pos", 0)))
            with self._wake_lock:
                self._wake_id = max(self._wake_id, int(msg.get("id", 0)))
                if msg.get("button") and "check" in p:
                    p["check"].buttons.add(t)
                self._camera[t] = bool(msg.get("camera", True))
                # 지난 턴 장면이 이번 질문에 뽑히지 않게. 새 장면은 이 뒤에 온다
                p["frames"].clear()
                p["barge"].on_detect(t, float(msg.get("score", 1.0)))
        elif kind == "ask":
            question = str(msg.get("question", "")).strip()
            if question:
                p["tts"].stop()
                self._asks.put((question, msg.get("image")))
        elif kind == "stop":
            p["tts"].stop()  # 화면을 떠났다
        elif kind == "voice":
            # 환경설정. 소리 크기는 브라우저가 받은 소리에 곱한다(voicelink.js)
            if p["tts"].configure(voice=msg.get("voice"), speed=msg.get("speed")):
                p["collector"].refresh_filler()

    # ---- 듣고 답하는 스레드 (voice_loop.py · assistant.run 과 같다) ----

    def _run(self) -> None:
        from jarviseo.workflow.follow_up import follow_up_turn

        barge = self._p["barge"]
        trigger = "WAKEWORD"
        while not self._stop.is_set():
            wake = barge.wait(timeout=0.2)
            if wake is None:
                try:
                    question, image = self._asks.get_nowait()
                except queue.Empty:
                    continue
                # 입력칸 질문은 브라우저가 마이크를 안 보내서 이어 듣지 않는다
                state = self._ask_text(question, image)
            else:
                state = self._voice_turn(wake, trigger)
                # 자비서가 되물었으면(성분표 다시 보여 주기 등) 대기로 가지 않고 호출어 없이
                # 이어 듣는다. idle 을 안 보냈으니 브라우저는 소리 · 장면을 계속 보낸다.
                while state is not None and (follow := follow_up_turn(state)) is not None:
                    if self._stop.is_set():
                        break
                    state = self._follow_up(follow)
            # 읽는 중에 "자비서"로 끊겼으면 그 호출이 다음 턴이다. 아니면 쌓인 호출은 버리고 대기로
            trigger = "BARGE_IN" if state is not None and state.get("interrupted") else "WAKEWORD"
            if trigger == "WAKEWORD":
                self._idle()

    def _voice_turn(self, wake, trigger: str) -> dict | None:
        """호출 한 번 = 그래프 한 턴(assistant.run 과 같다)."""
        self._catch_up(wake.t)
        self._p["chime"].start()
        self.send({"type": "wake", "barge_in": trigger == "BARGE_IN"})
        info = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "wake_at": wake.t,
            "trigger_type": trigger,
        }
        with self._wake_lock:
            camera = self._camera.pop(wake.t, True)
            self._camera.clear()  # 같이 쌓였다 버려진 호출
        # 이어 듣는 턴도 이 호출과 같이 장면을 쓰거나 안 쓴다
        self._turn_graph = self.graph if camera else self.graph_blind
        return self._invoke(self._turn_graph, info)

    def _follow_up(self, info: dict) -> dict | None:
        """호출어 없이 이어 듣는 턴. 시작음은 그래프의 듣기가 낸다."""
        with self._wake_lock:
            # 대답을 읽기 전 장면이 다시 보여 준 장면 대신 뽑히지 않게. 새 장면은 이 뒤에 온다
            self._p["frames"].clear()
        self.send({"type": "wake", "follow_up": True})
        return self._invoke(self._turn_graph, info)

    def _idle(self) -> None:
        """턴이 끝났다. 브라우저는 이 번호까지의 호출 뒤로는 소리 · 장면을 그만 보낸다."""
        with self._wake_lock:
            self._p["barge"].clear()
            upto = self._wake_id
        self.send({"type": "idle", "upto": upto})

    def _catch_up(self, t: float) -> None:
        """호출 직전 소리(브라우저가 한꺼번에 보낸 묶음)가 마이크 버퍼에 들어올 때까지 기다린다.

        안 그러면 1.5초 앞 읽기 · 2단계 확인이 빈 구간을 읽는다.
        """
        mic = self._p["mic"]
        deadline = time.monotonic() + CATCH_UP_SEC
        while not len(mic.clip(t - 0.1, t)) and time.monotonic() < deadline:
            time.sleep(0.01)

    def _ask_text(self, question: str, image) -> dict | None:
        """입력칸 질문. 받아쓰기만 글자로 바꾸고 읽기 · "잠깐만요" · 기억은 같은 부품으로."""
        from jarviseo.api.main import _decode_image
        from jarviseo.voice import FakeSpeechToText
        from jarviseo.workflow import build_graph

        now = time.monotonic()
        frames = None
        if image:
            try:
                picture = _decode_image(image)
            except Exception:
                picture = None
            if picture is not None:
                frames = _TurnFrames(4, self.turn)
                frames.push(Frame(image=picture, timestamp=now, source_id="upload"))
        stt = _HeardSTT(FakeSpeechToText([question], starts=[now]), self.turn)
        self.send({"type": "thinking"})
        info = {"session_id": self.session_id, "user_id": self.user_id, "trigger_type": "WAKEWORD"}
        self._typed = True
        try:
            return self._invoke(build_graph(replace(self.deps, stt=stt, frames=frames)), info)
        finally:
            self._typed = False

    def _invoke(self, graph, info: dict) -> dict | None:
        """그래프 한 턴. 결과를 브라우저에 알린다."""
        self.turn.reset(info.get("follow_up_of"))
        t0 = time.perf_counter()
        try:
            state = graph.invoke(info)
        except Exception as e:
            log.exception("브라우저 음성 질문 처리 실패")
            self.send({"type": "error", "message": str(e)})
            return None
        if state.get("wake_rejected"):
            self.send({"type": "rejected", "text": state.get("wake_check_text")})
        elif state.get("utterance") is None:
            self.send({"type": "no_question", "follow_up": "follow_up_of" in info})
        else:
            checker = self.turn.checker
            self.retried_whole = checker is not None and checker.retried
            self.judged_image = checker.used if checker is not None else None
            save_photos(state, PHOTO_DIR, self.store)
            self.send({"type": "result", "data": self.to_result(state, t0, self)})
        return state
