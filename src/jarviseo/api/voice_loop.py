"""웹 대시보드의 팀 음성 모드 — 안경 실행(scripts/assistant.py)과 같은 팀 부품을 그대로 돌린다.

서버가 이 PC 의 마이크 · 스피커 · 웹캠을 직접 연다. 부품은 assistant.py 의 main() 과 같다.
    호출어        voice/wakeword.py (data/models/jarviseo.onnx)          문태현
    시작음·종료음  voice/chime.py                                         문태현
    받아쓰기      voice/stt.py (OpenAI, 되묻기 대답에 후보 이름 힌트)     문태현
    "잠깐만요"    voice/collector.py                                     문태현
    읽기          voice/tts.py · tts_text.py (OpenAI 목소리)              문태현
    끼어들기      voice/bargein.py (읽는 중 "자비서" → 끊고 새 질문)       문태현
    대화 기억     memory/dialog.py · vector.py                            문태현
    그래프 · 가리킴 · 되묻기  workflow/ · pointing/ (손 방향 v2)           최홍묵
    성분 판정     nutrition/ (성분 페이지와 같은 판정기 · 같은 잠금)       권용현
카메라만 브라우저가 맡는다. 웹캠은 한 프로그램만 잡을 수 있어서, 대시보드 탭이 잡은 카메라의
장면을 /api/voice/frame 으로 받아 FrameBuffer 에 넣는다(assistant.py 의 Camera 자리).
턴이 끝날 때마다 결과를 events 에 넣고, 페이지가 /api/voice/events 로 가져간다.

assistant.py 와 다른 점은 장면이 들어오는 길 · 화면에 알리는 것(events) · 성분 판정기와 DB 를
웹 서버 것을 같이 쓰는 것뿐이다. 실행 흐름(run)은 assistant.run 을 그대로 옮겼다.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from jarviseo import config
from jarviseo.types import Frame

__all__ = ["VoiceLoop"]

log = logging.getLogger(__name__)

PHOTO_DIR = Path("data/captures/web")
EVENTS_KEPT = 100


class VoiceLoop:
    """start() 하면 "자비서"를 기다렸다가 그래프로 질문 한 번씩 돌린다. stop() 으로 끝낸다."""

    def __init__(
        self,
        store,
        checker,
        user_id: int,
        lock: threading.Lock,
        to_result: Callable[[dict, float, VoiceLoop], dict],
    ) -> None:
        """to_result(state, t0, loop) 는 끝난 상태를 /api/ask 응답 모양으로 바꾼다(main.py)."""
        self.store, self.checker, self.user_id = store, checker, user_id
        self.lock, self.to_result = lock, to_result
        self.running = False
        self.reason: str | None = None  # 못 켰으면 그 이유
        self.pointing = "v1"
        self._events: deque[dict] = deque(maxlen=EVENTS_KEPT)
        self._seq = 0
        self._ev_lock = threading.Lock()
        self._stop = threading.Event()
        self._parts: dict = {}
        self._failed: str | None = None  # 못 켠 이유(다시 시도하지 않는다)

    # ---- 화면에 알리기 ----

    def emit(self, kind: str, **data) -> None:
        with self._ev_lock:
            self._seq += 1
            self._events.append({"id": self._seq, "type": kind, **data})

    def events_after(self, after: int) -> tuple[list[dict], int]:
        with self._ev_lock:
            return [e for e in self._events if e["id"] > after], self._seq

    def push_frame(self, image: np.ndarray) -> None:
        """브라우저 카메라의 장면 한 장. 받은 시각으로 넣는다(그래프가 말 시작 근처를 고른다)."""
        frames = self._parts.get("frames")
        if frames is not None:
            frames.push(Frame(image=image, timestamp=time.monotonic(), source_id="web"))

    def ask_text(self, question: str, image: np.ndarray | None = None) -> dict:
        """입력창 질문. 받아쓰기만 글자로 바꾸고 나머지는 같은 팀 부품으로 돈다.

        대답은 서버 스피커(voice/tts.py)가 읽고, 늦으면 "잠깐만요"(collector), 기억 저장 · 검색,
        장면은 카메라 버퍼에서 고른다(FrameBuffer.pick_best). 사진을 올렸으면 그 사진으로 묻는다.
        """
        from dataclasses import replace

        from jarviseo.api.assistant_web import save_photos
        from jarviseo.capture import FrameBuffer
        from jarviseo.voice import FakeSpeechToText
        from jarviseo.workflow import build_graph

        now = time.monotonic()
        deps = replace(self._deps, stt=FakeSpeechToText([question], starts=[now]))
        if image is not None:
            frames = FrameBuffer(4)
            frames.push(Frame(image=image, timestamp=now, source_id="upload"))
            deps = replace(deps, frames=frames)
        info = {"session_id": self.session_id, "user_id": self.user_id, "trigger_type": "WAKEWORD"}
        t0 = time.perf_counter()
        with self.lock:
            state = build_graph(deps).invoke(info)
        save_photos(state, PHOTO_DIR)
        return self.to_result(state, t0, self)

    def call(self) -> None:
        """화면의 호출 버튼. "자비서"라고 부른 것과 같다(읽는 중이면 끼어들기)."""
        barge = self._parts.get("barge")
        if barge is not None:
            barge.on_detect(time.monotonic(), 1.0)

    # ---- 켜고 끄기 ----

    def start(self) -> None:
        """부품을 만들고 마이크를 연다. 못 열면 RuntimeError(이유).

        한 번 못 켜면(키 · 마이크 · 스피커 · 호출어 모델 없음) 이유를 기억하고 다시 시도하지 않는다.
        클라우드 서버(장치 없음)는 페이지를 열 때마다 켜 보느라 검출 모델을 매번 다시 읽었다.
        장치를 꽂았으면 서버를 다시 켠다.
        """
        if self.running:
            return
        if self._failed:
            raise RuntimeError(self._failed)
        try:
            self._start()
        except Exception as e:
            self._close()
            self._failed = str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}"
            raise RuntimeError(self._failed) from e

    def _start(self) -> None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 가 없어 받아쓰기 · 읽기 · VLM 을 못 쓴다")
        from jarviseo.capture import FrameBuffer
        from jarviseo.memory import DialogMemory, VectorMemory
        from jarviseo.pointing import resolve_target
        from jarviseo.pointing.detector import ObjectDetector
        from jarviseo.vlm import VLMClient
        from jarviseo.voice import (
            BargeIn,
            Chime,
            ResponseCollector,
            SpeechToText,
            TextToSpeech,
            WakeWordDetector,
        )
        from jarviseo.voice.devices import find_device
        from jarviseo.voice.mic import MicStream
        from jarviseo.workflow import GraphDeps, build_graph

        try:
            mic_device = find_device(config.MIC_DEVICE, "input")
            speaker_device = find_device(config.SPEAKER_DEVICE, "output")
        except ValueError as e:
            raise RuntimeError(str(e)) from None
        p = self._parts
        p["vlm"] = VLMClient()
        p["dialog"] = DialogMemory(VectorMemory(self.store))
        p["tts"] = TextToSpeech(device=speaker_device)
        p["chime"] = chime = Chime(device=speaker_device)
        p["barge"] = barge = BargeIn(p["tts"])
        p["collector"] = ResponseCollector(p["tts"], called_since=barge.called_since)
        p["frames"] = frames = FrameBuffer(config.FRAME_BUFFER_SIZE)
        detector = ObjectDetector()
        try:
            from jarviseo.pointing.hands import HandDetector

            hands = HandDetector()
        except (FileNotFoundError, ImportError) as e:
            log.warning("손 단서 꺼짐(가리킴 v1): %s", e)
            hands = None
        self.pointing = "v2" if hands is not None else "v1"

        p["mic"] = mic = MicStream(mic_device)
        mic.__enter__()

        def speech_end() -> None:
            chime.end()  # 질문 말이 끝나면 종료음(assistant.py 와 같다)
            self.emit("thinking")

        deps = GraphDeps(
            stt=SpeechToText(mic=mic, on_speech_end=speech_end),
            tts=p["tts"],
            vlm=p["vlm"],
            frames=frames,
            store=self.store,
            memory=p["dialog"].memory,
            dialog=p["dialog"],
            collector=p["collector"],
            called_since=barge.called_since,
            start_chime=chime.start,
            detector=detector,
            hands=hands,
            resolve=resolve_target,
            checker=self.checker,
        )
        graph = build_graph(deps)
        self._deps = deps
        p["wake"] = wake = WakeWordDetector(mic, on_detect=barge.on_detect)
        self.session_id = self.store.start_session(self.user_id, title="web-voice")
        self._stop.clear()
        wake.start()
        threading.Thread(target=self._run, args=(graph, barge, chime), daemon=True).start()
        self.running, self.reason = True, None
        self.emit("ready", pointing=self.pointing, recall=deps.recall_enabled)
        log.info("팀 음성 모드 켜짐 (가리킴 %s)", self.pointing)

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self._stop.set()
        wake = self._parts.get("wake")
        if wake is not None:
            wake.stop()
        time.sleep(0.6)  # 돌던 대기(0.5초)가 끝나게
        self._close()
        self.store.end_session(self.session_id)
        self.emit("stopped")

    def _close(self) -> None:
        p = self._parts
        if "mic" in p:
            p["mic"].__exit__(None, None, None)
        for name in ("collector", "dialog"):
            if name in p:
                p[name].close()
        self._parts = {}

    # ---- 스레드 ----

    def _run(self, graph, barge, chime) -> None:
        """assistant.run 과 같다: 호출어를 기다리다가 질문 한 번을 그래프로 돌린다."""
        from jarviseo.api.assistant_web import save_photos

        trigger = "WAKEWORD"
        while not self._stop.is_set():
            wake = barge.wait(timeout=0.5)
            if wake is None:
                continue
            chime.start()
            self.emit("wake", score=round(wake.score, 3), barge_in=trigger == "BARGE_IN")
            info = {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "wake_at": wake.t,
                "trigger_type": trigger,
            }
            t0 = time.perf_counter()
            try:
                with self.lock:  # 성분 판정기(OCR)를 성분 페이지와 같이 쓴다
                    state = graph.invoke(info)
            except Exception as e:
                log.exception("질문 처리 실패")
                self.emit("error", message=str(e))
                continue
            if state.get("utterance") is None:
                self.emit("no_question")
            else:
                save_photos(state, PHOTO_DIR)
                self.emit("result", data=self.to_result(state, t0, self))
            trigger = "BARGE_IN" if state.get("interrupted") else "WAKEWORD"
            if trigger == "WAKEWORD":
                barge.clear()
