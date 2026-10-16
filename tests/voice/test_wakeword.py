"""호출어 감지기가 언제 깨어나는지 가짜 점수로 확인한다. 모델·마이크·API 는 부르지 않는다.

가짜 점수는 프레임 안에서 가장 큰 샘플 값 ÷ 1000 이다. 값이 900 인 블록이 든 프레임은 0.9 점이다.
"""

import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jarviseo.voice import stt
from jarviseo.voice.gate import MicGate
from jarviseo.voice.mic import BLOCK, SAMPLE_RATE, MicStream
from jarviseo.voice.wakeword import FRAME, WakeWordDetector

BLOCK_SEC = BLOCK / SAMPLE_RATE
DATA = Path(__file__).parents[1] / "data"


def fake_score(frame):
    return float(frame.max()) / 1000


def detector(hits, gate=None, on_detect=None, mic=None):
    return WakeWordDetector(
        mic,
        on_detect=on_detect or (lambda t, s: hits.append((t, s))),
        threshold=0.5,
        playback_threshold=0.8,
        gate=gate or MicGate(),
        scorer=fake_score,
    )


def feed(det, values, t0=0.0):
    """값이 values 인 블록을 t0 부터 30ms 간격으로 넣는다."""
    for i, v in enumerate(values):
        det.feed(np.full(BLOCK, v, np.int16), t0 + i * BLOCK_SEC)


def test_groups_30ms_blocks_into_80ms_frames():
    """30ms 블록을 80ms 로 모아 점수를 낸다."""
    frames = []
    det = WakeWordDetector(None, gate=MicGate(), scorer=lambda f: frames.append(len(f)) or 0.0)

    feed(det, [0] * 8)  # 8 × 480 = 3840 샘플

    assert frames == [FRAME] * 3


def test_calls_once_with_frame_end_time_over_threshold():
    """임계값을 넘으면 그 프레임이 끝난 시각으로 한 번 부른다."""
    hits = []
    det = detector(hits)

    feed(det, [0] * 10 + [900] * 3 + [0] * 10)  # 0.30~0.39초에 호출어

    # 900 이 처음 들어간 프레임은 0.24~0.32초 프레임이다
    assert hits == [(pytest.approx(0.32), 0.9)]


def test_detections_within_2s_count_once():
    """2초 안의 감지는 한 번으로 치고 2초가 지나면 다시 깨어난다."""
    hits = []
    det = detector(hits)

    feed(det, [900] * 40)  # 1.2초 내내 높은 점수
    feed(det, [0] * 40, t0=1.2)
    feed(det, [900] * 8, t0=2.4)  # 첫 감지 뒤 2초가 지났다

    assert [round(t, 2) for t, _ in hits] == [0.08, 2.48]


def test_threshold_is_0_8_during_playback_and_grace():
    """재생 중과 끝난 뒤 유예 안에서는 임계값이 0.8 이다."""
    gate, hits = MicGate(grace_sec=0.3), []
    det = detector(hits, gate=gate)

    with gate.playing():
        now = time.monotonic()
        feed(det, [600] * 8, t0=now)  # 0.6: 평소라면 깨지만 재생 중이라 안 깬다
        feed(det, [900] * 8, t0=now + 0.24)  # 0.9: 재생 중이어도 깬다(끼어들기)
    end = time.monotonic()
    feed(det, [600] * 8, t0=end + 0.05)  # 끝나고 0.3초 안: 아직 0.8
    feed(det, [600] * 8, t0=end + 3.0)  # 유예가 지났고 앞 감지에서 2초도 지났다

    assert [s for _, s in hits] == [0.9, 0.6]
    assert hits[1][0] > end + 3.0


def test_callback_error_keeps_detecting():
    """콜백이 예외를 내도 감지는 계속된다."""
    calls = []

    def boom(t, s):
        calls.append(t)
        raise RuntimeError("콜백 오류")

    det = detector([], on_detect=boom)

    feed(det, [900] * 8)
    feed(det, [900] * 8, t0=3.0)

    assert len(calls) == 2


def test_start_reads_mic_stream_and_stop_halts():
    """`start()` 하면 `MicStream` 을 읽어 감지하고 `stop()` 하면 멈춘다."""
    go, hits = threading.Event(), []

    def source():
        go.wait(5)
        for v in [0] * 10 + [900] * 3 + [0] * 10:
            yield np.full(BLOCK, v, np.int16), 0.0
            time.sleep(0.001)

    with MicStream(source=source()) as mic:
        det = detector(hits, mic=mic)
        det.start()
        go.set()
        det._thread.join(5)  # 소리를 다 흘리면 MicStream 이 닫히고 감지 스레드도 끝난다
        det.stop()

    assert len(hits) == 1
    assert not det._thread.is_alive()


class FakeClient:
    """openai 클라이언트 대신 받아쓰기 요청을 세고, 정해 둔 글자를 차례로 돌려준다."""

    def __init__(self, texts=("자비서, 저거 뭐야?",)):
        self.requests = []
        self.texts = list(texts)
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(text=self.texts[min(len(self.requests), len(self.texts)) - 1])


def question_audio(*parts):
    """(초, 소리 있음) 조각을 이어 붙여 공유 마이크 블록으로 만든다. 소리는 테스트용 녹음이다."""
    with wave.open(str(DATA / "jarviseo_question.wav"), "rb") as f:
        voice = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    audio = np.concatenate(
        [
            voice[: round(sec * SAMPLE_RATE)]
            if loud
            else np.zeros(round(sec * SAMPLE_RATE), np.int16)
            for sec, loud in parts
        ]
    )
    return [
        (audio[i : i + BLOCK], i / SAMPLE_RATE) for i in range(0, len(audio) - BLOCK + 1, BLOCK)
    ]


def test_listen_transcribes_only_after_since_on_shared_mic(monkeypatch):
    """호출어를 잡은 뒤 listen() 을 늦게 불러도, since 이후에 이미 지나간 말을 받아쓴다."""
    with wave.open(str(DATA / "jarviseo_question.wav"), "rb") as f:
        voice = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    silence = np.zeros(SAMPLE_RATE, np.int16)
    audio = np.concatenate([silence, voice, silence, silence])
    blocks = [
        (audio[i : i + BLOCK], i / SAMPLE_RATE) for i in range(0, len(audio) - BLOCK + 1, BLOCK)
    ]
    monkeypatch.setattr(stt, "MIC_GATE", MicGate())  # 다른 테스트의 재생 구간과 섞이지 않게

    with MicStream(source=blocks, history_sec=10.0) as mic:
        time.sleep(0.2)  # 소리가 다 지나간 뒤에 부른다
        client = FakeClient()
        listener = stt.SpeechToText(mic=mic, client=client)
        before = listener.listen(since=0.5, timeout=2.0)  # 말 시작(1초) 앞을 잡았다
        # 말이 끝나고 1.2초 뒤 (앞으로 1초 당겨 읽어도 말이 끝난 뒤다)
        after = listener.listen(since=2.2 + len(voice) / SAMPLE_RATE, timeout=2.0)

    assert before.text == "저거 뭐야?"
    assert before.started_at == pytest.approx(1.0, abs=0.06)
    assert after is None and len(client.requests) == 1


def test_question_started_before_detection_is_transcribed_whole(monkeypatch):
    """감지는 호출어가 끝나고 조금 뒤에 된다. 그새 시작한 질문 앞부분을 놓치면 안 된다."""
    with wave.open(str(DATA / "jarviseo_question.wav"), "rb") as f:
        voice = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    silence = np.zeros(SAMPLE_RATE, np.int16)
    audio = np.concatenate([silence, voice, silence])
    blocks = [
        (audio[i : i + BLOCK], i / SAMPLE_RATE) for i in range(0, len(audio) - BLOCK + 1, BLOCK)
    ]
    monkeypatch.setattr(stt, "MIC_GATE", MicGate())
    spoken_at = 1.0 + np.flatnonzero(np.abs(voice) > 500)[0] / SAMPLE_RATE  # 실제 소리가 나는 곳

    with MicStream(source=blocks, history_sec=10.0) as mic:
        listener = stt.SpeechToText(mic=mic, client=FakeClient())
        utterance = listener.listen(since=spoken_at + 0.3, timeout=2.0)  # 말이 0.3초 지나서 감지

    assert utterance.started_at <= spoken_at


def test_speech_start_is_after_wake_word_in_same_recording(monkeypatch):
    """호출어부터 같이 녹음해도 말 시작은 호출어가 끝난 뒤로 둔다."""
    monkeypatch.setattr(stt, "MIC_GATE", MicGate())
    blocks = question_audio((1.0, False), (2.0, True), (1.0, False))
    starts = []

    with MicStream(source=blocks, history_sec=10.0) as mic:
        listener = stt.SpeechToText(mic=mic, client=FakeClient())
        # 말하는 도중(1.8초)에 감지됐다. 1초 앞부터 읽으면 VAD 는 1초에서 말 시작을 잡는다
        utterance = listener.listen(on_speech_start=starts.append, since=1.8, timeout=2.0)

    assert utterance.text == "저거 뭐야?"
    assert utterance.started_at == pytest.approx(1.8 - stt.WAKE_LAG_SEC)
    assert starts == [utterance.started_at]


def test_wake_word_then_pause_waits_for_question(monkeypatch):
    """호출어만 말하고 쉬면 받아쓰지 않고 이어서 질문을 받는다."""
    monkeypatch.setattr(stt, "MIC_GATE", MicGate())
    # "자비서" 0.6초, 1초 쉬고, 질문 1.2초. 감지는 호출어가 끝나고 0.1초 뒤
    blocks = question_audio((1.0, False), (0.6, True), (1.0, False), (1.2, True), (1.0, False))
    client = FakeClient(["저거 뭐야?"])
    starts = []

    with MicStream(source=blocks, history_sec=10.0) as mic:
        listener = stt.SpeechToText(mic=mic, client=client)
        utterance = listener.listen(on_speech_start=starts.append, since=1.7, timeout=2.0)

    assert utterance.text == "저거 뭐야?"
    assert len(client.requests) == 1  # 호출어 쪽은 STT 에 보내지 않았다
    assert utterance.started_at == pytest.approx(2.6, abs=0.06)  # 질문 시작
    assert starts[-1] == utterance.started_at


def test_drawn_out_wake_word_only_waits_for_question(monkeypatch):
    """늘여 말한 호출어만 받아써지면 이어서 질문을 받는다."""
    monkeypatch.setattr(stt, "MIC_GATE", MicGate())
    # "자~비~서" 1.4초 중간(1.7초)에 감지됐고, 1초 쉬고 질문
    blocks = question_audio((1.0, False), (1.4, True), (1.0, False), (1.2, True), (1.0, False))
    client = FakeClient(["자비서.", "저거 뭐야?"])

    with MicStream(source=blocks, history_sec=10.0) as mic:
        listener = stt.SpeechToText(mic=mic, client=client)
        utterance = listener.listen(since=1.7, timeout=2.0)

    assert utterance.text == "저거 뭐야?"
    assert len(client.requests) == 2


def test_missing_custom_model_explains_download(tmp_path):
    """커스텀 모델 파일이 없으면 받는 법을 알려 준다."""
    # 모델은 Git 에 없다. openwakeword 가 없는 CI 에서도 먼저 걸러야 한다.
    from jarviseo.voice.wakeword import load_scorer

    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_scorer(str(tmp_path / "jarviseo.onnx"))
