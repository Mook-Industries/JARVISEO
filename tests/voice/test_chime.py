"""시작음·종료음 파일을 맞게 읽고, 시작음은 게이트를 열어 둔 채 종료음은 닫고 내는지 확인한다.

스피커는 부르지 않는다.
"""

import sys
import time
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from jarviseo.voice.chime import END_WAV, START_WAV, Chime, load_sound
from jarviseo.voice.gate import SOUND_TAIL_SEC, MicGate
from jarviseo.voice.speaker import SAMPLE_RATE


@pytest.fixture
def speaker_log(monkeypatch):
    """sounddevice 대신 스피커를 연 설정과 쓴 소리를 순서대로 적는다."""
    log = []

    class FakeOutputStream:
        latency = 0.05  # 쓴 소리가 스피커에서 나오기까지(초)

        def __init__(self, **kwargs):
            log.append(("open", kwargs["latency"]))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            log.append("stop")

        def write(self, data):
            log.append(len(data))

        def abort(self):
            log.append("abort")

    sounddevice = SimpleNamespace(RawOutputStream=FakeOutputStream)
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    return log


def seconds(pcm):
    return len(pcm) / 2 / SAMPLE_RATE


def written(log):
    """스피커에 쓴 바이트 수. 게이트 훅이 적은 True/False 는 int 이기도 해서 type 으로 거른다."""
    return sum(entry for entry in log if type(entry) is int)


def test_load_sound_trims_both_files_short():
    """두 소리 모두 앞뒤 무음을 잘라 0.5초 안쪽이다(원본 시작음 1초, 종료음 0.68초)."""
    start, end = load_sound(START_WAV), load_sound(END_WAV)

    assert 0.2 < seconds(start) < 0.5
    assert 0.2 < seconds(end) < 0.5


def test_load_sound_scales_by_volume():
    """volume 을 곱한다. 0.5 면 가장 큰 소리가 절반이다."""
    full = np.frombuffer(load_sound(START_WAV), np.int16)
    half = np.frombuffer(load_sound(START_WAV, volume=0.5), np.int16)

    assert len(half) == len(full)
    assert abs(int(np.abs(half).max()) - int(np.abs(full).max()) / 2) <= 1


def test_load_sound_zero_volume_is_empty():
    """volume 이 0 이면 파일을 읽지 않고 빈 바이트다."""
    assert load_sound(START_WAV, volume=0) == b""


def test_load_sound_rejects_other_format(tmp_path):
    """TTS 와 다른 형식(16kHz)이면 Speaker 가 그대로 낼 수 없어서 ValueError."""
    path = tmp_path / "16k.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x10\x27" * 1600)

    with pytest.raises(ValueError):
        load_sound(path)


def test_start_plays_without_closing_gate(speaker_log):
    """시작음은 게이트를 닫지 않는다. 그 사이에 이어 말한 질문이 무음으로 바뀌면 안 된다."""
    gate = MicGate()
    gate.on_playback(speaker_log.append)
    chime = Chime(gate=gate)

    asked = time.monotonic()
    chime.start()
    chime.wait()

    assert speaker_log[0] == ("open", "low")
    assert True not in speaker_log and False not in speaker_log  # 게이트 훅이 안 불렸다
    assert written(speaker_log) == len(chime.start_pcm)
    assert gate.is_open(time.monotonic())
    assert chime.speaker.started_at >= asked


def test_start_marks_sound_for_vad(speaker_log):
    """시작음이 스피커에서 나오는 구간(출력 지연 0.05초 뒤부터)을 게이트에 표시한다.

    STT 의 VAD 가 그 구간을 말로 보지 않아야 "자비서." 하고 쉰 틈을 시작음이 메우지 않는다.
    """
    gate = MicGate()
    chime = Chime(gate=gate)

    chime.start()
    chime.wait()
    done = time.monotonic()

    started = chime.speaker.started_at
    assert not gate.is_sounding(started)  # 아직 스피커에서 안 나왔다
    assert gate.is_sounding(started + 0.05)
    assert gate.is_sounding(done + 0.05)  # 다 쓴 뒤에도 출력 지연 + 울림만큼 남는다
    assert not gate.is_sounding(done + 0.05 + SOUND_TAIL_SEC + 0.01)


def test_end_does_not_mark_sound(speaker_log):
    """종료음은 게이트를 닫아서 따로 표시하지 않는다."""
    gate = MicGate()

    Chime(gate=gate).end()

    assert not gate.is_sounding(time.monotonic() - 0.01)


def test_end_closes_gate_like_tts(speaker_log):
    """종료음은 TTS 처럼 소리 전에 게이트를 닫고 다 낸 뒤 연다(유예 0.3초)."""
    gate = MicGate(grace_sec=0.3)
    gate.on_playback(speaker_log.append)
    chime = Chime(gate=gate)

    chime.end()

    assert speaker_log[0] is True and speaker_log[-2:] == ["stop", False]
    assert written(speaker_log) == len(chime.end_pcm)
    assert not gate.is_open(time.monotonic())  # 끝난 직후는 아직 유예 중


def test_end_waits_for_start_to_finish(speaker_log):
    """시작음이 나는 중에 end() 를 부르면 시작음을 다 낸 뒤 종료음을 낸다."""
    chime = Chime(gate=MicGate())

    chime.start()
    chime.end()

    opened = [i for i, entry in enumerate(speaker_log) if isinstance(entry, tuple)]
    assert len(opened) == 2
    assert "stop" in speaker_log[opened[0] : opened[1]]  # 시작음 스트림을 닫은 뒤 다시 열었다


def test_zero_volume_never_opens_speaker(speaker_log):
    """volume 0 이면 스피커를 열지 않는다(.env JARVISEO_CHIME_VOLUME=0 으로 끈다)."""
    chime = Chime(volume=0, gate=MicGate())

    chime.start()
    chime.end()

    assert speaker_log == []
