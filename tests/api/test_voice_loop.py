"""팀 음성 모드(api/voice_loop.py) — 장치를 열지 않고 볼 수 있는 것만 본다.

마이크 · 스피커 · 호출어 모델을 여는 실제 흐름은 scripts/assistant.py 와 같은 부품이라
노트북에서 직접 돌려 확인한다.
"""

import threading

import numpy as np
import pytest

from jarviseo import config
from jarviseo.api.voice_loop import VoiceLoop


def make_loop():
    return VoiceLoop(store=None, checker=None, user_id=1, lock=threading.Lock(), to_result=None)


def test_events_after_returns_only_newer():
    loop = make_loop()
    loop.emit("wake", score=0.99)
    loop.emit("thinking")
    events, last = loop.events_after(1)
    assert [e["type"] for e in events] == ["thinking"] and last == 2


def test_frame_and_call_before_start_are_ignored():
    loop = make_loop()
    loop.push_frame(np.zeros((4, 4, 3), np.uint8))  # 켜기 전이라 버퍼가 없다
    loop.call()
    assert loop.events_after(0) == ([], 0)


def test_start_needs_openai_key(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_loop().start()


def test_failed_start_is_not_retried(monkeypatch):
    """장치가 없어 못 켰으면 다시 켜 보지 않는다(클라우드에서 페이지마다 모델을 다시 읽지 않게)."""
    calls = []
    loop = make_loop()

    def no_mic():
        calls.append(1)
        raise OSError("마이크 없음")

    monkeypatch.setattr(loop, "_start", no_mic)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="마이크 없음"):
            loop.start()
    assert calls == [1] and not loop.running


def test_no_mic_fails_before_loading_models(monkeypatch):
    """마이크가 없으면(클라우드 서버) 검출기 · 손 모델을 만들기 전에 끝나고, 그 이유를 기억한다.

    예전에는 손 모델을 만든 뒤 마이크에서 실패해, 닫히지 않은 손 모델이 나중에 다른 질문을 멈췄다.
    """
    import jarviseo.pointing.detector as detector_mod
    import jarviseo.voice.mic as mic_mod

    class NoMic:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise OSError("PortAudio: 기본 입력 장치 없음")

    made = []
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(mic_mod, "Microphone", NoMic)
    monkeypatch.setattr(detector_mod, "ObjectDetector", lambda *a, **k: made.append("detector"))
    loop = make_loop()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="기본 입력 장치 없음"):
            loop.start()
    assert made == [] and not loop.running


def test_voice_settings_before_start_are_kept_for_start():
    """켜기 전에 받은 환경설정은 들고 있다가 켤 때 쓴다(없는 칸은 그대로)."""
    loop = make_loop()
    loop.configure_voice(voice="nova", speed=1.3, volume=None)
    loop.configure_voice(volume=0.5)
    assert loop.voice_settings == {"voice": "nova", "speed": 1.3, "volume": 0.5}
