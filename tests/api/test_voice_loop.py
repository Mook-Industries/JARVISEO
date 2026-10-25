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
