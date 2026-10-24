"""호출어 2단계 확인(WakeCheck)이 감지 직전 구간을 받아써서 맞게 통과·거절하는지 확인한다.

OpenAI API 와 마이크는 부르지 않는다. 가짜 client 와 블록을 흘리는 MicStream 을 쓴다.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from openai import OpenAIError

from jarviseo import config
from jarviseo.voice import wakecheck
from jarviseo.voice.mic import BLOCK, SAMPLE_RATE, MicStream
from jarviseo.voice.wakecheck import WakeCheck

BLOCK_SEC = BLOCK / SAMPLE_RATE


class FakeClient:
    """가짜 openai 클라이언트. 정해 둔 글자를 주고, hold 면 release 까지 붙잡는다."""

    def __init__(self, text="자비서.", error=None, hold=False):
        self.requests, self.text, self.error = [], text, error
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        self.release.wait(timeout=5)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text)


@pytest.fixture
def mic():
    """3초치 블록(값은 블록 번호)을 다 받아 둔 MicStream."""
    source = [(np.full(BLOCK, i, np.int16), i * BLOCK_SEC) for i in range(100)]
    with MicStream(source=source) as stream:
        list(stream.blocks(since=0.0))
        yield stream


def test_wake_word_passes(mic):
    """감지 직전 구간에 "자비서"가 받아써지면 통과한다."""
    verdict = WakeCheck(mic, client=FakeClient("자비서.")).start(2.0)

    assert verdict.passed() is True
    assert verdict.rejected() is False
    assert verdict.text == "자비서."


@pytest.mark.parametrize("text", ["자비스?", "재비서야", "Jarvis"])
def test_similar_spellings_of_wake_word_pass(mic, text):
    """받아쓰기가 "자비스"·"재비서야"처럼 써도 호출어로 친다(transcript.clean 과 같은 표기)."""
    assert WakeCheck(mic, client=FakeClient(text)).start(2.0).passed() is True


@pytest.mark.parametrize("text", ["아, 비서.", "다 비서?", "비, 서.", "차비서.", "비서."])
def test_broken_wake_word_passes(mic, text):
    """끊어 말한 "자, 비서"가 짧은 조각에서 바뀌어 받아써진 것도 호출어로 친다(eval 실제 녹음)."""
    assert WakeCheck(mic, client=FakeClient(text)).start(2.0).passed() is True


@pytest.mark.parametrize("text", ["김신조 사건.", "형사.", "대통령 비서실에서 밝혔다.", "자비."])
def test_tv_speech_is_rejected(mic, text):
    """TV 오탐 조각처럼 받아써진 말과, 문장 가운데 "비서"는 호출어로 치지 않는다."""
    assert WakeCheck(mic, client=FakeClient(text)).start(2.0).passed() is False


def test_other_speech_is_rejected(mic):
    """호출어가 없는 소리(TV 말소리)면 거절한다."""
    verdict = WakeCheck(mic, client=FakeClient("오늘 서울은 맑겠습니다.")).start(2.0)

    assert verdict.passed() is False
    assert verdict.rejected() is True
    assert verdict.text == "오늘 서울은 맑겠습니다."


def test_sends_clip_just_before_detection(mic):
    """감지 시각 앞 seconds 초만 보낸다. 힌트를 켜면 호출어를 prompt 로 준다."""
    client = FakeClient()
    WakeCheck(mic, seconds=0.6, hint=True, client=client).start(2.0).passed()
    WakeCheck(mic, seconds=0.6, hint=False, client=client).start(2.0).passed()

    with_hint, without = client.requests
    _, wav = with_hint["file"]
    samples = (len(wav) - 44) // 2  # WAV 헤더 44바이트, int16
    assert samples == pytest.approx(0.6 * SAMPLE_RATE, abs=BLOCK)
    assert with_hint["prompt"] == config.WAKE_WORD
    assert "prompt" not in without


def test_transcription_failure_passes(mic):
    """받아쓰기가 실패하면 통과로 본다. 인터넷이 끊겨도 호출은 돼야 한다."""
    verdict = WakeCheck(mic, client=FakeClient(error=OpenAIError("연결 끊김"))).start(2.0)

    assert verdict.passed() is True
    assert verdict.rejected() is False
    assert verdict.text is None


def test_slow_transcription_passes_after_timeout(mic, monkeypatch):
    """TIMEOUT_SEC 안에 받아쓰기가 안 끝나면 기다리지 않고 통과로 본다."""
    monkeypatch.setattr(wakecheck, "TIMEOUT_SEC", 0.05)
    client = FakeClient("오늘 서울은 맑겠습니다.", hold=True)
    verdict = WakeCheck(mic, client=client).start(2.0)

    assert verdict.rejected() is False  # 아직 결과가 없으면 거절하지 않는다
    assert verdict.passed() is True
    client.release.set()


def test_rejected_does_not_wait(mic):
    """rejected() 는 기다리지 않는다. 듣는 동안 블록마다 부른다(listen 의 stop)."""
    client = FakeClient("오늘 서울은 맑겠습니다.", hold=True)
    verdict = WakeCheck(mic, client=client).start(2.0)

    assert verdict.rejected() is False
    client.release.set()
    assert verdict.passed() is False
    assert verdict.rejected() is True


def test_clip_outside_history_passes_without_request():
    """마이크가 그 구간을 들고 있지 않으면 보내지 않고 통과로 본다."""
    client = FakeClient("오늘 서울은 맑겠습니다.")
    with MicStream(source=[]) as stream:
        verdict = WakeCheck(stream, client=client).start(10.0)

        assert verdict.passed() is True
    assert client.requests == []


def test_start_needs_mic():
    """mic 없이 만든 WakeCheck(녹음 파일 측정용)는 start() 를 못 부른다."""
    with pytest.raises(ValueError):
        WakeCheck(client=FakeClient()).start(1.0)
