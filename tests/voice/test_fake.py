"""Fake STT·TTS 가 실제 클래스와 같은 모양인지 확인한다.

그래프는 Fake 로 개발하다가 실제 클래스로 바꿔 끼운다. 메서드 이름이나 인자가
하나라도 다르면 바꿔 끼우는 순간 깨지므로, 공개 메서드의 시그니처를 그대로 비교한다.
"""

import inspect
import time

import pytest

from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech, SpeechToText, TextToSpeech


@pytest.mark.parametrize(
    ("real", "fake"), [(SpeechToText, FakeSpeechToText), (TextToSpeech, FakeTextToSpeech)]
)
def test_fake_public_methods_match_real(real, fake):
    """Fake 의 공개 메서드가 실제와 같다."""
    for name, method in vars(real).items():
        if callable(method) and not name.startswith("_"):
            assert inspect.signature(getattr(fake, name)) == inspect.signature(method), name


@pytest.mark.parametrize(
    ("real", "fake"), [(SpeechToText, FakeSpeechToText), (TextToSpeech, FakeTextToSpeech)]
)
def test_fake_has_same_result_attributes(real, fake):
    """Fake 도 실제와 같은 결과 속성을 가진다."""

    def last_attrs(obj):
        return {name for name in vars(obj) if name.startswith("last_")}

    # 실제 쪽은 키 없이 만들려고 가짜 client 를 넘긴다
    assert last_attrs(fake()) == last_attrs(real(client=object()))


def test_fake_stt_calls_start_callback_before_return():
    """Fake STT 는 돌려주기 전에 시작 콜백을 부른다."""
    starts = []
    stt = FakeSpeechToText(["저거 뭐야?"])

    utterance = stt.listen(on_speech_start=starts.append)

    assert utterance.text == "저거 뭐야?"
    assert starts == [utterance.started_at]
    assert utterance.started_at <= utterance.ended_at
    assert stt.listen() is None  # 문장을 다 쓰면 말이 없는 것으로 친다


def test_fake_stt_can_set_speech_start():
    """사진 폴더(0초, 0.033초…)로 돌릴 때 말 시작 시각을 사진 시각에 맞춘다."""
    starts = []
    stt = FakeSpeechToText(["저거 뭐야?", "자비서", "컵이요", "뭐야?"], starts=[0.5, 1.0, 2.0])

    first = stt.listen(on_speech_start=starts.append)
    assert (first.started_at, first.ended_at) == (0.5, 1.5)
    assert starts == [0.5]
    assert stt.listen() is None  # 호출어뿐인 문장도 시각 하나를 쓴다
    assert stt.listen().started_at == 2.0

    before = time.monotonic()
    last = stt.listen()  # 정해 둔 시각을 다 쓰면 지금 시각
    assert before <= last.started_at <= time.monotonic()


def test_fake_stt_accepts_one_letter_with_min_letters():
    """Fake STT 도 `min_letters` 를 주면 한 글자 대답을 받는다."""
    assert FakeSpeechToText(["컵"]).listen() is None
    assert FakeSpeechToText(["컵"]).listen(min_letters=1).text == "컵"


def test_fake_stt_stop_keeps_sentence_for_next_listen():
    """Fake STT 는 `stop()` 이 True 면 None 을 주고, 그 문장은 다음 listen() 이 받는다."""
    stt = FakeSpeechToText(["이건 뭐야?"])

    assert stt.listen(stop=lambda: True) is None
    assert stt.listen().text == "이건 뭐야?"  # 실제 listen(since=...) 이 지나간 소리를 다시 읽듯


def test_fake_tts_collects_sentences():
    """Fake TTS 는 받은 문장을 모은다."""
    tts = FakeTextToSpeech()

    assert tts.speak("안녕하세요") >= 0
    assert tts.spoken == ["안녕하세요"]


def test_fake_tts_interrupts_sentence_once():
    """Fake TTS 는 `interrupt` 글자가 든 문장을 한 번 끊긴 것으로 친다."""
    tts = FakeTextToSpeech(interrupt=["왼쪽 컵이요"])

    tts.speak("왼쪽 컵이요, 오른쪽 컵이요?")
    assert tts.last_interrupted is True
    tts.speak("왼쪽 컵이요, 오른쪽 컵이요?")  # 같은 글자는 한 번만 끊는다
    assert tts.last_interrupted is False
    assert tts.spoken == ["왼쪽 컵이요, 오른쪽 컵이요?"] * 2


def test_fake_tts_fails_sentence_once():
    """Fake TTS 는 `fail` 글자가 든 문장을 한 번 소리 없이 실패한 것으로 친다(None)."""
    tts = FakeTextToSpeech(fail=["왼쪽 컵이요"])

    assert tts.speak("왼쪽 컵이요, 오른쪽 컵이요?") is None
    assert (tts.last_latency_ms, tts.last_interrupted) == (None, False)
    assert tts.speak("왼쪽 컵이요, 오른쪽 컵이요?") == 0.0  # 같은 글자는 한 번만 실패한다
    assert tts.failed == ["왼쪽 컵이요, 오른쪽 컵이요?"]
    assert tts.spoken == ["왼쪽 컵이요, 오른쪽 컵이요?"]
