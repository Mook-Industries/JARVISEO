"""Fake STT·TTS 가 실제 클래스와 같은 모양인지 확인한다.

그래프는 Fake 로 개발하다가 실제 클래스로 바꿔 끼운다. 메서드 이름이나 인자가
하나라도 다르면 바꿔 끼우는 순간 깨지므로, 공개 메서드의 시그니처를 그대로 비교한다.
"""

import inspect

import pytest

from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech, SpeechToText, TextToSpeech


@pytest.mark.parametrize(
    ("real", "fake"), [(SpeechToText, FakeSpeechToText), (TextToSpeech, FakeTextToSpeech)]
)
def test_fake_의_공개_메서드가_실제와_같다(real, fake):
    for name, method in vars(real).items():
        if callable(method) and not name.startswith("_"):
            assert inspect.signature(getattr(fake, name)) == inspect.signature(method), name


@pytest.mark.parametrize(
    ("real", "fake"), [(SpeechToText, FakeSpeechToText), (TextToSpeech, FakeTextToSpeech)]
)
def test_fake_도_실제와_같은_결과_속성을_가진다(real, fake):
    def last_attrs(obj):
        return {name for name in vars(obj) if name.startswith("last_")}

    # 실제 쪽은 키 없이 만들려고 가짜 client 를 넘긴다
    assert last_attrs(fake()) == last_attrs(real(client=object()))


def test_fake_stt_는_돌려주기_전에_시작_콜백을_부른다():
    starts = []
    stt = FakeSpeechToText(["저거 뭐야?"])

    utterance = stt.listen(on_speech_start=starts.append)

    assert utterance.text == "저거 뭐야?"
    assert starts == [utterance.started_at]
    assert utterance.started_at <= utterance.ended_at
    assert stt.listen() is None  # 문장을 다 쓰면 말이 없는 것으로 친다


def test_fake_tts_는_받은_문장을_모은다():
    tts = FakeTextToSpeech()

    assert tts.speak("안녕하세요") >= 0
    assert tts.spoken == ["안녕하세요"]
