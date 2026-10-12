"""음성 입출력 — 담당: 문태현.

흐름: 호출어 감지 -> 마이크·VAD 로 발화 구간과 시작 시각 -> STT -> (처리) -> TTS

그래프는 ``SpeechToText.listen()`` 과 ``TextToSpeech.speak()`` 만 부른다.
재생 중에 STT 쪽 마이크를 막는 게이트(``gate.py``)는 둘이 알아서 같이 쓴다.
호출어 감지(``WakeWordDetector``)와 STT 는 ``MicStream`` 하나를 같이 읽는다.
실제 구현이 끝나기 전이나 CI 에서는 모양이 같은 Fake 를 끼운다.
"""

from jarviseo.voice.fake import FakeSpeechToText, FakeTextToSpeech
from jarviseo.voice.stt import SpeechToText
from jarviseo.voice.tts import TextToSpeech
from jarviseo.voice.wakeword import WakeWordDetector

__all__ = [
    "FakeSpeechToText",
    "FakeTextToSpeech",
    "SpeechToText",
    "TextToSpeech",
    "WakeWordDetector",
]
