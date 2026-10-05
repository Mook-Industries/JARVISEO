"""실제로 TTS 를 합성해 첫 소리까지 시간(TTFB)을 재고 들어 본다.  담당: 문태현

속도(SLOW/NORMAL/FAST)마다 같은 문장을 합성해 바로 재생한다. OpenAI 키가 필요하다.
PR 에 "실제 합성의 TTFB"를 적을 때 쓴다.

실행:
    python scripts/check_tts.py                  # 기본 문장
    python scripts/check_tts.py "읽어 볼 문장"
"""

import sys

import numpy as np
import sounddevice as sd

from jarviseo.voice import TextToSpeech
from jarviseo.voice.tts import SAMPLE_RATE

TEXT = "이 과자에는 우유와 대두가 들어 있어요. 나트륨은 1,200mg으로 하루 권장량의 60%예요."


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else TEXT
    for speed in ("SLOW", "NORMAL", "FAST"):
        tts = TextToSpeech(speed=speed)
        audio = np.frombuffer(b"".join(tts.synthesize(text)), dtype="<i2")
        took = len(audio) / SAMPLE_RATE
        print(f"[{speed:6}] 첫 소리까지 {tts.last_latency_ms:.0f}ms, 소리 길이 {took:.2f}s")
        sd.play(audio, SAMPLE_RATE)
        sd.wait()


if __name__ == "__main__":
    main()
