"""Enter 를 누르고 말하면, 받아쓴 질문을 그대로 목소리로 되돌려 준다.  담당: 문태현

카메라·VLM 없이 음성 입출력만 잇는 개발용 루프다(STT → 에코 → TTS).
OpenAI 키가 필요하다. q 를 입력하면 끝난다.

실행:
    python scripts/voice_loop.py          # 기본 마이크·스피커
    python scripts/voice_loop.py 1 4      # 1번 마이크, 4번 스피커
"""

import sys

from jarviseo.voice import SpeechToText, TextToSpeech


def main() -> None:
    mic = int(sys.argv[1]) if len(sys.argv) > 1 else None
    speaker = int(sys.argv[2]) if len(sys.argv) > 2 else None
    stt, tts = SpeechToText(device=mic), TextToSpeech(device=speaker)
    while input("\nEnter 를 누르고 말하세요 (끝내려면 q): ").strip().lower() != "q":
        utterance = stt.listen(timeout=8.0)
        if utterance is None:
            print(f"  질문 없음 (원문 {stt.last_raw_text!r})")
            continue
        print(f"  질문 {utterance.text!r}")
        if tts.speak(utterance.text) is None:
            print("  합성 실패, 소리를 못 냈다")


if __name__ == "__main__":
    main()
