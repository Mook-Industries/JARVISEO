"""Enter 를 누르고 말하면, 받아쓴 질문을 그대로 목소리로 되돌려 준다.  담당: 문태현

카메라·VLM 없이 음성 입출력만 잇는 개발용 루프다(STT → 에코 → TTS).
턴마다 단계별 지연을 찍고, 끝낼 때 중앙값을 모아 찍는다. PR 에 지연을 적을 때 쓴다.
OpenAI 키가 필요하다. q 를 입력하면 끝난다.

실행:
    python scripts/voice_loop.py          # 기본 마이크·스피커
    python scripts/voice_loop.py 1 4      # 1번 마이크, 4번 스피커

찍는 값 (ms)
    말끝 대기    말이 끝나고 VAD 가 끝으로 판정해 녹음을 멈추기까지 (무음 700ms 가 대부분)
    STT 왕복     받아쓰기 요청을 보내고 글자를 받기까지
    TTS 첫 조각  합성 요청을 보내고 첫 PCM 조각을 받기까지 (스피커 출력 지연 약 0.2초는 빠짐)
    재생         첫 조각을 받고 마지막 소리가 다 나오기까지
    말 끝 → 첫 조각  사용자가 말을 멈추고 대답이 나오기 시작하기까지. 위 셋을 더한 값
"""

import statistics
import sys
import time

from jarviseo.voice import SpeechToText, TextToSpeech

STAGES = ("말끝 대기", "STT 왕복", "TTS 첫 조각", "재생", "말 끝 → 첫 조각")


def main() -> None:
    mic = int(sys.argv[1]) if len(sys.argv) > 1 else None
    speaker = int(sys.argv[2]) if len(sys.argv) > 2 else None
    stt, tts = SpeechToText(device=mic), TextToSpeech(device=speaker)
    turns: list[dict[str, float]] = []
    while input("\nEnter 를 누르고 말하세요 (끝내려면 q): ").strip().lower() != "q":
        utterance = stt.listen(timeout=8.0)
        if utterance is None:
            print(f"  질문 없음 (원문 {stt.last_raw_text!r})")
            continue
        print(f"  질문 {utterance.text!r}")
        heard = time.monotonic()
        if tts.speak(utterance.text) is None:
            print("  합성 실패, 소리를 못 냈다")
            continue
        spoken = time.monotonic()

        to_text = (heard - utterance.ended_at) * 1000
        turn = {
            "말끝 대기": to_text - stt.last_latency_ms,
            "STT 왕복": stt.last_latency_ms,
            "TTS 첫 조각": tts.last_latency_ms,
            "재생": (spoken - heard) * 1000 - tts.last_latency_ms,
            "말 끝 → 첫 조각": to_text + tts.last_latency_ms,
        }
        turns.append(turn)
        print("  " + ", ".join(f"{name} {turn[name]:.0f}" for name in STAGES))

    if turns:
        print(f"\n중앙값 ({len(turns)}턴, ms)")
        for name in STAGES:
            print(f"  {name} {statistics.median(t[name] for t in turns):.0f}")


if __name__ == "__main__":
    main()
