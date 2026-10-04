"""실제 마이크로 SpeechToText.listen() 을 확인한다.  담당: 문태현

말할 때마다 받아쓴 원문, 호출어·필러를 뺀 질문, 시각과 지연을 출력한다.
PR 에 "실제 발화 인식 결과"를 적을 때 쓴다. OpenAI 키가 필요하다.

실행:
    python scripts/check_listen.py        # 5번, 기본 마이크
    python scripts/check_listen.py 8 2    # 8번, 2번 장치
"""

import sys
import time

from jarviseo.voice import SpeechToText


def main() -> None:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    stt = SpeechToText(device=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    for i in range(rounds):
        print(f"\n[{i + 1}/{rounds}] 말해 보세요")
        utterance = stt.listen(on_speech_start=lambda t: print(f"  시작 {t:.3f}"), timeout=10.0)
        done = time.monotonic()
        print(f"  원문 {stt.last_raw_text!r}")
        if utterance is None:
            print("  질문 없음 (말이 없었거나, 정리하고 나니 비었거나, 받아쓰기 실패)")
            continue
        print(
            f"  질문 {utterance.text!r}  말 길이 {utterance.ended_at - utterance.started_at:.2f}s"
        )
        print(
            f"  받아쓰기 왕복 {stt.last_latency_ms:.0f}ms, "
            f"말 끝 → 글자 {(done - utterance.ended_at) * 1000:.0f}ms"
        )


if __name__ == "__main__":
    main()
