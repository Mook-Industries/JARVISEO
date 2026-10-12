"""말하면 받아쓴 질문을 그대로 목소리로 되돌려 준다.  담당: 문태현

카메라·VLM 없이 음성 입출력만 잇는 개발용 루프다(STT → 에코 → TTS).
턴마다 단계별 지연을 찍고, 끝낼 때 중앙값을 모아 찍는다. PR 에 지연을 적을 때 쓴다.
OpenAI 키가 필요하다.

실행:
    python scripts/voice_loop.py              # Enter 를 누르고 말한다. q 로 끝
    python scripts/voice_loop.py 1 4          # 1번 마이크, 4번 스피커
    python scripts/voice_loop.py --wake       # 호출어를 부르고 이어서 말한다. Ctrl+C 로 끝

--wake 는 config.WAKEWORD_MODEL(기본 data/models/jarviseo.onnx)로 호출어를 기다린다.
"자비서, 저거 뭐야?" 처럼 부른다. 호출어만 부르고 멈추면 "네, 말씀하세요."라고 되묻고 다시 듣는다.

찍는 값 (ms)
    말끝 대기    말이 끝나고 VAD 가 끝으로 판정해 녹음을 멈추기까지 (무음 700ms 가 대부분)
    STT 왕복     받아쓰기 요청을 보내고 글자를 받기까지
    TTS 첫 조각  합성 요청을 보내고 첫 PCM 조각을 받기까지 (스피커 출력 지연 약 0.2초는 빠짐)
    재생         첫 조각을 받고 마지막 소리가 다 나오기까지
    말 끝 → 첫 조각  사용자가 말을 멈추고 대답이 나오기 시작하기까지. 위 셋을 더한 값
"""

import argparse
import queue
import statistics
import time

from jarviseo import config
from jarviseo.types import Utterance
from jarviseo.voice import SpeechToText, TextToSpeech, WakeWordDetector
from jarviseo.voice.mic import MicStream

STAGES = ("말끝 대기", "STT 왕복", "TTS 첫 조각", "재생", "말 끝 → 첫 조각")


def echo(stt: SpeechToText, tts: TextToSpeech, utterance: Utterance) -> dict[str, float] | None:
    """질문을 그대로 읽어 주고 단계별 지연(ms)을 돌려준다. 소리를 못 냈으면 None."""
    print(f"  질문 {utterance.text!r}")
    heard = time.monotonic()
    if tts.speak(utterance.text) is None:
        print("  합성 실패, 소리를 못 냈다")
        return None
    spoken = time.monotonic()

    to_text = (heard - utterance.ended_at) * 1000
    turn = {
        "말끝 대기": to_text - stt.last_latency_ms,
        "STT 왕복": stt.last_latency_ms,
        "TTS 첫 조각": tts.last_latency_ms,
        "재생": (spoken - heard) * 1000 - tts.last_latency_ms,
        "말 끝 → 첫 조각": to_text + tts.last_latency_ms,
    }
    print("  " + ", ".join(f"{name} {turn[name]:.0f}" for name in STAGES))
    return turn


def enter_loop(stt: SpeechToText, tts: TextToSpeech, turns: list) -> None:
    while input("\nEnter 를 누르고 말하세요 (끝내려면 q): ").strip().lower() != "q":
        utterance = stt.listen(timeout=8.0)
        if utterance is None:
            print(f"  질문 없음 (원문 {stt.last_raw_text!r})")
        elif turn := echo(stt, tts, utterance):
            turns.append(turn)


def wake_loop(stt: SpeechToText, tts: TextToSpeech, mic: MicStream, turns: list) -> None:
    found: queue.Queue[tuple[float, float]] = queue.Queue()
    detector = WakeWordDetector(mic, on_detect=lambda t, score: found.put((t, score)))
    detector.start()
    print(f"\n호출어({config.WAKEWORD_MODEL})를 부르고 이어서 말하세요. 끝내려면 Ctrl+C")
    try:
        while True:
            try:
                # 윈도우에서는 시간 제한 없이 기다리면 Ctrl+C 가 안 먹는다
                t, score = found.get(timeout=0.5)
            except queue.Empty:
                continue
            print(f"\n호출어 감지 (점수 {score:.2f})")
            utterance = stt.listen(timeout=2.0, since=t)
            if utterance is None:
                # 호출어만 부르고 멈췄다. 되묻고 다시 듣는다(재생 뒤 0.3초는 게이트가 막는다)
                tts.speak("네, 말씀하세요.")
                utterance = stt.listen(timeout=5.0)
            if utterance is None:
                print(f"  질문 없음 (원문 {stt.last_raw_text!r})")
            elif turn := echo(stt, tts, utterance):
                turns.append(turn)
            # 대답하는 동안 들어온 감지는 버린다. 재생 중 끼어들기는 다음 이슈에서 다룬다.
            while not found.empty():
                found.get_nowait()
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="음성 루프 (STT → 에코 → TTS)")
    parser.add_argument("mic", type=int, nargs="?", help="sounddevice 입력 장치 번호")
    parser.add_argument("speaker", type=int, nargs="?", help="sounddevice 출력 장치 번호")
    parser.add_argument("--wake", action="store_true", help="Enter 대신 호출어로 시작한다")
    args = parser.parse_args()

    turns: list[dict[str, float]] = []
    tts = TextToSpeech(device=args.speaker)
    if args.wake:
        with MicStream(args.mic) as mic:
            wake_loop(SpeechToText(mic=mic), tts, mic, turns)
    else:
        enter_loop(SpeechToText(device=args.mic), tts, turns)

    if turns:
        print(f"\n중앙값 ({len(turns)}턴, ms)")
        for name in STAGES:
            print(f"  {name} {statistics.median(t[name] for t in turns):.0f}")


if __name__ == "__main__":
    main()
