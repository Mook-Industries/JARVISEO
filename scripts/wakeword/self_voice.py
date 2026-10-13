"""자비서 자기 목소리(TTS 대답)에 호출어 점수가 얼마나 나오는지 잰다.  담당: 문태현

대답 문장을 실제 TTS 로 합성해 16kHz 로 바꾸고, 감지기처럼 80ms 씩 모델에 넣어 최고 점수를 찍는다.
스피커·방 울림을 거치지 않은 소리라 참고용이다. 실제 재생은 voice_loop.py --wake 가
끝까지 읽은 대답마다 "재생 중 호출어 최고 점수"로 찍는다.

재생 중 임계값(config.WAKEWORD_PLAYBACK_THRESHOLD)을 평소와 같은 0.97 로 둔 근거다
(docs/experiments.md "끼어들기·턴 음성 기록 (D11)"). 대답에 "자비서"가 들어가면 당연히 깬다.

실행:
    python scripts/wakeword/self_voice.py
    python scripts/wakeword/self_voice.py --text "저는 자비서예요." --voice alloy
"""

import argparse

import numpy as np

from jarviseo import config
from jarviseo.voice.speaker import SAMPLE_RATE as TTS_RATE
from jarviseo.voice.tts import TextToSpeech
from jarviseo.voice.wakeword import FRAME, load_scorer

RATE = 16000
ANSWERS = [
    "지금부터 조금 긴 대답을 읽어 드릴게요. 중간에 끼어들 수 있는지 시험하려고 일부러 길게 만든 "
    "문장입니다. 읽는 도중에 호출어를 부르시면 바로 멈추고 새 질문을 들을게요. 부르지 않으시면 "
    "끝까지 읽은 다음 호출을 기다립니다. 마지막으로, 오늘 하루도 편안하게 보내세요.",
    "앞에 있는 건 우유 한 팩이에요. 유통기한은 십월 이십일까지고, 땅콩은 들어 있지 않아요.",
    "아까 보신 검은색 가방은 책상 위 왼쪽에 있었어요. 지갑은 그 옆 서랍 근처에서 봤어요.",
    "이 과자에는 밀, 대두, 우유가 들어 있어요. "
    "알레르기 성분 중 우유가 걸려서 드시지 않는 게 좋겠어요.",
    "네, 말씀하세요.",
]


def to_16k(pcm: bytes) -> np.ndarray:
    """TTS 의 24kHz PCM 을 모델이 받는 16kHz 로 바꾼다(선형 보간)."""
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    at = np.arange(0, len(x), TTS_RATE / RATE)
    return np.interp(at, np.arange(len(x)), x).astype(np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS 대답의 호출어 점수")
    parser.add_argument("--text", nargs="+", help="기본 대답 대신 잴 문장")
    parser.add_argument("--voice", default="coral", help="TTS 목소리 (TextToSpeech 기본과 같게)")
    parser.add_argument("--threshold", type=float, default=config.WAKEWORD_PLAYBACK_THRESHOLD)
    args = parser.parse_args()

    tts = TextToSpeech(voice=args.voice)
    pad = np.zeros(RATE, np.int16)
    peaks = []
    for text in args.text or ANSWERS:
        audio = np.concatenate([pad, to_16k(b"".join(tts.synthesize(text))), pad])
        score = load_scorer(config.WAKEWORD_MODEL)
        scores = [score(audio[i : i + FRAME]) for i in range(0, len(audio) - FRAME + 1, FRAME)]
        peaks.append(max(scores))
        over = sum(s >= args.threshold for s in scores)
        seconds = len(audio) / RATE
        print(f"{seconds:5.1f}s  최고 {max(scores):.3f}  넘은 프레임 {over}  {text[:30]}")
    print(f"대답 {len(peaks)}개, 최고 점수 {max(peaks):.3f} (재생 중 임계값 {args.threshold})")


if __name__ == "__main__":
    main()
