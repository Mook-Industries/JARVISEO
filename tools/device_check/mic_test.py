"""
JARVISEO - 마이크 테스트

USB 카메라에 달린 마이크가 쓸 만한지 확인합니다.
faster-whisper 가 기대하는 16kHz 모노 기준으로 녹음합니다.

준비:
    pip install sounddevice numpy
    (에러가 나면 macOS 에서는  brew install portaudio  먼저)

사용법:
    python mic_test.py              # 장치 목록만 보기
    python mic_test.py 2            # 2번 장치로 전체 테스트
    python mic_test.py 2 meter      # 2번 장치 레벨 미터만
"""

import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000     # faster-whisper 표준
CHANNELS    = 1         # 모노
SECONDS     = 5         # 녹음 길이

# 산출물 저장 위치.
# Path(__file__) 은 '이 파일 자신의 경로'라서, 어느 폴더에서 실행하든
# 항상 스크립트 옆 _captures/ 에 떨어진다.
OUT_DIR  = Path(__file__).resolve().parent / "_captures"
OUT_PATH = str(OUT_DIR / "mic_test.wav")


def list_devices():
    """입력 가능한 장치를 번호와 함께 출력한다."""
    print("\n=== 오디오 장치 목록 ===")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            mark = "  <- 입력 가능"
            print(f"[{i:2}] {dev['name']}{mark}")
            print(f"     채널 {dev['max_input_channels']} / "
                  f"기본 샘플레이트 {int(dev['default_samplerate'])}Hz")
    print("\n카메라 이름이 보이는 번호를 인자로 넣어 다시 실행하세요.")
    print("예:  python mic_test.py 2\n")


def level_meter(device, seconds=10):
    """실시간 입력 레벨을 막대로 보여준다."""
    print(f"\n{seconds}초간 레벨 미터를 띄웁니다. 말해보세요. (Ctrl+C 로 중단)")

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\n[경고] {status}")
        rms = float(np.sqrt(np.mean(indata ** 2)))
        level = min(int(rms * 400), 50)
        bar = "#" * level
        print(f"\r[{bar:<50}] rms {rms:.4f}", end="")

    try:
        with sd.InputStream(device=device, channels=CHANNELS,
                            samplerate=SAMPLE_RATE, callback=callback):
            sd.sleep(seconds * 1000)
    except KeyboardInterrupt:
        pass
    print("\n")


def record(device):
    """SECONDS 초 녹음해서 wav 로 저장하고, 품질 수치를 알려준다."""
    print(f"{SECONDS}초 녹음 시작 — 평소 말하는 크기로 또박또박 말해보세요.")
    print("추천 문장: \"자비서, 저거 뭐야?\"")

    audio = sd.rec(int(SECONDS * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=CHANNELS,
                   dtype="float32",
                   device=device)
    sd.wait()
    print("녹음 끝\n")

    mono = audio[:, 0]
    peak = float(np.max(np.abs(mono)))
    rms  = float(np.sqrt(np.mean(mono ** 2)))
    clip = int(np.sum(np.abs(mono) >= 0.999))

    # wav 저장 (16bit PCM)
    OUT_DIR.mkdir(exist_ok=True)
    pcm16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(OUT_PATH, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())

    print("=== 결과 ===")
    print(f"최대 진폭 (peak) : {peak:.3f}")
    print(f"평균 크기 (rms)  : {rms:.4f}")
    print(f"클리핑 샘플 수   : {clip}")
    print(f"저장 파일        : {OUT_PATH}\n")

    # 해석
    if peak < 0.02:
        print("판정: 소리가 거의 안 들어옵니다. 장치 번호나 마이크 권한을 확인하세요.")
    elif peak < 0.1:
        print("판정: 너무 작습니다. 시스템 설정 > 사운드 > 입력에서 입력 볼륨을 올리세요.")
    elif clip > 100:
        print("판정: 소리가 깨집니다(클리핑). 입력 볼륨을 낮추거나 입에서 떼세요.")
    else:
        print("판정: 양호합니다. STT 에 쓸 만한 수준입니다.")

    return mono


def play(audio):
    print("\n녹음한 소리를 재생합니다...")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
    print("재생 끝")


def main():
    if len(sys.argv) < 2:
        list_devices()
        return

    device = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "all"

    info = sd.query_devices(device)
    print(f"\n선택한 장치: [{device}] {info['name']}")

    if mode == "meter":
        level_meter(device)
        return

    level_meter(device, seconds=5)
    audio = record(device)
    play(audio)


if __name__ == "__main__":
    main()
