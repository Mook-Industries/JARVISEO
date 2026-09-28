"""
JARVISEO - 마이크 SNR 측정 + 게인 조정

소리가 작을 때, 그게 진짜 문제인지 판단합니다.
절대 크기보다 '신호 대 잡음비(SNR)'가 중요하기 때문입니다.

준비:
    pip install sounddevice numpy

사용법:
    python mic_snr_test.py 2          # 2번 장치, 게인 자동
    python mic_snr_test.py 2 10       # 2번 장치, 게인 10배 고정
"""

import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS    = 1
NOISE_SEC   = 2      # 무음 측정 시간
SPEECH_SEC  = 5      # 발화 측정 시간
# 게인 자동 계산의 목표 peak.
# 1.0 으로 맞추면 조금만 크게 말해도 바로 클리핑되므로,
# 0.3 만큼 여유(헤드룸)를 두고 0.7 로 잡았다.
TARGET_PEAK = 0.7

# 산출물 저장 위치.
# Path(__file__) 은 '이 파일 자신의 경로'라서, 어느 폴더에서 실행하든
# 항상 스크립트 옆 _captures/ 에 떨어진다.
OUT_DIR    = Path(__file__).resolve().parent / "_captures"
OUT_RAW    = str(OUT_DIR / "mic_raw.wav")
OUT_GAINED = str(OUT_DIR / "mic_gained.wav")


def capture(seconds, device):
    audio = sd.rec(int(seconds * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=CHANNELS,
                   dtype="float32", device=device)
    sd.wait()
    return audio[:, 0]


def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def save_wav(path, mono):
    OUT_DIR.mkdir(exist_ok=True)
    pcm16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16.tobytes())


def countdown(msg, seconds=3):
    import time
    for i in range(seconds, 0, -1):
        print(f"\r{msg} {i}...", end="", flush=True)
        time.sleep(1)
    print(f"\r{msg} 시작!      ")


def main():
    if len(sys.argv) < 2:
        print("사용법: python mic_snr_test.py <장치번호> [게인]")
        print("\n입력 가능한 장치:")
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"  [{i:2}] {d['name']}")
        return

    device = int(sys.argv[1])
    manual_gain = float(sys.argv[2]) if len(sys.argv) > 2 else None

    info = sd.query_devices(device)
    print(f"장치: [{device}] {info['name']}\n")

    # ── 1. 노이즈 플로어 ──────────────────────
    countdown(f"{NOISE_SEC}초간 아무 말도 하지 마세요.")
    noise = capture(NOISE_SEC, device)
    noise_rms = rms(noise)
    print(f"  잡음 바닥 rms : {noise_rms:.5f}\n")

    # ── 2. 발화 ──────────────────────────────
    countdown(f"{SPEECH_SEC}초간 평소 크기로 말하세요. (예: 자비서, 저거 뭐야?)")
    speech = capture(SPEECH_SEC, device)
    speech_rms = rms(speech)
    speech_peak = float(np.max(np.abs(speech)))
    print(f"  발화 rms      : {speech_rms:.5f}")
    print(f"  발화 peak     : {speech_peak:.5f}\n")

    # ── 3. SNR ───────────────────────────────
    if noise_rms < 1e-9:
        snr_db = 99.0
    else:
        snr_db = 20 * np.log10(speech_rms / noise_rms)

    # ── 4. 게인 ──────────────────────────────
    auto_gain = TARGET_PEAK / speech_peak if speech_peak > 1e-9 else 1.0
    gain = manual_gain if manual_gain else auto_gain

    gained = np.clip(speech * gain, -1.0, 1.0)
    clipped = int(np.sum(np.abs(speech * gain) >= 1.0))

    save_wav(OUT_RAW, speech)
    save_wav(OUT_GAINED, gained)

    # ── 5. 결과 ──────────────────────────────
    print("=" * 46)
    print(f"  SNR            : {snr_db:5.1f} dB")
    print(f"  권장 게인      : {auto_gain:5.1f} 배")
    print(f"  적용한 게인    : {gain:5.1f} 배")
    print(f"  게인 후 클리핑 : {clipped} 샘플")
    print(f"  저장           : {OUT_RAW} / {OUT_GAINED}")
    print("=" * 46)

    print("\n[SNR 판정]")
    if snr_db >= 30:
        print("  훌륭합니다. 소리가 작아도 게인만 주면 STT 에 문제없습니다.")
    elif snr_db >= 20:
        print("  양호합니다. 조용한 곳에서는 충분히 쓸 수 있습니다.")
    elif snr_db >= 12:
        print("  보통입니다. 조용한 실내는 되지만 카페 같은 곳에서는 흔들립니다.")
    else:
        print("  나쁩니다. 게인을 올려도 잡음이 같이 커집니다.")
        print("  → 마이크를 입에 더 가까이 두거나, 별도 마이크를 고려하세요.")

    if clipped > 100:
        print("\n  주의: 게인이 너무 셉니다. 숫자를 낮춰서 다시 해보세요.")

    print(f"\n두 파일을 들어보세요. {OUT_GAINED} 쪽이 쓸 만하면 그대로 가면 됩니다.")
    print("재생합니다...")
    sd.play(gained, SAMPLE_RATE)
    sd.wait()


if __name__ == "__main__":
    main()
