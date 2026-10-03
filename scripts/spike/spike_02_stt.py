"""스파이크 2 — 마이크 → VAD → STT, 발화 시작 시각 출력.  담당: 문태현

확인할 것:
  1. sounddevice 로 마이크가 열리는가 (윈도우에서도)
  2. 발화 '시작' 시각을 잡을 수 있는가  <- 이게 제일 중요하다
  3. 한국어가 받아써지는가, 왕복 몇 ms 걸리는가

VAD 는 소리 크기(RMS) 문턱값 방식이다. 처음 0.5초로 잡음 크기를 재고, 30ms 블록이
그 3배를 넘으면 말 시작, 0.8초 조용하면 말 끝으로 본다.
오차 = VAD 가 잡은 시작 - 같은 녹음을 1ms 단위로 다시 훑어 찾은 시작.

실행:
    python scripts/spike/spike_02_stt.py      # 기본 마이크
    python scripts/spike/spike_02_stt.py 2    # 2번 장치 (번호는 tools/device_check/mic_test.py)
"""

import io
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

from jarviseo import config

SR, BLOCK = 16000, 480  # 30ms 블록
CALIB_SEC, SILENCE_SEC, MAX_SEC = 0.5, 0.8, 10


def rms(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    return np.sqrt(np.mean(x**2, axis=axis))


def main() -> None:
    device = int(sys.argv[1]) if len(sys.argv) > 1 else None
    with sd.InputStream(samplerate=SR, blocksize=BLOCK, device=device, channels=1) as mic:
        print("잡음 재는 중... 0.5초만 조용히")
        blocks = [mic.read(BLOCK)[0][:, 0] for _ in range(round(CALIB_SEC * SR / BLOCK))]
        latency = mic.latency
        # 녹음 첫 샘플이 마이크에 들어온 시각. 이후 시각은 샘플 수로 센다.
        t_rec = time.monotonic() - len(blocks) * BLOCK / SR - latency
        thr = max(float(rms(np.concatenate(blocks))), 1e-3) * 3
        print(f'지금 말하세요. 예: "자비서, 저거 뭐야?"  (문턱값 {thr:.4f})')
        loud = []  # 문턱값을 넘은 블록 번호
        while len(blocks) < MAX_SEC * SR / BLOCK:
            blocks.append(mic.read(BLOCK)[0][:, 0])
            if rms(blocks[-1]) > thr:
                loud.append(len(blocks) - 1)
            elif loud and (len(blocks) - 1 - loud[-1]) * BLOCK >= SILENCE_SEC * SR:
                break
    if not loud:
        sys.exit("말소리를 못 잡았다. 장치 번호와 입력 볼륨을 확인할 것.")

    audio, s0, s1 = np.concatenate(blocks), loud[0] * BLOCK, (loud[-1] + 1) * BLOCK
    lo = max(s0 - SR // 5, 0)  # VAD 시작 200ms 앞부터 그 블록 끝까지 1ms(16샘플)씩
    onset = lo + int(np.argmax(rms(audio[lo : s0 + BLOCK].reshape(-1, 16), axis=1) > thr)) * 16
    started_at, ended_at = t_rec + s0 / SR, t_rec + s1 / SR
    print(f"started_at {started_at:.3f}  녹음 +{s0 / SR:.2f}s, 오차 {(s0 - onset) / 16:+.0f}ms")
    print(f"  입력 지연 {latency * 1000:.0f}ms 는 장치가 알려준 값으로 이미 뺐다")
    print(f"ended_at   {ended_at:.3f}  끝 시각으로 프레임을 고르면 {(s1 - s0) / SR:.2f}s 늦다")

    wav = io.BytesIO()
    sf.write(wav, audio[lo : s1 + SR // 5], SR, format="WAV", subtype="PCM_16")
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    sent_at = time.monotonic()
    text = client.audio.transcriptions.create(
        model=config.STT_MODEL, file=("speech.wav", wav.getvalue()), language="ko"
    ).text
    done = time.monotonic()
    print(f"[{config.STT_MODEL}] {text}")
    print(f"STT 왕복 {(done - sent_at) * 1000:.0f}ms, 말 끝→글자 {(done - ended_at) * 1000:.0f}ms")


if __name__ == "__main__":
    main()
