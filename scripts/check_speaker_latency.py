"""스피커 스트림을 여는 시간과 출력 지연을 잰다. 소리는 내지 않는다(무음만 쓴다).  담당: 문태현

시작음(#104)은 "자비서"를 알아들은 순간 바로 나야 해서, 출력 버퍼 크기(latency)에 따라
첫 소리가 얼마나 늦는지 본다. 첫 소리 ≈ 열기 + 출력 지연이다. TTS 는 "high", 시작음·종료음은 "low".

실행:
    python scripts/check_speaker_latency.py        # 기본 스피커
    python scripts/check_speaker_latency.py 4      # 4번 스피커
"""

import argparse
import statistics
import time

import sounddevice as sd

from jarviseo.voice.speaker import PIECE_BYTES, SAMPLE_RATE


def main() -> None:
    parser = argparse.ArgumentParser(description="스피커 열기 시간과 출력 지연")
    parser.add_argument("speaker", type=int, nargs="?", help="sounddevice 출력 장치 번호")
    parser.add_argument("--n", type=int, default=8, help="설정마다 몇 번 열지")
    args = parser.parse_args()

    device = sd.query_devices(args.speaker, kind="output")
    print(f"{device['name']} ({sd.query_hostapis(device['hostapi'])['name']})")
    silent = bytes(PIECE_BYTES)  # 20ms 무음
    for latency in ("high", "low"):
        opens, lags = [], []
        for _ in range(args.n):
            asked = time.perf_counter()
            with sd.RawOutputStream(
                samplerate=SAMPLE_RATE,
                device=args.speaker,
                channels=1,
                dtype="int16",
                latency=latency,
            ) as out:
                opens.append((time.perf_counter() - asked) * 1000)
                out.write(silent)
                lags.append(out.latency * 1000)
        opened, lag = statistics.median(opens), statistics.median(lags)
        print(
            f"{latency:>4}: 열기 {opened:.0f}ms + 출력 지연 {lag:.0f}ms"
            f" = 첫 소리까지 약 {opened + lag:.0f}ms"
        )


if __name__ == "__main__":
    main()
