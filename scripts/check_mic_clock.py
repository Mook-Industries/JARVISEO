"""마이크 블록 시각이 실제 시각을 따라가는지 잰다 (#69).  담당: 문태현

마이크를 N분 열어 두고 블록마다 세 값을 견준다.
- 상한: 블록을 받은 monotonic 시각 − 블록 길이 − 장치 입력 지연. 실제 시각은 이보다 늦을 수 없다
- 고치기 전: 첫 블록에서 정한 시각 + 샘플 수 (예전 Microphone.blocks())
- 고친 뒤: BlockClock (지금 Microphone.blocks())

10초마다 최근 1초 동안 "상한 − 시각" 의 최솟값(= 제때 받은 블록 기준으로 얼마나 늦었나)을 찍는다.
고치기 전 값이 시간에 비례해 늘면 장치가 16kHz 보다 느린 것이고, 그 기울기가 장치 속도 차이다.
소리는 저장하지 않는다.

실행:
    python scripts/check_mic_clock.py              # 기본 마이크, 10분
    python scripts/check_mic_clock.py 1 --minutes 3
"""

import argparse
import time
from collections import deque

import sounddevice as sd

from jarviseo.voice.mic import BLOCK, BLOCK_SEC, SAMPLE_RATE, BlockClock

REPORT_SEC = 10.0
WINDOW_SEC = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("device", nargs="?", type=int, help="마이크 장치 번호. 없으면 기본 장치")
    parser.add_argument("--minutes", type=float, default=10.0)
    args = parser.parse_args()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCK, device=args.device, channels=1, dtype="int16"
    )
    name = sd.query_devices(stream.device)["name"]
    print(f"장치 {stream.device} {name!r}, 입력 지연 {stream.latency * 1000:.0f}ms")
    print(f"{args.minutes:g}분 잰다. Ctrl+C 로 일찍 끝낼 수 있다.\n")
    print(f"{'경과':>6} {'고치기 전 늦음':>12} {'고친 뒤 늦음':>11} {'overflow':>9}")

    clock = BlockClock(stream.latency)
    recent: deque[tuple[float, float, float]] = deque()  # (받은 시각, 고치기 전 늦음, 고친 뒤 늦음)
    rows = []
    overflows = 0
    with stream:
        _, overflowed = stream.read(BLOCK)
        start = time.monotonic()
        raw0 = start - BLOCK_SEC - stream.latency
        fixed = clock.stamp(start, overflowed)
        n, next_report = 0, REPORT_SEC
        try:
            while time.monotonic() - start < args.minutes * 60:
                _, overflowed = stream.read(BLOCK)
                read_at = time.monotonic()
                n += 1
                overflows += overflowed
                fixed = clock.stamp(read_at, overflowed)
                latest = read_at - BLOCK_SEC - stream.latency
                recent.append((read_at, latest - (raw0 + n * BLOCK_SEC), latest - fixed))
                while read_at - recent[0][0] > WINDOW_SEC:
                    recent.popleft()
                elapsed = read_at - start
                if elapsed >= next_report:
                    raw_lag = min(r for _, r, _ in recent)
                    fixed_lag = min(f for _, _, f in recent)
                    rows.append((elapsed, raw_lag, fixed_lag))
                    print(
                        f"{elapsed:5.0f}s {raw_lag * 1000:10.0f}ms {fixed_lag * 1000:9.0f}ms "
                        f"{overflows:9d}"
                    )
                    next_report += REPORT_SEC
        except KeyboardInterrupt:
            pass

    if len(rows) < 2:
        print("너무 짧아서 기울기를 못 낸다")
        return
    (t1, r1, _), (t2, r2, _) = rows[0], rows[-1]
    slope = (r2 - r1) / (t2 - t1)
    print(f"\n{t2:.0f}초 동안 블록 {n + 1}개, overflow {overflows}번")
    print(f"고치기 전: 1초에 {slope * 1000:.1f}ms 늦어짐 ({slope * 100:.2f}%), ", end="")
    print(f"끝에서 {r2 * 1000:.0f}ms")
    worst = max(abs(f) for _, _, f in rows)
    print(f"고친 뒤: 늦음 최대 {worst * 1000:.0f}ms (10초마다 찍은 값 중)")


if __name__ == "__main__":
    main()
