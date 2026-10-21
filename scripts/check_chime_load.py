"""파이썬이 바쁠 때 시작음·종료음이 끊기는지 잰다. 실제 스피커로 소리가 난다.  담당: 문태현

같은 프로세스의 다른 파이썬 스레드가 GIL 을 잡고 바쁘면, 20ms 씩 나눠 쓰는 재생은 조각마다
GIL 을 다시 받느라 늦어서 출력 버퍼가 빈다(소리가 끊기며 늘어짐). 통째로 한 번에 쓰면
write() 가 GIL 을 놓고 C 안에서 기다려서 끊기지 않는다. 시작음·종료음은 통째로 쓴다.

부하: none / cpu(다른 프로세스가 코어 수만큼 바쁨) / gil(같은 프로세스 스레드 2개가 GIL 을
잡음) / both. "소리가 빈 시간"은 k 번째 조각을 쓰기 시작한 시각이 앞 조각들이 다 나갔어야 할
시각을 넘은 만큼의 최댓값이다.
부하가 없어도 스트림이 시작되는 시간 때문에 30ms 쯤 나온다(들리지 않는다).

실행:
    python scripts/check_chime_load.py              # 부하 4가지 × 20ms씩 / 한 번에, 6번씩
    python scripts/check_chime_load.py --listen     # gil 부하에서 둘을 번갈아 내서 귀로 비교
"""

import argparse
import multiprocessing as mp
import os
import threading
import time

import sounddevice as sd

from jarviseo.voice.chime import Chime
from jarviseo.voice.speaker import PIECE_BYTES, SAMPLE_RATE, Speaker

WAYS = (("20ms씩", PIECE_BYTES), ("한 번에", None))
_writes: list[tuple[float, int]] = []  # (쓰기 시작 시각, 바이트 수)


class _TimedStream(sd.RawOutputStream):
    """write() 를 시작한 시각과 크기를 남기는 출력 스트림."""

    def write(self, data):
        _writes.append((time.perf_counter(), len(data)))
        return super().write(data)


def _burn_cpu(stop) -> None:
    while not stop.is_set():
        sum(i * i for i in range(10000))


def _hold_gil(stop: threading.Event) -> None:
    while not stop.is_set():
        x = 0
        for i in range(200000):
            x += i


class Load:
    """with 동안 부하를 건다. 빠져나가면 부하 프로세스·스레드를 모두 멈춘다."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._stop_cpu, self._stop_gil = mp.Event(), threading.Event()
        self._procs: list[mp.Process] = []

    def __enter__(self) -> "Load":
        if self.kind in ("cpu", "both"):
            self._procs = [
                mp.Process(target=_burn_cpu, args=(self._stop_cpu,), daemon=True)
                for _ in range(os.cpu_count() or 4)
            ]
            for p in self._procs:
                p.start()
        if self.kind in ("gil", "both"):
            for _ in range(2):
                threading.Thread(target=_hold_gil, args=(self._stop_gil,), daemon=True).start()
        time.sleep(1.0)
        return self

    def __exit__(self, *exc) -> None:
        self._stop_cpu.set()
        self._stop_gil.set()
        for p in self._procs:
            p.join(5)
            p.kill()
        time.sleep(0.5)


def play_once(speaker: Speaker, pcm: bytes) -> tuple[float, float]:
    """pcm 을 한 번 내고 (play() 걸린 초, 소리가 빈 시간 최대 초)."""
    _writes.clear()
    asked = time.perf_counter()
    speaker.play([pcm], mute_mic=False)
    took = time.perf_counter() - asked
    first, queued, late = _writes[0][0], 0.0, 0.0
    for started, size in _writes:
        late = max(late, (started - first) - queued)
        queued += size / 2 / SAMPLE_RATE
    return took, late


def measure(n: int) -> None:
    chime = Chime()
    for kind in ("none", "cpu", "gil", "both"):
        with Load(kind):
            for name, pcm in (("시작음", chime.start_pcm), ("종료음", chime.end_pcm)):
                for label, piece in WAYS:
                    speaker = Speaker(latency="low", piece_bytes=piece)
                    res = []
                    for _ in range(n):
                        res.append(play_once(speaker, pcm))
                        time.sleep(0.2)
                    lates = [round(r[1] * 1000) for r in res]
                    tooks = [round(r[0] * 1000) for r in res]
                    head = f"부하 {kind:<4} {name} {label}"
                    print(f"{head}: 소리가 빈 시간 {lates}ms, play() {tooks}ms", flush=True)


def listen(n: int) -> None:
    chime = Chime()
    with Load("gil"):
        for i in range(n):
            for label, piece in WAYS:
                print(f"{i + 1}: {label}", flush=True)
                Speaker(latency="low", piece_bytes=piece).play([chime.start_pcm], mute_mic=False)
                time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="파이썬이 바쁠 때 시작음·종료음 끊김")
    parser.add_argument("--n", type=int, default=6, help="조합마다 몇 번 낼지")
    parser.add_argument("--listen", action="store_true", help="gil 부하에서 둘을 번갈아 낸다")
    args = parser.parse_args()
    sd.RawOutputStream = _TimedStream  # Speaker 가 여는 스트림의 write 시각을 남긴다
    if args.listen:
        listen(args.n)
    else:
        measure(args.n)


if __name__ == "__main__":
    main()
