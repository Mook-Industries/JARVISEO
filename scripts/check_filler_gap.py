"""필러("잠깐만요") 뒤 대답 첫 소리까지의 빈틈을 미리 합성(prefetch) 끔/켬으로 잰다.  담당: 문태현

VLM 은 정해 둔 시간만큼 기다렸다 답하는 가짜이고, TTS 는 실제 API 로 합성한다(키 필요).
스피커는 소리를 내지 않고 조각 길이만큼 기다리기만 한다. 재생 시간은 실제와 같다.

VLM 지연과 대답 문장은 assistant.py 가 남긴 DB 의 턴을 그대로 쓴다(기본 data/assistant.db,
D12 실측 21턴). DB 가 없으면 아래 SAMPLE 5턴을 쓴다. 턴마다 끔·켬을 번갈아 돌려서
그사이 API 속도가 바뀌어도 두 쪽에 고르게 섞이게 한다.

잰 것 (ms)
- 빈틈 : "잠깐만요" 재생이 끝난 뒤 대답 첫 조각이 스피커에 들어가기까지. 필러가 안 나간 턴은 뺀다.
- tts  : speak() 를 부른 뒤 첫 조각까지 (turn_voice.tts_ms)
- 대답 : 질문을 VLM 에 보낸 뒤 대답 첫 소리까지

실행:
    python scripts/check_filler_gap.py
    python scripts/check_filler_gap.py --limit 5
    python scripts/check_filler_gap.py --db data/voice_loop.db
"""

import argparse
import itertools
import sqlite3
import statistics
import time
from pathlib import Path

from jarviseo.voice import ResponseCollector, TextToSpeech
from jarviseo.voice.tts import SAMPLE_RATE

# (VLM ms, 대답) — DB 가 없을 때. D12 실측에서 VLM 이 빠른 턴부터 느린 턴까지 고루 골랐다.
SAMPLE = [
    (1734, "손에 들고 계신 건 갈아만든 배 음료 캔입니다."),
    (1990, "바나프레소 로고가 적힌 플라스틱 음료 컵입니다."),
    (2274, "아까 음료 컵은 오른쪽 앞에 있었고, 뒤로는 흰색 사물함이 보였어요."),
    (2608, "보라색 통에 담긴 감자칩으로 보입니다."),
    (3235, "기록과 지금 보이는 장면에는 이어폰이 없어서 어디 있었는지 기억에 없습니다."),
]


class NoPrefetch(TextToSpeech):
    """미리 합성을 끈 TTS. 고치기 전과 같다."""

    def prefetch(self, text: str) -> None:
        pass


class SilentSpeaker:
    """소리 없이 조각 길이만큼 기다린다. 재생마다 (첫 조각 시각, 끝 시각) 을 남긴다."""

    def __init__(self) -> None:
        self.plays: list[tuple[float, float]] = []
        self.interrupted = False

    def play(self, chunks) -> None:
        chunks = iter(chunks)
        first = next(chunks, None)
        if first is None:
            return
        began = time.monotonic()
        for chunk in itertools.chain([first], chunks):
            time.sleep(len(chunk) / (SAMPLE_RATE * 2))
        self.plays.append((began, time.monotonic()))

    def stop(self) -> None:
        pass


def load_turns(db: Path) -> list[tuple[float, str]]:
    if not db.exists():
        print(f"{db} 가 없어서 SAMPLE {len(SAMPLE)}턴을 쓴다")
        return SAMPLE
    with sqlite3.connect(db) as con:
        rows = con.execute(
            "SELECT ti.llm_ms, st.answer_text FROM session_turn st"
            " JOIN turn_inference ti ON ti.turn_id = st.turn_id"
            " WHERE ti.llm_ms IS NOT NULL AND st.answer_text != '' ORDER BY st.turn_id"
        ).fetchall()
    print(f"{db} 의 {len(rows)}턴")
    return rows


def one_turn(collector: ResponseCollector, vlm_ms: float, answer: str) -> dict:
    """질문 한 번: VLM 을 기다리며 필러를 읽고, 대답을 읽는다. 그래프의 generate·speak 와 같다."""
    tts = collector.tts
    tts.speaker.plays.clear()

    def ask() -> str:
        time.sleep(vlm_ms / 1000)
        return answer

    asked = time.monotonic()
    tts.speak(collector.collect(ask))
    first, _ = tts.speaker.plays[-1]
    result = {"tts": tts.last_latency_ms, "대답": (first - asked) * 1000, "필러": False}
    if collector.filler_sent:
        _, filler_end = tts.speaker.plays[0]
        result.update(필러=True, 빈틈=(first - filler_end) * 1000)
    return result


def summary(name: str, rows: list[dict]) -> str:
    def med(key: str) -> str:
        values = [r[key] for r in rows if r.get(key) is not None]
        return f"{statistics.median(values):6.0f}" if values else "     -"

    filler = sum(r["필러"] for r in rows)
    medians = f"빈틈 {med('빈틈')}  tts {med('tts')}  대답 {med('대답')}"
    return f"{name} 중앙값 {medians}  (필러 {filler}/{len(rows)}턴)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=Path("data/assistant.db"))
    parser.add_argument("--limit", type=int, default=0, help="앞에서 이만큼만 (0 이면 전부)")
    args = parser.parse_args()

    turns = load_turns(args.db)
    if args.limit:
        turns = turns[: args.limit]
    collectors = {}
    for name, tts in (("끔", NoPrefetch()), ("켬", TextToSpeech())):
        tts.speaker = SilentSpeaker()
        collectors[name] = ResponseCollector(tts)  # "잠깐만요"는 여기서 한 번 합성한다
    filler_sec = len(collectors["켬"]._filler) / (SAMPLE_RATE * 2)
    print(f'"잠깐만요." {filler_sec:.2f}초\n')

    results: dict[str, list[dict]] = {name: [] for name in collectors}
    print(f"{'VLM':>5} | {'끔: 빈틈':>8} {'tts':>5} | {'켬: 빈틈':>8} {'tts':>5}")
    for vlm_ms, answer in turns:
        line = f"{vlm_ms:5.0f} |"
        for name, collector in collectors.items():
            r = one_turn(collector, vlm_ms, answer)
            results[name].append(r)
            gap = f"{r['빈틈']:8.0f}" if r["필러"] else f"{'-':>8}"
            line += f" {gap} {r['tts']:5.0f} |"
        print(line)
    print()
    for name, rows in results.items():
        print(summary(name, rows))
    for collector in collectors.values():
        collector.close()


if __name__ == "__main__":
    main()
