"""TTS 합성이 읽는 도중 멈추는 비율을 모델마다 잰다. 소리는 내지 않는다.  담당: 문태현

gpt-4o-mini-tts 는 조각을 보내다가 몇 초~30초 넘게 멈출 때가 있다(#117 · #118). 우리 TTS 는
읽기 타임아웃 4초가 넘으면 폴백 모델로 처음부터 다시 요청한다(#130). 이 스크립트는

- 기본: 모델마다 같은 문장을 n 번 합성해 첫 조각 · 조각 사이 최대 멈춤 · 전체 시간을 잰다.
  읽기 타임아웃은 넉넉히(30초) 걸어서 얼마나 멈추는지 끝까지 본다.
- ``--speak``: 실제 ``TextToSpeech.speak()`` 를 스피커 없이 n 번 돌려, 멈춘 턴이 폴백으로
  끝까지 읽혔는지 본다(요청한 모델 순서, 읽은 소리 길이).

실행:
    python scripts/check_tts_stall.py                       # TTS_MODEL · TTS_FALLBACK_MODEL 8번씩
    python scripts/check_tts_stall.py --n 4 --models tts-1
    python scripts/check_tts_stall.py --speak --n 10
    python scripts/check_tts_stall.py --short                # 되묻기 길이의 짧은 문장
"""

import argparse
import logging
import statistics
import sys
import time
from types import SimpleNamespace

import httpx2
from openai import OpenAI, OpenAIError

from jarviseo import config
from jarviseo.voice.speaker import SAMPLE_RATE
from jarviseo.voice.tts import CHUNK_BYTES, READ_TIMEOUT_SEC, TextToSpeech
from jarviseo.voice.tts_text import normalize

# voice_loop.py --long 과 같은 길이의 대답(약 15초). 멈춤은 긴 대답에서 많이 났다(#118).
LONG_TEXT = (
    "지금부터 조금 긴 대답을 읽어 드릴게요. 중간에 끼어들 수 있는지 시험하려고 일부러 길게 만든 "
    "문장입니다. 읽는 도중에 호출어를 부르시면 바로 멈추고 새 질문을 들을게요. 부르지 않으시면 "
    "끝까지 읽은 다음 호출을 기다립니다. 마지막으로, 오늘 하루도 편안하게 보내세요."
)
SHORT_TEXT = "네, 10월 14일에 흰색 사물함 앞에서 본 검은색 카드 지갑 기억납니다."
# 기본 측정에서 이만큼 기다려도 다음 조각이 안 오면 실패로 친다
PROBE_READ_TIMEOUT_SEC = 30.0


def probe(client: OpenAI, model: str, text: str) -> dict:
    """합성 하나를 끝까지 받으며 첫 조각 · 조각 사이 최대 멈춤 · 전체 시간(ms)을 잰다."""
    asked = time.perf_counter()
    first, gap, size, last, error = None, 0.0, 0, asked, ""
    try:
        with client.audio.speech.with_streaming_response.create(
            model=model, voice="coral", input=normalize(text), response_format="pcm"
        ) as response:
            for chunk in response.iter_bytes(CHUNK_BYTES):
                now = time.perf_counter()
                if first is None:
                    first = now - asked
                else:
                    gap = max(gap, now - last)
                last, size = now, size + len(chunk)
    except (OpenAIError, httpx2.HTTPError) as e:
        error = type(e).__name__
        if first is not None:  # 받기 시작한 뒤 멈췄다
            gap = max(gap, time.perf_counter() - last)
    return {
        "first_ms": None if first is None else first * 1000,
        "gap_ms": gap * 1000,
        "total_ms": (time.perf_counter() - asked) * 1000,
        "audio_s": size / 2 / SAMPLE_RATE,
        "error": error,
    }


def run_probe(models: list[str], n: int, text: str) -> None:
    client = OpenAI(
        api_key=config.OPENAI_API_KEY or None,
        timeout=httpx2.Timeout(PROBE_READ_TIMEOUT_SEC),
        max_retries=0,
    )
    limit = READ_TIMEOUT_SEC * 1000
    for model in models:
        results = []
        for i in range(1, n + 1):
            r = probe(client, model, text)
            results.append(r)
            first = "    -" if r["first_ms"] is None else f"{r['first_ms']:5.0f}"
            mark = r["error"] or ("멈춤" if r["gap_ms"] > limit else "정상")
            print(
                f"{model:<16} {i:2}: 첫 조각 {first}ms, 최대 멈춤 {r['gap_ms']:6.0f}ms,"
                f" 전체 {r['total_ms']:6.0f}ms, 소리 {r['audio_s']:5.1f}s  {mark}",
                flush=True,
            )
        stalled = [r for r in results if r["error"] or r["gap_ms"] > limit]
        ok = [r for r in results if r not in stalled]
        line = f"→ {model}: 읽기 제한 {READ_TIMEOUT_SEC:.0f}초를 넘게 멈춤 {len(stalled)}/{n}"
        if ok:
            firsts = [r["first_ms"] for r in ok]
            gaps = [r["gap_ms"] for r in ok]
            line += (
                f", 정상일 때 첫 조각 중앙 {statistics.median(firsts):.0f}ms"
                f" · 최대 멈춤 {min(gaps):.0f}~{max(gaps):.0f}ms"
            )
        print(line + "\n", flush=True)


class _CountingTTS(TextToSpeech):
    """요청한 모델을 차례로 남기는 TextToSpeech."""

    def _stream(self, text, model=None):
        self.models.append(model or self.model)
        return super()._stream(text, model)


class _Silent:
    """sounddevice.RawOutputStream 대신 쓴 바이트 수만 센다."""

    written = 0

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def write(self, data):
        _Silent.written += len(data)

    def abort(self):
        pass


class _Failures(logging.Handler):
    """speak() 의 경고를 찍고, "합성 실패"(폴백 요청도 멈춘 턴)를 센다."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record):
        message = record.getMessage()
        print(f"    경고: {message}", flush=True)
        self.count += message.startswith("합성 실패")


def run_speak(n: int, text: str) -> None:
    sys.modules["sounddevice"] = SimpleNamespace(RawOutputStream=_Silent)  # 소리를 내지 않는다
    failures = _Failures()
    logging.getLogger("jarviseo.voice.tts").addHandler(failures)
    fell_back = failed = 0
    for i in range(1, n + 1):
        tts = _CountingTTS()
        tts.models = []
        _Silent.written, before = 0, failures.count
        started = time.perf_counter()
        took = tts.speak(text)
        total = (time.perf_counter() - started) * 1000
        broke = failures.count > before
        fell_back += len(tts.models) > 1
        failed += broke or took is None
        first = "    -" if took is None else f"{took:5.0f}"
        print(
            f"{i:2}: 요청 {' → '.join(tts.models)}, 첫 조각 {first}ms, 전체 {total:6.0f}ms,"
            f" 읽은 소리 {_Silent.written / 2 / SAMPLE_RATE:5.1f}s"
            f"  {'끝까지 못 읽음' if broke or took is None else '끝까지'}",
            flush=True,
        )
    print(f"→ 폴백으로 다시 요청 {fell_back}/{n}, 끝까지 못 읽음 {failed}/{n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS 합성 도중 멈춤 비율")
    parser.add_argument("--n", type=int, default=8, help="모델마다 몇 번 합성할지")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[config.TTS_MODEL, config.TTS_FALLBACK_MODEL],
        help="잴 모델들 (기본: TTS_MODEL, TTS_FALLBACK_MODEL)",
    )
    parser.add_argument("--short", action="store_true", help="짧은 대답 문장으로 잰다")
    parser.add_argument(
        "--speak", action="store_true", help="실제 speak() 로 폴백이 끝까지 읽는지 본다"
    )
    args = parser.parse_args()

    text = SHORT_TEXT if args.short else LONG_TEXT
    if args.speak:
        run_speak(args.n, text)
    else:
        run_probe(args.models, args.n, text)


if __name__ == "__main__":
    main()
