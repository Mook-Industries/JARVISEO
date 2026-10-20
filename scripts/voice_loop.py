"""말하면 받아쓴 질문을 그대로 목소리로 되돌려 준다.  담당: 문태현

카메라·VLM 없이 음성 입출력만 잇는 개발용 루프다(STT → 에코 → TTS).
턴마다 단계별 지연을 찍고, 끝낼 때 중앙값을 모아 찍는다. PR 에 지연을 적을 때 쓴다.
OpenAI 키가 필요하다.

실행:
    python scripts/voice_loop.py              # Enter 를 누르고 말한다. q 로 끝
    python scripts/voice_loop.py 1 4          # 1번 마이크, 4번 스피커
    python scripts/voice_loop.py --wake       # 호출어를 부르고 이어서 말한다. Ctrl+C 로 끝
    python scripts/voice_loop.py --wake --long --db   # 끼어들기 측정, 턴을 DB 에 남긴다

--wake 는 config.WAKEWORD_MODEL(기본 data/models/jarviseo.onnx)로 호출어를 기다린다.
"자비서, 저거 뭐야?" 처럼 부른다. 호출어만 부르고 멈추면 "네, 말씀하세요."라고 되묻고 다시 듣는다.
대답을 읽는 중에 "자비서"라고 부르면 바로 끊고 새 질문을 받는다(끼어들기).

--long 은 질문 뒤에 15초쯤 걸리는 문장을 붙여 읽는다. 끼어들 틈을 만들려는 것이다.
--short 는 질문 대신 되묻기 길이(약 2초)의 "왼쪽 컵이요, 오른쪽 컵이요?"를 읽는다.
읽을 때마다 "자비서"라고 불러서, 짧은 문장을 읽는 중에도 끼어들기가 되는지 잰다(#105).
읽는 중에 끊으면 몇 초째에 끊겼는지, 끝까지 읽었는데 감지가 재생 뒤로 밀렸으면 끝나고 몇 ms 에
잡혔는지, 아예 못 잡았으면 재생 중 최고 점수를 찍고 끝낼 때 모아 찍는다.
늦게 잡힌 호출도 assistant.py 의 되묻기처럼 다음 턴으로 받는다.
    python scripts/voice_loop.py --wake --short --chime
--db 는 턴마다 session_turn·turn_voice 에 남긴다. 주소를 안 주면 data/voice_loop.db(SQLite).
--playback-threshold 는 재생 중 호출어 임계값을 바꿔 본다(기본 config.WAKEWORD_PLAYBACK_THRESHOLD).
--wake 는 끝까지 읽은 대답마다 재생 중 호출어 최고 점수를 찍는다. 자기 목소리 점수와
못 잡은 끼어들기가 임계값에서 얼마나 떨어져 있는지 보려는 것이다.
--chime 은 --wake 에서 호출을 받으면 시작음, 질문하는 말이 끝나면 종료음을 낸다(assistant.py 와
같다). 빼면 둘 다 안 낸다. 켜고 끈 채 같은 질문을 해서 받아쓰기가 나빠지지 않는지 본다(#104).
    python scripts/voice_loop.py --wake --chime     # 켬
    python scripts/voice_loop.py --wake             # 끔

찍는 값 (ms)
    말끝 대기    말이 끝나고 VAD 가 끝으로 판정해 녹음을 멈추기까지 (무음 700ms 가 대부분)
    STT 왕복     받아쓰기 요청을 보내고 글자를 받기까지
    TTS 첫 조각  합성 요청을 보내고 첫 PCM 조각을 받기까지 (스피커 출력 지연 약 0.2초는 빠짐)
    재생         첫 조각을 받고 마지막 소리가 다 나오기까지 (끊긴 턴은 끊길 때까지)
    말 끝 → 첫 조각  사용자가 말을 멈추고 대답이 나오기 시작하기까지. 위 셋을 더한 값
    감지 → 멈춤  끼어들기에서 호출어가 끝난 시각부터 스피커를 멈추기까지
    시작음       호출을 받고 시작음 첫 조각을 스피커에 쓰기까지, 괄호는 호출어가 끝난 시각부터
    종료음       사용자가 말을 멈추고 종료음 첫 조각을 스피커에 쓰기까지 (말끝 대기 0.7초가 대부분)
                 (둘 다 스피커 출력 지연은 빠짐. 노트북 스피커 MME 에서 약 0.09초)
"""

import argparse
import statistics
import time
from collections.abc import Callable

import numpy as np

from jarviseo import config
from jarviseo.memory import MemoryStore
from jarviseo.types import AssistantResponse, Intent, Utterance
from jarviseo.voice import BargeIn, Chime, SpeechToText, TextToSpeech, WakeWordDetector
from jarviseo.voice.gate import MIC_GATE
from jarviseo.voice.mic import MicStream
from jarviseo.voice.wakeword import load_scorer

STAGES = ("말끝 대기", "STT 왕복", "TTS 첫 조각", "재생", "말 끝 → 첫 조각")
# 호출어("자비서")는 일부러 넣지 않는다. 자기 목소리에 깨는지 볼 때 답에 호출어가 있으면 안 된다.
LONG_ANSWER = (
    " 지금부터 조금 긴 대답을 읽어 드릴게요. 중간에 끼어들 수 있는지 시험하려고 일부러 길게 만든 "
    "문장입니다. 읽는 도중에 호출어를 부르시면 바로 멈추고 새 질문을 들을게요. 부르지 않으시면 "
    "끝까지 읽은 다음 호출을 기다립니다. 마지막으로, 오늘 하루도 편안하게 보내세요."
)
# 되묻기와 같은 길이의 문장(--short). 되묻는 말을 읽는 중에도 끼어들 수 있는지 본다(#105).
SHORT_ANSWER = "왼쪽 컵이요, 오른쪽 컵이요?"
# 짧은 문장을 끝까지 읽은 뒤 늦은 감지를 기다리는 시간(초). 감지는 호출어 끝보다 최대 0.33초 늦다.
LATE_SEC = 0.5
DEFAULT_DB = "sqlite:///data/voice_loop.db"

# 턴 하나를 DB 에 남기는 함수. (질문, 대답, 단계별 지연, trigger_type)
Record = Callable[[Utterance, str, dict[str, float], str], None]


def echo(
    stt: SpeechToText, tts: TextToSpeech, utterance: Utterance, long: bool, short: bool = False
) -> tuple[str, dict[str, float]] | None:
    """질문을 그대로(--long 이면 긴 문장을 붙여, --short 면 짧은 되묻기 문장으로 바꿔) 읽어 주고
    대답과 단계별 지연(ms)을 돌려준다.

    소리를 못 냈으면 None.
    """
    print(f"  질문 {utterance.text!r} (원문 {stt.last_raw_text!r})")
    answer = SHORT_ANSWER if short else utterance.text + (LONG_ANSWER if long else "")
    heard = time.monotonic()
    if tts.speak(answer) is None:
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
    cut = " (끊김)" if tts.last_interrupted else ""
    print("  " + ", ".join(f"{name} {turn[name]:.0f}" for name in STAGES) + cut)
    return answer, turn


class PlaybackPeak:
    """호출어 점수를 그대로 넘기면서, 재생하는 동안 나온 점수를 모은다.

    peak 는 재생 중 최고 점수, missed 는 임계값을 못 넘은 높은 점수(못 잡은 끼어들기로 본다)다.
    """

    MISS = 0.3  # 이 점수를 넘었는데 임계값은 못 넘으면 못 잡은 끼어들기로 본다
    GAP = 12  # 프레임(80ms) 수. 이만큼 안에 이어진 높은 점수는 한 번 부른 것으로 친다

    def __init__(self, threshold: float) -> None:
        self.scorer = load_scorer(config.WAKEWORD_MODEL)
        self.threshold = threshold
        self.playing, self.peak = False, 0.0
        self._calls: list[float] = []  # 재생 중 높은 점수가 이어진 구간마다 최고 점수
        self._frame = self._last_high = 0
        MIC_GATE.on_playback(self._playback)

    @property
    def missed(self) -> list[float]:
        return [score for score in self._calls if score < self.threshold]

    def _playback(self, playing: bool) -> None:
        if playing:
            self.peak, self._calls = 0.0, []
        self.playing = playing

    def __call__(self, frame: np.ndarray) -> float:
        score = self.scorer(frame)
        self._frame += 1
        if self.playing:
            self.peak = max(self.peak, score)
            if score >= self.MISS:
                if not self._calls or self._frame - self._last_high > self.GAP:
                    self._calls.append(score)
                else:
                    self._calls[-1] = max(self._calls[-1], score)
                self._last_high = self._frame
        return score


def enter_loop(
    stt: SpeechToText, tts: TextToSpeech, turns: list, long: bool, record: Record | None
) -> None:
    while input("\nEnter 를 누르고 말하세요 (끝내려면 q): ").strip().lower() != "q":
        utterance = stt.listen(timeout=8.0)
        if utterance is None:
            print(f"  질문 없음 (원문 {stt.last_raw_text!r})")
        elif result := echo(stt, tts, utterance, long):
            turns.append(result[1])
            if record is not None:
                record(utterance, *result, "WAKEWORD")


def wake_loop(
    stt: SpeechToText,
    tts: TextToSpeech,
    mic: MicStream,
    turns: list,
    long: bool,
    record: Record | None,
    playback_threshold: float,
    chime: Chime | None,
    short: bool = False,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[tuple[float, float]],
    list[float],
    int,
    dict[str, list[float]],
]:
    """호출어로 턴을 연다. 끼어들기마다 감지 → 멈춤(ms)을, 끝까지 읽은 대답마다 재생 중
    호출어 최고 점수를, 재생 중 임계값을 못 넘은 높은 점수를, 시작음마다 (호출 받음 → 시작음,
    호출어 끝 → 시작음)(ms)을, 질문마다 말 끝 → 종료음(ms)을 모으고, 되물은 횟수와 같이 돌려준다.
    short 면 짧은 문장마다 읽는 중 끊긴 때(cut, 재생 ms)·끝난 뒤 늦게 잡힌 때(late, 끝 기준 ms)·
    못 잡았을 때 재생 중 최고 점수(missed)도 모아 마지막에 돌려준다.

    종료음은 stt 를 만들 때 ``on_speech_end=chime.end`` 로 넘겨 둔다(main).
    """
    barge = BargeIn(tts)
    detected: list[float] = []  # 호출어를 잡은 시각. --short 에서 재생 뒤 늦은 감지를 찾는다

    def on_detect(t: float, score: float) -> None:
        detected.append(t)
        barge.on_detect(t, score)

    peak = PlaybackPeak(playback_threshold)
    detector = WakeWordDetector(
        mic, on_detect=on_detect, playback_threshold=playback_threshold, scorer=peak
    )
    detector.start()
    stops: list[float] = []
    peaks: list[float] = []
    missed: list[float] = []
    starts: list[tuple[float, float]] = []
    ends: list[float] = []
    reprompts = 0
    shorts: dict[str, list[float]] = {"cut": [], "late": [], "missed": []}
    trigger = "WAKEWORD"
    print(
        f"\n호출어({config.WAKEWORD_MODEL}, 임계값 {detector.threshold}, 재생 중 "
        f"{detector.playback_threshold})를 부르고 이어서 말하세요. 끝내려면 Ctrl+C"
    )
    print(f"시작음·종료음 {'켜짐' if chime else '꺼짐 (--chime)'}")
    try:
        while True:
            wake = barge.wait(timeout=0.5)
            if wake is None:
                continue
            called = time.monotonic()
            if chime:
                chime.start()
            if trigger == "BARGE_IN" and tts.speaker.stopped_at is None:
                # --short 에서 재생이 끝난 뒤에 잡힌 호출이다. 멈춘 소리가 없다
                print(f"\n끼어들기 (점수 {wake.score:.3f}, 재생 뒤 감지)")
            elif trigger == "BARGE_IN":
                stops.append((tts.speaker.stopped_at - wake.t) * 1000)
                print(f"\n끼어들기 (점수 {wake.score:.3f}, 감지 → 멈춤 {stops[-1]:.0f}ms)")
            else:
                print(f"\n호출어 감지 (점수 {wake.score:.3f})")
            utterance = stt.listen(timeout=2.0, since=wake.t)
            if utterance is None:
                # 호출어만 부르고 멈췄다. 되묻고 다시 듣는다(재생 뒤 0.3초는 게이트가 막는다)
                reprompts += 1
                print(f"  되물음 (원문 {stt.last_raw_text!r})")
                tts.speak("네, 말씀하세요.")
                utterance = stt.listen(timeout=5.0)
            if chime:
                chime.wait()  # 받아쓰는 동안 둘 다 나서 보통 이미 끝났다
            if chime and chime.start_played_at is not None:
                started = chime.start_played_at
                starts.append(((started - called) * 1000, (started - wake.t) * 1000))
                line = f"  시작음 {starts[-1][0]:.0f}ms (호출어 끝부터 {starts[-1][1]:.0f}ms)"
                if utterance is not None and chime.end_played_at is not None:
                    ends.append((chime.end_played_at - utterance.ended_at) * 1000)
                    line += f", 말 끝 → 종료음 {ends[-1]:.0f}ms"
                print(line)
            result = late = None
            before = time.monotonic()
            if utterance is None:
                print(f"  질문 없음 (원문 {stt.last_raw_text!r})")
            elif result := echo(stt, tts, utterance, long, short):
                ended = time.monotonic()
                turns.append(result[1])
                if record is not None:
                    record(utterance, *result, trigger)
                if not tts.last_interrupted:
                    peaks.append(peak.peak)
                    print(f"  재생 중 호출어 최고 점수 {peak.peak:.3f}")
                if peak.missed:
                    missed += peak.missed
                    print(
                        "  재생 중 임계값 못 넘은 점수 "
                        + ", ".join(f"{s:.3f}" for s in peak.missed)
                    )
                if short and tts.last_interrupted:
                    shorts["cut"].append(result[1]["재생"])
                    print(f"  {result[1]['재생'] / 1000:.1f}초째에 끊김")
                elif short:
                    # 짧은 문장 끝에 걸쳐 부르면 감지가 재생 뒤로 밀린다. 감지 지연만큼 더 본다
                    time.sleep(LATE_SEC)
                    late = next((t for t in detected if t > before), None)
                    if late is not None:
                        shorts["late"].append((late - ended) * 1000)
                        print(f"  끝까지 읽고 {shorts['late'][-1]:+.0f}ms 에 감지 (늦은 감지)")
                    else:
                        shorts["missed"].append(peak.peak)
                        print(f"  못 잡음 (재생 중 최고 점수 {peak.peak:.3f})")
            # 대답을 읽다 끊겼거나 (--short 에서) 끝난 직후 늦게 잡혔으면 그 호출이 큐에 있고,
            # 그게 다음 턴이 된다. 끝까지 읽었으면 질문하는 동안 다시 잡힌 호출어 같은 지난 감지는
            # 버린다.
            called_again = result is not None and (tts.last_interrupted or late is not None)
            trigger = "BARGE_IN" if called_again else "WAKEWORD"
            if trigger == "WAKEWORD":
                barge.clear()
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop()
    return stops, peaks, missed, starts, ends, reprompts, shorts


def recorder(url: str, tts: TextToSpeech, stt: SpeechToText) -> Record:
    """턴을 DB 에 남기는 함수를 만든다. 세션은 실행 한 번에 하나다."""
    store = MemoryStore(database_url=url)
    store.init_schema()
    session_id = store.start_session(store.ensure_user(), title="voice_loop")
    print(f"턴 기록: {url} (session_id {session_id})")

    def record(utterance: Utterance, answer: str, turn: dict[str, float], trigger: str) -> None:
        response = AssistantResponse(
            text=answer,
            intent=Intent.GENERAL,
            latency_ms={
                "stt": turn["STT 왕복"],
                "tts": turn["TTS 첫 조각"],
                "total": turn["말 끝 → 첫 조각"],
            },
        )
        turn_id = store.log_turn(
            session_id,
            utterance,
            response,
            trigger_type=trigger,
            stt_raw_text=stt.last_raw_text,
            interrupted=tts.last_interrupted,
        )
        print(f"  기록 turn_id {turn_id} ({trigger}{', 끊김' if tts.last_interrupted else ''})")

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="음성 루프 (STT → 에코 → TTS)")
    parser.add_argument("mic", type=int, nargs="?", help="sounddevice 입력 장치 번호")
    parser.add_argument("speaker", type=int, nargs="?", help="sounddevice 출력 장치 번호")
    parser.add_argument("--wake", action="store_true", help="Enter 대신 호출어로 시작한다")
    parser.add_argument("--long", action="store_true", help="질문 뒤에 긴 문장을 붙여 읽는다")
    parser.add_argument(
        "--short",
        action="store_true",
        help="질문 대신 되묻기 길이(약 2초) 문장을 읽는다. 읽는 중 끼어들기를 잰다 (#105)",
    )
    parser.add_argument(
        "--db", nargs="?", const=DEFAULT_DB, help=f"턴을 DB 에 남긴다 (기본 {DEFAULT_DB})"
    )
    parser.add_argument(
        "--playback-threshold",
        type=float,
        default=config.WAKEWORD_PLAYBACK_THRESHOLD,
        help="재생 중 호출어 임계값",
    )
    parser.add_argument(
        "--chime", action="store_true", help="--wake 에서 시작음·종료음을 낸다 (#104)"
    )
    args = parser.parse_args()

    turns: list[dict[str, float]] = []
    stops: list[float] = []
    peaks: list[float] = []
    missed: list[float] = []
    starts: list[tuple[float, float]] = []
    ends: list[float] = []
    reprompts = 0
    shorts: dict[str, list[float]] = {}
    tts = TextToSpeech(device=args.speaker)
    if args.wake:
        chime = Chime(device=args.speaker) if args.chime else None
        with MicStream(args.mic) as mic:
            stt = SpeechToText(mic=mic, on_speech_end=chime.end if chime else None)
            record = recorder(args.db, tts, stt) if args.db else None
            stops, peaks, missed, starts, ends, reprompts, shorts = wake_loop(
                stt, tts, mic, turns, args.long, record, args.playback_threshold, chime, args.short
            )
    else:
        stt = SpeechToText(device=args.mic)
        record = recorder(args.db, tts, stt) if args.db else None
        enter_loop(stt, tts, turns, args.long, record)

    if turns:
        print(f"\n중앙값 ({len(turns)}턴, ms)")
        for name in STAGES:
            print(f"  {name} {statistics.median(t[name] for t in turns):.0f}")
    if stops:
        print(f"끼어들기 {len(stops)}회, 감지 → 멈춤 중앙값 {statistics.median(stops):.0f}ms")
    if peaks:
        print(f"끝까지 읽은 대답 {len(peaks)}개, 재생 중 호출어 최고 점수 {max(peaks):.3f}")
    if missed:
        print(
            f"재생 중 임계값 못 넘은 높은 점수 {len(missed)}번: "
            + ", ".join(f"{s:.3f}" for s in missed)
        )
    if starts:
        print(
            f"시작음 {len(starts)}번, 호출 받음 → 시작음 중앙값 "
            f"{statistics.median(c[0] for c in starts):.0f}ms "
            f"(호출어 끝부터 {statistics.median(c[1] for c in starts):.0f}ms)"
        )
    if ends:
        print(f"종료음 {len(ends)}번, 말 끝 → 종료음 중앙값 {statistics.median(ends):.0f}ms")
    if args.wake:
        print(f"되물음 {reprompts}번")
    if args.wake and args.short:
        cut, late, miss = shorts["cut"], shorts["late"], shorts["missed"]
        line = f"짧은 문장 {len(cut) + len(late) + len(miss)}번: 읽는 중 끊김 {len(cut)}"
        if cut:
            line += f"(중앙 {statistics.median(cut) / 1000:.1f}초째)"
        line += f", 끝난 뒤 늦게 감지 {len(late)}"
        if late:
            line += "(" + ", ".join(f"{ms:+.0f}" for ms in late) + "ms)"
        line += f", 못 잡음 {len(miss)}"
        if miss:
            line += "(재생 중 최고 점수 " + ", ".join(f"{s:.3f}" for s in miss) + ")"
        print(line)


if __name__ == "__main__":
    main()
