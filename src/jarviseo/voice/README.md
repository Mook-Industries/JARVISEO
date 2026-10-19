# voice — 음성 입출력 · 담당 문태현

흐름: 호출어 감지 → VAD 로 발화 구간 파악 → STT → 호출어·필러 정리 → (처리) → TTS

| 파일 | 하는 일 | 어디서 도나 | 책임 지표 |
|---|---|---|---|
| `wakeword.py` | 호출어 감지. 80ms 마다 openWakeWord 점수를 내고 임계값을 넘으면 `on_detect(t, score)` | 로컬 (상시) | FAR / FRR |
| `stt.py` | 호출어 뒤 발화 받아쓰기 (`listen()` 이 녹음·VAD·받아쓰기·정리를 묶는다) | OpenAI STT | — |
| `transcript.py` | 받아쓴 원문에서 호출어·필러를 빼고, 비었거나 너무 짧은 발화를 거른다 | 로컬 | — |
| `tts.py` | 응답 음성 합성과 재생 (`speak()` 이 `synthesize()` 의 PCM 조각을 받는 대로 재생한다) | OpenAI TTS | — |
| `tts_text.py` | TTS 에 넣기 전에 단위·쉼표·물결표를 읽는 말로 바꾸고 마크다운 기호를 뺀다. 문장 분할 함수도 있다 | 로컬 | — |
| `speaker.py` | PCM 조각을 받는 순서대로 스피커로 낸다. 재생하는 동안 마이크 게이트를 닫는다. `stop()` 으로 바로 끊는다 | 로컬 | — |
| `bargein.py` | 재생 중 끼어들기. 호출어가 들리면 읽던 답을 끊고 그 호출을 다음 턴으로 넘긴다 | 로컬 | — |
| `chime.py` | 호출을 받으면 시작음, 턴이 끝나면 종료음(`sounds/*.wav`). 시작음은 마이크 게이트를 닫지 않는다 | 로컬 | — |
| `gate.py` | 재생 중과 끝난 뒤 0.3초 동안 STT 쪽 마이크를 막고, 호출어 쪽에는 그 구간을 알려 임계값을 올리게 한다 | 로컬 | — |
| `mic.py` | 마이크를 30ms 블록으로 읽고 블록마다 들어온 시각을 붙인다. `MicStream` 은 마이크 하나를 여럿이 같이 읽게 한다 | 로컬 | — |
| `vad.py` | webrtcvad 로 발화 하나를 잘라 낸다 (무음 700ms 면 끝) | 로컬 | — |
| `fake.py` | 키 없이 쓰는 가짜 STT · TTS (그래프 개발 · CI 용) | 로컬 | — |

그래프에서는 `from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech` 로 먼저 개발하고,
실제 구현이 들어오면 `SpeechToText`, `TextToSpeech` 로 바꿔 끼운다. 메서드 모양은 같다.

`listen()` 은 정리한 질문을 돌려주고, 원문은 `last_raw_text`, 받아쓰기 왕복 시간(ms)은
`last_latency_ms` 에 남긴다. 그래프는 원문을 `log_turn(..., stt_raw_text=...)` 로,
왕복 시간을 응답의 `latency_ms["stt"]` 로 넘긴다.

TTS 는 `user_setting.tts_speed` 를 OpenAI speed 값으로 바꿔 보낸다(SLOW 0.85, FAST 1.2, NORMAL 은 안 보냄).
첫 조각을 받기까지 걸린 시간은 `last_latency_ms` 에 남고, 그래프가 `latency_ms["tts"]` 로 넘기면
`turn_voice.tts_ms` 에 적힌다. `speak()` 은 재생이 끝날 때까지 기다리고 같은 값을 돌려준다.
합성이 도중에 끊기면 받은 만큼만 재생하고, 소리를 하나도 못 냈으면 `None` 을 돌려준다.

모델 이름은 `config.py` 에서만 정한다. 지금은 STT `gpt-transcribe`, TTS `gpt-4o-mini-tts`
(`.env` 의 `JARVISEO_STT_MODEL`, `JARVISEO_TTS_MODEL`).

호출어는 `MicStream` 하나를 열어 `WakeWordDetector` 와 `SpeechToText(mic=...)` 가 같이 읽는다.

```python
with MicStream() as mic:
    stt = SpeechToText(mic=mic)
    detector = WakeWordDetector(mic, on_detect=lambda t, score: woke.put(t))
    detector.start()
    t = woke.get()  # 감지 스레드에서 받은 감지 시각
    utterance = stt.listen(since=t)  # 호출어에 이어 말한 질문부터 받아쓴다
```

`MicStream` 은 최근 3초를 들고 있어서, 감지하고 `listen()` 을 부르기까지 지나간 말도 놓치지 않는다.

읽어 주는 중에 "자비서"라고 부르면 끊고 새 질문을 받으려면(끼어들기) 감지를 `BargeIn` 으로 받는다.

```python
barge = BargeIn(tts)
detector = WakeWordDetector(mic, on_detect=barge.on_detect)  # 들리면 tts.stop()
...
wake = barge.wait(timeout=0.5)  # 다음 호출 (Wake(t, score))
utterance = stt.listen(since=wake.t)
tts.speak(answer)  # 도중에 호출어가 들리면 바로 돌아온다
interrupted = tts.last_interrupted  # True 면 이 턴은 is_interrupted, 다음 턴은 BARGE_IN
if not interrupted:
    barge.clear()  # 질문하는 동안 다시 잡힌 호출어 같은 지난 감지는 버린다
```

끊기면 남은 소리는 버리고(20ms 안에 멈춤 신호를 본다), 덜 받은 합성 스트림은 닫는다.
`wait()` 이 재생 중에 들린 호출을 넘길 때는 게이트를 질문을 읽기 시작하는 곳까지 거슬러 열어서,
재생 중에 말한 "자비서"와 바로 이어 말한 질문이 무음으로 바뀌지 않는다.

`listen(since=t)` 는 감지 시각보다 `JARVISEO_WAKE_LOOKBACK_SEC`(1초) 앞부터 읽는다. 감지는 호출어가
끝나고 0.1~0.3초 뒤에 돼서, 감지 시각부터 읽으면 바로 이어 말한 질문 앞부분이 빠진다.
호출어도 같이 받아써지고 `clean()` 이 호출어와 그 앞을 지운다. "자비서." 하고 쉬었다가 질문하면
녹음이 호출어에서 끝나는데(감지 0.4초 안), 그때는 받아쓰지 않고 이어서 한 번 더 듣는다.
말 시작 시각은 감지 0.35초 전보다 앞으로 두지 않는다. 측정 스크립트는 `scripts/wakeword/lag.py`·`lookback.py`.
모델·임계값은 `.env` 의 `JARVISEO_WAKEWORD_MODEL`(기본 `data/models/jarviseo.onnx`), `JARVISEO_WAKEWORD_THRESHOLD`(0.97),
`JARVISEO_WAKEWORD_PLAYBACK_THRESHOLD`(0.97)로 바꾼다. 기본 모델은 "자비서"로 학습한 커스텀 모델이고
Git 에 없어서 따로 받아야 한다(`data/README.md`). 평가 결과와 한계는 `docs/experiments.md` 학습 기록 6회차에 있다.

## 알아둘 것

- **상시 동작하는 것은 호출어 감지뿐이고, 로컬에서 돈다.** 클라우드로 나가는 것은 호출어 뒤의 발화 한 번이다.
- `stt.py` 의 `started_at`(말을 시작한 시각)이 핵심이다. 이 시각으로 `capture` 가 프레임을 고른다.
  잡히는지는 `scripts/spike/spike_02_stt.py` 로 확인한다.
- **되먹임 루프 주의.** 스피커 소리를 마이크가 다시 듣는다. 그래서 `gate.py` 의 `MIC_GATE` 가
  재생하는 동안과 끝난 뒤 `TTS_MIC_GATE_SEC`(0.3초) 동안 들어온 소리를 무음으로 바꿔 STT 에 넘긴다.
  STT·TTS 가 같은 게이트를 자동으로 쓰므로 그래프에서 따로 연결할 것은 없다.
- 호출어는 재생 중에도 막지 않는다(소프트 게이트, 2단). 같은 구간에 들어온 소리에는 임계값을 따로
  쓸 수 있다. 자기 목소리(TTS 대답) 점수가 실제 재생에서 0.001 이하라 기본은 평소와 같은 0.97 이다.
  0.99 로 올리면 재생 중에 붙여 말한 "자비서오늘…"을 자주 놓쳤다.
  재생 시작·끝을 알리는 `MIC_GATE.on_playback(hook)` 은 재생음과 비교해 자기 목소리를 거르는 단계가
  필요해지면 쓴다.
- **시작음은 게이트 밖이다.** `Chime.start()` 는 사용자가 "자비서, 이거 뭐야"처럼 이어 말하는 도중에
  울려서, 게이트를 닫으면 그동안 말한 질문이 무음으로 바뀐다. 그래서 시작음은 마이크에 같이 들어간다.
  대신 소리가 마이크에 들어가는 구간을 `MIC_GATE.sounding()` 으로 표시하고, `listen(since)` 의 VAD 는 그 구간을
  말로 보지 않는다. 안 그러면 "자비서." 하고 쉬었다가 질문할 때 시작음이 쉼을 메워 호출어와 질문이 한 발화로 붙는다.
  받아쓰기에 끼치는 영향은 `scripts/wakeword/lookback.py --chime` 으로 잰다(docs/experiments.md #104).
  종료음(`Chime.end()`)은 TTS 처럼 게이트를 닫는다. 둘 다 0.4초 안쪽이고 `JARVISEO_CHIME_VOLUME` 으로
  크기를 바꾸거나(0 이면 끔) `sounds/start.wav`·`end.wav`(24kHz 16비트 모노)를 바꿔 끼운다.

```python
chime = Chime()
wake = barge.wait(timeout=0.5)
chime.start()  # 바로 돌아온다. 이어서 listen() 을 불러야 질문을 안 놓친다
utterance = stt.listen(since=wake.t)
tts.speak(answer)
if not tts.last_interrupted:
    barge.clear()
    chime.end()  # 끊긴 턴은 바로 다음 턴의 시작음이 나서 내지 않는다
```
