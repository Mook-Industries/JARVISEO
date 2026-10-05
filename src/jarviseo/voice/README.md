# voice — 음성 입출력 · 담당 문태현

흐름: 호출어 감지 → VAD 로 발화 구간 파악 → STT → 호출어·필러 정리 → (처리) → TTS

| 파일 | 하는 일 | 어디서 도나 | 책임 지표 |
|---|---|---|---|
| `wakeword.py` | "자비서" 호출어 감지 | 로컬 (상시) | FAR / FRR |
| `stt.py` | 호출어 뒤 발화 받아쓰기 (`listen()` 이 녹음·VAD·받아쓰기·정리를 묶는다) | OpenAI STT | — |
| `transcript.py` | 받아쓴 원문에서 호출어·필러를 빼고, 비었거나 너무 짧은 발화를 거른다 | 로컬 | — |
| `tts.py` | 응답 음성 합성 (`synthesize()` 가 PCM 조각을 받는 대로 낸다. 재생은 다음 이슈) | OpenAI TTS | — |
| `tts_text.py` | TTS 에 넣기 전에 단위·쉼표·물결표를 읽는 말로 바꾸고 마크다운 기호를 뺀다. 문장 분할 함수도 있다 | 로컬 | — |
| `mic.py` | 마이크를 30ms 블록으로 읽고 블록마다 들어온 시각을 붙인다 | 로컬 | — |
| `vad.py` | webrtcvad 로 발화 하나를 잘라 낸다 (무음 700ms 면 끝) | 로컬 | — |
| `fake.py` | 키 없이 쓰는 가짜 STT · TTS (그래프 개발 · CI 용) | 로컬 | — |

그래프에서는 `from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech` 로 먼저 개발하고,
실제 구현이 들어오면 `SpeechToText`, `TextToSpeech` 로 바꿔 끼운다. 메서드 모양은 같다.

`listen()` 은 정리한 질문을 돌려주고, 원문은 `last_raw_text`, 받아쓰기 왕복 시간(ms)은
`last_latency_ms` 에 남긴다. 그래프는 원문을 `log_turn(..., stt_raw_text=...)` 로,
왕복 시간을 응답의 `latency_ms["stt"]` 로 넘긴다.

TTS 는 `user_setting.tts_speed` 를 OpenAI speed 값으로 바꿔 보낸다(SLOW 0.85, FAST 1.2, NORMAL 은 안 보냄).
첫 조각을 받기까지 걸린 시간은 `last_latency_ms` 에 남고, 그래프가 `latency_ms["tts"]` 로 넘기면
`turn_voice.tts_ms` 에 적힌다.

모델 이름은 `config.py` 에서만 정한다. 지금은 STT `gpt-transcribe`, TTS `gpt-4o-mini-tts`
(`.env` 의 `JARVISEO_STT_MODEL`, `JARVISEO_TTS_MODEL`).

## 알아둘 것

- **상시 동작하는 것은 호출어 감지뿐이고, 로컬에서 돈다.** 클라우드로 나가는 것은 호출어 뒤의 발화 한 번이다.
- `stt.py` 의 `started_at`(말을 시작한 시각)이 핵심이다. 이 시각으로 `capture` 가 프레임을 고른다.
  잡히는지는 `scripts/spike/spike_02_stt.py` 로 확인한다.
- **되먹임 루프 주의.** 스피커 소리를 마이크가 다시 듣는다. 재생 중에는 마이크를 막는다.
