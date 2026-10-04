# voice — 음성 입출력 · 담당 문태현

흐름: 호출어 감지 → VAD 로 발화 구간 파악 → STT → (처리) → TTS

| 파일 | 하는 일 | 어디서 도나 | 책임 지표 |
|---|---|---|---|
| `wakeword.py` | "자비서" 호출어 감지 | 로컬 (상시) | FAR / FRR |
| `stt.py` | 호출어 뒤 발화 받아쓰기 | OpenAI STT | — |
| `tts.py` | 응답 음성 합성 · 재생 | OpenAI TTS | — |
| `fake.py` | 키 없이 쓰는 가짜 STT · TTS (그래프 개발 · CI 용) | 로컬 | — |

그래프에서는 `from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech` 로 먼저 개발하고,
실제 구현이 들어오면 `SpeechToText`, `TextToSpeech` 로 바꿔 끼운다. 메서드 모양은 같다.

모델 이름은 `config.py` 에서만 정한다. 지금은 STT `gpt-transcribe`, TTS `gpt-4o-mini-tts`
(`.env` 의 `JARVISEO_STT_MODEL`, `JARVISEO_TTS_MODEL`).

## 알아둘 것

- **상시 동작하는 것은 호출어 감지뿐이고, 로컬에서 돈다.** 클라우드로 나가는 것은 호출어 뒤의 발화 한 번이다.
- `stt.py` 의 `started_at`(말을 시작한 시각)이 핵심이다. 이 시각으로 `capture` 가 프레임을 고른다.
  잡히는지는 `scripts/spike/spike_02_stt.py` 로 확인한다.
- **되먹임 루프 주의.** 스피커 소리를 마이크가 다시 듣는다. 재생 중에는 마이크를 막는다.
