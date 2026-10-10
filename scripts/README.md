# scripts — 기술 검증 · 보조 스크립트

| 폴더 | 무엇 |
|---|---|
| `spike/` | 기술 검증 5종 — 각 기술이 되는지만 확인하는 최소 코드 |
| `check_listen.py` | 실제 마이크로 `listen()` 확인 — 원문·정리된 질문·시각·지연 출력 |
| `check_tts.py` | 실제 TTS 합성 확인 — 속도별 첫 소리까지 시간·길이 출력하고 재생 |
| `check_ingredient_ocr.py` | 실제 성분표 사진 OCR 확인 — 줄별 판독·신뢰도·지연과 원재료/알레르기 표시/같은 제조시설 구간 출력, 여러 장이면 신뢰도 요약 |
| `check_memory.py` | 기억 검색 측정 — 임베딩 왕복, 관찰 N행 검색 지연, 실제 임베딩 점수(맞는 장면·틀린 장면·상관없는 질문). 테스트용 DB 필요 |
| `voice_loop.py` | Enter → STT → 에코 → TTS 음성 루프 — 턴마다 단계별 지연, 끝낼 때 중앙값 출력 |
| `wakeword/record.py` | 웨이크워드 녹음 — 말할 때마다 잘라 16kHz wav 와 화자·거리·환경 메타데이터 저장, `--tts` 로 합성 |
| `wakeword/evaluate.py` | 웨이크워드 평가 — 평가용 녹음으로 임계값별 FRR·FAR·시간당 오탐 계산, CSV 저장 |
| `wakeword/check.py` | 웨이크워드 녹음 검사 — 무음·잡음 합성 클립, 형식이 틀린 wav, metadata 와 안 맞는 파일을 찾아 `--fix` 로 지움 |
| `wakeword/merge.py` | 팀원 녹음 zip 합치기 — 화자 번호를 붙여 train·eval 에 넣고, 다른 split 에 있는 화자면 멈춤 |

## spike 란

"이게 되긴 하나"를 가장 짧은 코드로 확인하는 것. 제품 코드가 아니다.
검증이 끝나면 여기 코드를 그대로 쓰지 말고 `src/jarviseo/` 에 제대로 옮긴다.

- `spike_01_capture.py` 카메라 프레임 받기
- `spike_02_stt.py` 음성 인식
- `spike_03_vlm.py` VLM 호출
- `spike_04_graph.py` LangGraph 최소 그래프
- `spike_05_yolo.py` YOLO 추론
