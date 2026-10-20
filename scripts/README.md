# scripts — 기술 검증 · 보조 스크립트

| 폴더 | 무엇 |
|---|---|
| `spike/` | 기술 검증 5종 — 각 기술이 되는지만 확인하는 최소 코드 |
| `check_listen.py` | 실제 마이크로 `listen()` 확인 — 원문·정리된 질문·시각·지연 출력 |
| `check_mic_clock.py` | 마이크 블록 시각이 실제 시각을 따라가는지 — N분 동안 10초마다 샘플 수로만 센 시각(고치기 전)과 `BlockClock`(고친 뒤)이 얼마나 늦었는지, 끝에 장치 속도 차이(%). 소리는 저장 안 함 (#69) |
| `check_speaker_latency.py` | 스피커를 여는 시간과 출력 지연(`latency` high·low) — 첫 소리까지 걸리는 시간. 무음만 쓴다 (#104 시작음) |
| `check_barge_turns.py` | 끼어드는 때(대답 읽는 중·되묻기 읽는 중·되묻기 대답 기다리는 중)마다 턴이 어떻게 나뉘고 total 이 얼마인지 — 실제 그래프·`BargeIn`·`resolve_target`·SQLite 기록에 시간이 흐르는 가짜 STT·TTS·VLM 을 끼워 재현. 키·마이크 없이 30초쯤 (#105) |
| `check_tts.py` | 실제 TTS 합성 확인 — 속도별 첫 소리까지 시간·길이 출력하고 재생 |
| `check_filler_gap.py` | "잠깐만요" 뒤 대답 첫 소리까지의 빈틈 — assistant.py DB 의 VLM 지연·대답으로 미리 합성(prefetch) 끔/켬을 번갈아 재고 중앙값 출력. 실제 TTS, 소리는 안 냄 |
| `check_ingredient_ocr.py` | 실제 성분표 사진 OCR 확인 — 줄별 판독·신뢰도·지연과 원재료/알레르기 표시/같은 제조시설 구간 출력, 여러 장이면 신뢰도 요약 |
| `ocr_eval/evaluate.py` | 성분표 OCR 평가 — 정답 글자가 있는 사진으로 엔진·전처리(easyocr, easyocr-pre, paddle)별 CER·알레르겐 재현율/정밀도·머리말/혼입 문장·신뢰도·지연 비교, CSV 저장. `--init` 으로 정답 초안 생성 |
| `check_memory.py` | 기억 검색 측정 — 임베딩 왕복, 관찰 N행 검색 지연, 실제 임베딩 점수(맞는 장면·틀린 장면·상관없는 질문). 테스트용 DB 필요 |
| `stt_compare.py` | 받아쓰기 모델 비교 — 같은 질문 20개를 마이크로 녹음(`--record`)하거나 TTS 로 합성(`--tts`, `--noise` 로 배경 소리 섞기)해 모델별 지연 중앙값·p95·맞은 문장·CER 출력, `--silence` 는 말끝 대기별로 말이 중간에 끊긴 녹음 수. 녹음은 `data/datasets/stt/` |
| `eval_recall.py` | 기억 질문 평가셋(`data/eval/recall_set.csv`: 지난 대화 기록 20·질문 30)으로 기억 검색 측정 — 1등·5등 안 정답률, 맞는 기록에 답(keyword)이 있는지, 문턱별 정답 유지·기록에 없는 질문에 넘어가는 기록 수, 라우터가 RECALL 로 보내는지. `--set` 으로 #65 장면 설명 평가셋도 잰다. 실제 임베딩, SQLite 임시 DB |
| `run_ablation.py` | 가리킴 ablation — `tools/label_targets.py` 로 정답을 표시한 사진을 사진마다 한 번 검출하고, v1~v5(단서를 하나씩 쌓아 켬)로 고른 박스를 정답과 IoU 0.5 로 채점. 정답률 · 답한 것 정답률 · 되묻기율 · 손 있음/없음 · 후보 재현율 · 검출/대상 결정 p50/p95 · 실패 분류를 DB `eval_run` · `eval_sample` 과 사진별 CSV(`data/datasets/pointing/runs/`)에 남기고 같은 버전의 이전 실행과 차이를 찍는다. 미구현 버전은 건너뜀, `--split test` 는 최종 측정 때만, `--db sqlite://` 로 돌리면 기록 없이 확인만 |
| `check_vlm.py` | 실제 VLM 확인 — 웹캠 사진(또는 `--image`) 한 장과 질문을 보내 답과 왕복 시간 출력, `--memory` 로 기억 검색 결과를 넘겨 보기 |
| `assistant.py` | 호출어 → 질문 → (기억 검색) → VLM → 음성 대답을 그래프로 도는 비서 — 턴마다 질문·의도·기억·대답·단계별 지연, 끝낼 때 중앙값 출력. 턴은 `data/assistant.db`, 턴마다 고른 사진·가리킴 후보 박스·VLM 에 간 크롭은 `data/captures/session_<번호>/` (`--no-photos` 로 끔) |
| `voice_loop.py` | Enter(또는 `--wake` 로 호출어) → STT → 에코 → TTS 음성 루프 — 턴마다 단계별 지연, 끝낼 때 중앙값 출력. `--wake` 는 읽는 중 "자비서"로 끼어들기(재생 중 최고 점수·못 넘은 점수 출력), `--long` 긴 대답, `--db` 턴을 DB 에 기록, `--playback-threshold` 재생 중 임계값, `--chime` 시작음·종료음(호출 → 시작음, 말 끝 → 종료음 지연·되물음 횟수 출력, 빼면 끔), `--short` 질문 대신 되묻기 길이(약 2초) 문장을 읽어 그 사이 끼어들기가 되는지(읽는 중 끊김·끝난 뒤 늦은 감지 ms·못 잡으면 재생 중 최고 점수) (#105) |
| `wakeword/record.py` | 웨이크워드 녹음 — 말할 때마다 잘라 16kHz wav 와 화자·거리·환경 메타데이터 저장, `--tts` 로 합성 |
| `wakeword/evaluate.py` | 웨이크워드 평가 — 평가용 녹음으로 임계값별 FRR·FAR·시간당 오탐 계산, CSV 저장 |
| `wakeword/check.py` | 웨이크워드 녹음 검사 — 무음·잡음 합성 클립, 형식이 틀린 wav, metadata 와 안 맞는 파일을 찾아 `--fix` 로 지움 |
| `wakeword/merge.py` | 팀원 녹음 zip 합치기 — 화자 번호를 붙여 train·eval 에 넣고, 다른 split 에 있는 화자면 멈춤 |
| `wakeword/lag.py` | 호출어 감지가 "자비서" 끝보다 얼마나 늦는지 — 평가셋 실제 녹음으로 감지 시각 - 말 끝 분포 출력 |
| `wakeword/lookback.py` | 호출어 + 질문을 이어 붙인 소리로 `listen(since=감지 시각)` 을 실제 STT 로 돌려 보기 — `--lookback`·`--gap` 비교, `--chime` 은 시작음을 질문보다 몇 dB 작게 섞어 받아쓰기 비교 |
| `wakeword/self_voice.py` | 자기 목소리(TTS 대답)의 호출어 점수 — 대답 문장을 합성해 프레임 최고 점수 출력 |
| `wakeword/prepare.py` | 웨이크워드 학습 데이터 준비 — train 녹음의 앞뒤 무음을 잘라 openWakeWord 학습 폴더(positive·negative × train·test)로 나누고 zip 으로 묶음 |

## spike 란

"이게 되긴 하나"를 가장 짧은 코드로 확인하는 것. 제품 코드가 아니다.
검증이 끝나면 여기 코드를 그대로 쓰지 말고 `src/jarviseo/` 에 제대로 옮긴다.

- `spike_01_capture.py` 카메라 프레임 받기
- `spike_02_stt.py` 음성 인식
- `spike_03_vlm.py` VLM 호출
- `spike_04_graph.py` LangGraph 최소 그래프
- `spike_05_yolo.py` YOLO 추론
