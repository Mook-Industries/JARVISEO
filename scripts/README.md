# scripts — 기술 검증 · 보조 스크립트

| 폴더 | 무엇 |
|---|---|
| `spike/` | 기술 검증 5종 — 각 기술이 되는지만 확인하는 최소 코드 |
| `check_listen.py` | 실제 마이크로 `listen()` 확인 — 원문·정리된 질문·시각·지연 출력 |
| `check_mic_clock.py` | 마이크 블록 시각이 실제 시각을 따라가는지 — N분 동안 10초마다 샘플 수로만 센 시각(고치기 전)과 `BlockClock`(고친 뒤)이 얼마나 늦었는지, 끝에 장치 속도 차이(%). 소리는 저장 안 함 (#69) |
| `check_speaker_latency.py` | 스피커를 여는 시간과 출력 지연(`latency` high·low) — 첫 소리까지 걸리는 시간. 무음만 쓴다 (#104 시작음) |
| `check_chime_load.py` | 파이썬이 바쁠 때(같은 프로세스 스레드가 GIL 을 잡음) 시작음·종료음이 끊기는지 — 20ms씩 / 한 번에 쓰기별 소리가 빈 시간·play() 시간. `--listen` 은 귀로 비교. 소리 남 |
| `check_barge_turns.py` | 끼어드는 때(대답 읽는 중·되묻기 읽는 중·되묻기 대답 기다리는 중)마다 턴이 어떻게 나뉘고 total 이 얼마인지 — 실제 그래프·`BargeIn`·`resolve_target`·SQLite 기록에 시간이 흐르는 가짜 STT·TTS·VLM 을 끼워 재현. 키·마이크 없이 30초쯤 (#105) |
| `check_tts.py` | 실제 TTS 합성 확인 — 속도별 첫 소리까지 시간·길이 출력하고 재생 |
| `check_tts_stall.py` | TTS 합성이 읽는 도중 멈추는 비율을 모델마다 잰다(첫 조각·조각 사이 최대 멈춤). `--speak` 는 실제 `speak()` 가 폴백으로 끝까지 읽는지 본다. 소리 안 냄 (#130) |
| `check_filler_gap.py` | "잠깐만요" 뒤 대답 첫 소리까지의 빈틈 — assistant.py DB 의 VLM 지연·대답으로 미리 합성(prefetch) 끔/켬을 번갈아 재고 중앙값 출력. 실제 TTS, 소리는 안 냄 |
| `check_ingredient_ocr.py` | 실제 성분표 사진 OCR 확인 — 줄별 판독·신뢰도·지연과 원재료/알레르기 표시/같은 제조시설 구간 출력, 여러 장이면 신뢰도 요약 |
| `ocr_eval/evaluate.py` | 성분표 OCR 평가 — 정답 글자가 있는 사진으로 엔진·전처리(easyocr, easyocr-pre, paddle)별 CER·알레르겐 재현율/정밀도·머리말/혼입 문장·신뢰도·지연 비교, CSV 저장. `--init` 으로 정답 초안 생성 |
| `check_memory.py` | 기억 검색 측정 — 임베딩 왕복, 관찰 N행 검색 지연, 실제 임베딩 점수(맞는 장면·틀린 장면·상관없는 질문). 테스트용 DB 필요 |
| `stt_compare.py` | 받아쓰기 모델 비교 — 같은 질문 20개를 마이크로 녹음(`--record`)하거나 TTS 로 합성(`--tts`, `--noise` 로 배경 소리 섞기)해 모델별 지연 중앙값·p95·맞은 문장·CER 출력, `--silence` 는 말끝 대기별로 말이 중간에 끊긴 녹음 수. 녹음은 `data/datasets/stt/` |
| `eval_recall.py` | 기억 질문 평가셋(`data/eval/recall_set.csv`: 지난 대화 기록 20·질문 30)으로 기억 검색 측정 — 1등·5등 안 정답률, 맞는 기록에 답(keyword)이 있는지, 문턱별 정답 유지·기록에 없는 질문에 넘어가는 기록 수, 라우터가 RECALL 로 보내는지. `--set` 으로 #65 장면 설명 평가셋도 잰다. `--answer` 는 실제 VLM 대답으로 RAG 응답 근거율(근거 있음·모른다고 함·확인 필요, `data/eval/recall_answers.csv`)도 잰다. 실제 임베딩, SQLite 임시 DB |
| `run_ablation.py` | 가리킴 ablation — `tools/label_targets.py` 로 정답을 표시한 사진을 사진마다 한 번 검출하고, v1~v5(단서를 하나씩 쌓아 켬)로 고른 박스를 정답과 IoU 0.5 로 채점. 정답률 · 답한 것 정답률 · 되묻기율 · 손 있음/없음 · 후보 재현율 · 검출/대상 결정 p50/p95 · 실패 분류를 DB `eval_run` · `eval_sample` 과 사진별 CSV(`data/datasets/pointing/runs/`)에 남기고 같은 조건(버전 · 가중치 내용 지문 · 세션)의 이전 실행과 차이를 찍는다(커밋은 안 봄). `--weights` 에 가중치를 여러 개 주면(`yolov8n.pt data/models/detector_r1.pt`) 한 실행에서 가중치 × 버전으로 나란히 재고, 첫 가중치 대비 새로 잡힌 / 새로 놓친 사진을 찍는다(#167). 무엇으로 쟀는지(커밋 · 가중치 이름과 sha256 앞 12자리 · 세션 · conf · 장비 · ultralytics 버전)는 `eval_run.metrics["conditions"]`. 세션은 기본 valid 1 · 2(62장) · test 3, 말 · 맥락 세션은 `--sessions 4` 로 따로. 미구현 버전은 건너뜀, `--split test` 는 최종 측정 때만, `--db sqlite://` 로 돌리면 DB 기록 없이 확인만 — **단 사진별 CSV 는 그대로 쓰고, 이름의 run 번호가 1부터 다시 시작해 진짜 실행의 CSV 를 덮어쓸 수 있다.** 확인만 할 때는 `--out` 을 다른 폴더로 준다(예: `--out /tmp/ablation`). 손 단서(v2)를 켜면 사진마다 MediaPipe 로 가리키는 손을 한 번 찾고(`hand_landmarker.task` 필요), "방향 오판"은 손 단서가 실제로 점수를 낸 사진에만 붙는다 |
| `check_hands.py` | 손 관절 측정 — 같은 평가셋에서 MediaPipe 손 검출 기준(`--confidences`, 기본 0.5 · 0.3 · 0.1) × 가리킴 판정 기준(`--gaps`, 기본 0 · 15 · 30°)마다 손 찾음 · 가리킴 통과 · 정답 박스 중심까지 30° 안 · 각도 중앙값 · 손 없음 오탐 · 지연 p50/p95. 기본 기준(0.1 · 15°)의 사진별 CSV 를 `data/datasets/pointing/runs/hands/` 에, `--draw` 면 관절 · 선 · 정답 박스 그림도 남김. DB 기록 없음, `hand_landmarker.task` 필요(`src/jarviseo/pointing/README.md`) (#113) |
| `measure_latency.py` | 단계별 지연 리포트 — `assistant.py --db` 가 쌓은 세션 하나(기본은 턴이 있는 가장 최근 세션, `--session`)의 stt · route · detect · vlm · tts · total p50/p95/건수, 길별(가리킴 · 성분 · 나머지) total, "잠깐만요" 읽은 턴 비율. 끊긴 턴은 빼고, 30턴 미만이면 경고. DB 를 읽기만 함 (#98) |
| `check_vlm.py` | 실제 VLM 확인 — 웹캠 사진(또는 `--image`) 한 장과 질문을 보내 답과 왕복 시간 출력, `--memory` 로 기억 검색 결과를 넘겨 보기 |
| `check_setup.py` | `assistant.py` 를 켜기 전 점검 — 파이썬 3.12·실행 폴더·패키지·`.env` 키·모델 파일(호출어·openWakeWord 특징·EasyOCR·yolov8n·손 관절)·마이크·스피커·카메라·OpenAI 연결·DB 를 [통과/주의/실패] 로 찍고 실패마다 할 일 한 줄. 인자는 `assistant.py` 와 같음(`mic speaker --camera --db`), `--offline` 은 OpenAI 건너뜀. 실패가 있으면 끝 코드 1 (#141) |
| `download_models.py` | 처음 켜기 전에 모델 파일을 미리 받음 — 호출어 `jarviseo.onnx`(GitHub 릴리스 `models-v1`, SHA256 확인)·손 관절 모델·openWakeWord 특징 모델·EasyOCR 한국어·영어·`yolov8n.pt`. 있으면 건너뜀, 다 받은 뒤 `check_setup.py` 로 확인 (#141) |
| `start.py` | 아이콘 하나로 켜기 — 새 버전 받기(main 이고 고친 파일 없을 때 `git pull --ff-only`, 인터넷 없으면 건너뜀, `.env` 의 `JARVISEO_AUTO_UPDATE=false` 면 안 함) → requirements 가 바뀌었으면 `pip install` → `download_models.py` → `check_setup.py`(실패면 멈춤) → `assistant.py`. 더블클릭용 `start.bat`(윈도우)·`start.command`(맥)가 conda 환경 JARVISEO 의 파이썬을 찾아 이걸 켠다 (#141) |
| `schema_sql.py` | DB 스키마(`memory/models.py`)를 Postgres SQL 로 뽑아 `deploy/schema.sql` 에 씀. 클라우드 서버가 배포 때마다 이 파일로 없는 테이블 · 인덱스만 만든다. `--check` 는 파일이 모델과 같은지만 봄 (#143) |
| `measure_web_memory.py` | 팀 웹 컨테이너(`Dockerfile.web`) 메모리 · 응답 시간 — docker 로 띄워 `docker stats` 를 계속 읽으며 대기 · 질문(`/api/ask`) · 성분 확인(`/api/ingredient`) 구간별 최고 메모리와 응답 초를 찍음. `-e KEY=VALUE` 로 `.env` 값을 덮어씀(`-e GOOGLE_VISION_API_KEY=` 면 EasyOCR 로 잼). **OpenAI · Vision 을 실제로 부름** (#179) |
| `assistant.py` | 호출어 → 질문 → (기억 검색) → VLM → 음성 대답을 그래프로 도는 비서 — 턴마다 질문·의도·기억·대답·단계별 지연, 끝낼 때 중앙값 출력. 턴은 `data/assistant.db`, 턴마다 고른 사진·가리킴 후보 박스·VLM 에 간 크롭은 `data/captures/session_<번호>/` (`--no-photos` 로 끔) |
| `voice_loop.py` | Enter(또는 `--wake` 로 호출어) → STT → 에코 → TTS 음성 루프 — 턴마다 단계별 지연, 끝낼 때 구간마다 p50 · p95 · 건수 출력. `--csv` 는 턴마다 구간을 csv 에 바로 씀(#118). `--wake` 는 읽는 중 "자비서"로 끼어들기(재생 중 최고 점수·못 넘은 점수 출력), `--long` 긴 대답, `--db` 턴을 DB 에 기록, `--playback-threshold` 재생 중 임계값, `--chime` 시작음·종료음(호출 → 시작음, 말 끝 → 종료음 지연·되물음 횟수 출력, 빼면 끔), `--short` 질문 대신 되묻기 길이(약 2초) 문장을 읽어 그 사이 끼어들기가 되는지(읽는 중 끊김·끝난 뒤 늦은 감지 ms·못 잡으면 재생 중 최고 점수) (#105) |
| `wakeword/record.py` | 웨이크워드 녹음 — 말할 때마다 잘라 16kHz wav 와 화자·거리·환경 메타데이터 저장, `--tts` 로 합성 |
| `wakeword/evaluate.py` | 웨이크워드 평가 — 평가용 녹음으로 임계값별 FRR·FAR·시간당 오탐 계산, CSV 저장. `--thresholds` 로 임계값을 고르고(기본 config 0.97), 마이크가 둘 이상이면 마이크별로도 낸다 |
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
