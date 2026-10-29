# pointing — ① 지시 대상 특정 · 담당 최홍묵

**이 프로젝트의 핵심 기여.** 시야에 물건이 여러 개일 때 "저거"가 무엇인지 정한다.
착용형 기기에는 화면을 터치해 대상을 고를 방법이 없기 때문에 필요하다.

| 파일 | 하는 일 | 책임 지표 |
|---|---|---|
| `detector.py` | 물체 후보 검출 (기본 사전학습 COCO `yolov8n.pt`, `.env` 로 학습한 10종 가중치) | 후보 재현율 (식품 장면 따로) |
| `labels.py` | 검출 라벨의 한글 이름(COCO 80종 + `product` = "식품")과 종류 낱말표 `KIND_WORDS`(텀블러 → `bottle`, 캔 · 봉지 · 컵라면 → `product` 같은 별칭 포함). 되묻기 문장 · 대답 찾기 · 말 단서 · VLM 맥락 문장에 쓴다 | — |
| `hands.py` | 손 관절 21점(MediaPipe `HandLandmarker`) → 가리키는 손인지 · 검지 방향(5 → 8) | 손 검출률 · 가리킴 판정 · 방향 오차 (`scripts/check_hands.py`) |
| `cues.py` | 후보별 단서 점수 — v1 center(가운데 · 크기), v2 point(검지 끝 선 ↔ 후보 중심 각도, 가우시안 σ 15°), v3 gaze(가리킨 물건이 화면에 잘 오는 자리 — 학습용 세션 s05 의 2D 가우시안, 꺼 둠), v4 lang(질문 속 색 · 종류 · 위치 · 크기, 속성 점수를 곱함), v5 ctx("그거" · "아까" 같은 이어 묻기에서 직전 대상과 같은 종류) | — |
| `resolver.py` | 단서 융합 → 1위 결정 · 되묻기 판단 | 대상 선택 정확도 · 되묻기 비율 |
| `crop.py` | 고른 박스를 여백 15% 넣어 자르기 · VLM 크기 맞추기 · 손 우선 크롭(가리키는 손이 있는데 선 위에 후보가 없으면 손끝 방향 사각형) | 크롭 담김률 |

## 알아둘 것

- **손은 MediaPipe Hands(Tasks API `HandLandmarker`)의 관절 21개로 본다** (2026-10-14 결정).
  손끝 박스 하나로는 손가락 방향을 구할 수 없어서, 검지 뿌리(5번) → 끝(8번)을 쓴다. 지표는 안경 시점의 손 검출률 · 방향 오차.
  키보드에 얹거나 물건을 든 손도 손으로 잡히므로 "가리키는 손"인지 3D 관절 굽힘으로 따로 가른다(나머지 손가락 평균 − 검지 > 15°).
  가리키는 손이 없으면 손 단서는 0점이 아니라 기권이다. `resolve_target(hand=None)` 이면 그 사진에서 POINT 를 빼고
  켜진 나머지 단서 가중치로 다시 나누므로, 손 없는 사진의 v2 결과는 v1 과 같다(valid 손 없음 30.0% 그대로, `docs/experiments.md` v2 절).
- **말 단서(v4)는 질문 원문을 낱말 단위로 읽는다**(`parse_lang`, LLM 안 씀 — 라우터와 같은 지연 이유).
  "왼쪽 컵"은 컵 중 왼쪽이고(속성 점수를 곱함, 위치 · 크기는 종류가 맞는 후보 안에서 순위), 색은 박스 안쪽 픽셀 중 그 색(HSV 범위)의 비율이다.
  관계 말("컵 옆에 있는 거") · 쓸 말이 없을 때 · 후보를 못 가를 때는 손 단서처럼 기권한다. 색 범위는 표준 HSV 출발점이고 평가셋(s04)을 보며 고치지 않는다.
  **실제 비서에서 켜져 있다**(말 · 맥락 세션 s04 에서 미리 정한 기준을 넘어 플래그를 지움, #166 — `docs/experiments.md` "말 · 이어 묻기 단서").
  라우터(`workflow/router.py`)도 "이 병 · 왼쪽 컵 · 빨간 거" 같은 고르는 말을 가리킴으로 보내고, 낱말 역할은 `lang_role` 로 같은 표에서 읽는다.
  ablation 은 `enabled_cues` 로 직접 고른다(임시 버전 `v2+lang`, 색은 사진을 넘겨 봄).
- **이어 묻기 단서(v5)는 직전 대상의 이름만 본다**(앞 사진의 박스 자리는 고개를 돌리면 뜻이 없다). 질문에 되가리키는 말(그 · 그거 · 그것 · 아까 · 방금 등)이
  있을 때만 쓰고, 같은 이름의 후보에 1점이다. 직전 대상은 `workflow/nodes/pointing.py` 의 `detect` 가 같은 세션에서 60초 안의 가장 최근 가리킴 턴
  (`turn_inference.target_label`)을 DB 에서 한 번 읽는다 — 턴마다 그래프를 새로 불러 상태가 안 남고, 부르는 쪽(assistant.py · 웹)을 고치지 않으려는 것이다.
  DB 가 없거나 직전 대상이 없으면 기권. **`JARVISEO_ENABLE_CONTEXT_CUE` 로 꺼 둔 채다** — s04 에서 기준을 못 넘었다(같은 이름의 후보가 둘이면 못 가르고, 검출기가 이름을 잘못 붙이면 틀린 박스를 민다, #166). ablation 은 정답표의 `prev_label` 로 잰다.
- **시선 단서(v3)는 눈을 보지 않는다.** 안경에 눈 추적 장치가 없어서, 머리를 따라가는 카메라 화면에서 가리킨 물건이 잘 오는 자리를 쓴다.
  학습용 세션 s05(30장, 채점 안 함)의 정답 박스 중심을 0~1 좌표로 모아 평균 · 공분산을 맞추고(`tools/fit_gaze.py` → `cues.GAZE_MEAN` · `GAZE_COV`),
  후보 중심의 마할라노비스 거리(퍼짐을 감안한 거리)를 가우시안으로 바꾼다. 화면 속 자리만 보므로 기권하지 않는다.
  **`JARVISEO_ENABLE_GAZE_CUE` 로 꺼 둔 채다** — valid · s04 에서 정답은 1~6장 늘었지만 확신 오답이 3~6장 더 늘어 미리 정한 기준을 못 넘었다.
  가운데 근처 후보가 여럿이면 "조금 더 가운데"인 쪽이 크게 이겨 되묻던 사진을 단정으로 바꾸는데, 그 단정이 자주 틀렸다(#245, `docs/experiments.md` "시선 단서").
- **파인튜닝은 측정으로 필요가 확인될 때만** 한다. 진열대 장면에서 과자 봉지가 후보에 안 잡히면(후보 재현율이 낮으면) `product` 클래스를 학습한다.
- 진열대에서 "어느 물건을 가리켰나"까지가 이 모듈의 일이다. 성분 판정은 여기서 만든 크롭을 받아 시작한다.

- `cues.py` 가 **ablation 실험의 축**이다. 단서를 하나씩 켜고 끄며 기여도를 잰다.
- `resolver.py` 의 출력이 VLM 에 넘길 **크롭**을 결정한다. 여기가 틀리면 답도 틀린다.
- **손 우선 크롭**(#205): 가리키는 손이 있는데 모든 후보의 손 점수가 0.3 미만(선에서 약 23° 밖)이거나 후보가 0개면, 후보를 고르지 않고
  손끝에서 0.2H 앞을 중심으로 한 변 0.6H(H = 사진 높이)를 자른다(`crop.hand_crop_region`). 정답이 YOLO 후보에 없는데 손가락은 그쪽을
  가리킨 경우를 위한 것이다. 선 위에 후보가 있으면 평소처럼 고르거나 되묻는다. 기본으로 켜져 있고, ablation 은 임시 버전 `v2+C` 로 잰다.
- 1위와 2위 점수차(`margin`)가 임계값 미만이면 답하지 않고 되묻는다.
- **점수 규칙**(#215): 사진 넓이의 절반을 넘게 덮는 박스(냉장고 · 노트북 헛 박스)는 `detector.filter_detections` 가 후보에서 뺀다
  (`MAX_AREA_RATIO` 0.5 — 가운데 단서에서 늘 1위라 정답이 후보에 없을 때 그 박스를 확신했다). 원래 후보가 하나뿐이면 margin 이 1.0 이라,
  그 후보의 총점이 `resolver.MIN_SINGLE_SCORE`(0.3) 미만이면 되물을 문장 없이 단정하지 않는다(화면 구석 물건 하나). 손끝이 박스 안이면
  손 점수를 1.0 으로 주는 규칙은 재 보니 나빠져서(valid 정답 13 → 12) 넣지 않았다.
- **같은 물건 박스는 하나로**(#266): 검출기가 이름과 상관없이 IoU 0.7 넘게 겹친 박스를 확신 높은 하나만 남긴다(`predict(agnostic_nms=True)`).
  기본 NMS 는 같은 이름끼리만 지워서, 컵 하나에 `bottle` · `product` 박스가 같이 남고 같은 자리라 점수 차가 0 이 돼 같은 물건을 둘로 되물었다.
  노트북 위 휴대폰처럼 큰 박스 안의 작은 물건은 IoU 가 낮아 그대로 남는다. 실제 비서 설정은 ablation 임시 버전 `v2+lang+C` 로 잰다.
- 실험 기록은 `docs/experiments.md`, 결과 적재는 `memory/models.py` 의
  `Detection` · `TargetResolution`.

## 손 관절 모델 파일 받기

`hand_landmarker.task`(약 7.6MB)는 git 에 올라가지 않는다(`data/models/*`). **각자 한 번 받는다.**
없으면 `HandDetector` 가 이 문서를 가리키는 오류를 낸다. 레포 맨 위 폴더에서:

```bash
curl -fL -o data/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

윈도우 PowerShell 에서는 `curl` 이 다른 명령(`Invoke-WebRequest`)의 별명이라 옵션이 안 먹는다. `curl.exe` 로 쓴다.

```powershell
curl.exe -fL -o data/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

받은 파일이 7,819,105 바이트면 맞다. mediapipe 는 `requirements.txt`(`>=0.10.35,<1`)로 설치된다.
모델은 경로가 아니라 바이트로 읽어 넘긴다. 윈도우에서 경로에 한글이 섞이면 MediaPipe 가 파일을 못 열기 때문이다.

## 검출기 가중치 고르기 (#167)

`ObjectDetector()` 는 가중치를 `config.DETECTOR_WEIGHTS`(`.env` 의 `JARVISEO_DETECTOR_WEIGHTS`)에서 읽는다.
비워 두면 사전학습 COCO `yolov8n.pt` 다. 실행 루프(`scripts/assistant.py`) · 웹 · 정답 표시 도구가 모두 이 설정 하나를 따른다.

- 학습한 가중치는 공용 Drive 의 `detector/runs/rN/weights/best.pt` 를 `data/models/detector_rN.pt` 로 받는다(`data/README.md`).
- 회차 비교는 `.env` 를 바꿔 가며 재지 않고, `scripts/run_ablation.py --weights yolov8n.pt data/models/detector_r1.pt` 로 한 실행에서 잰다.
  같은 코드 · 같은 사진 · 같은 손 결과로 재야 표의 차이가 가중치 몫이 된다.
- 기본값(`yolov8n.pt`)을 학습한 가중치로 바꾸는 것은 회차 비교로 나아진 것을 확인한 뒤에 한다. 그 전까지는 시험할 때만 `.env` 로 바꾼다.
- 학습한 가중치는 10종(`FINETUNE_CLASSES`)만 찾는다. COCO 의 나머지(의자 · 가방 · 리모컨 등)는 후보에서 사라진다.
  R1(`detector_r1.pt`)만 `remote` 를 넣은 11종이다. `ObjectDetector` 는 가중치 안에 저장된 이름을 쓰므로 R1 도 그대로 잴 수 있다.
