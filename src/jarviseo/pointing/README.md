# pointing — ① 지시 대상 특정 · 담당 최홍묵

**이 프로젝트의 핵심 기여.** 시야에 물건이 여러 개일 때 "저거"가 무엇인지 정한다.
착용형 기기에는 화면을 터치해 대상을 고를 방법이 없기 때문에 필요하다.

| 파일 | 하는 일 | 책임 지표 |
|---|---|---|
| `detector.py` | 물체 후보 검출 (기본 사전학습 COCO `yolov8n.pt`, `.env` 로 학습한 11종 가중치) | 후보 재현율 (식품 장면 따로) |
| `labels.py` | 검출 라벨의 한글 이름(COCO 80종 + `product` = "식품")과 종류 낱말표 `KIND_WORDS`(텀블러 → `bottle`, 캔 · 봉지 · 컵라면 → `product` 같은 별칭 포함). 되묻기 문장 · 대답 찾기 · 말 단서 · VLM 맥락 문장에 쓴다 | — |
| `hands.py` | 손 관절 21점(MediaPipe `HandLandmarker`) → 가리키는 손인지 · 검지 방향(5 → 8) | 손 검출률 · 가리킴 판정 · 방향 오차 (`scripts/check_hands.py`) |
| `cues.py` | 후보별 단서 점수 — v1 center(가운데 · 크기), v2 point(검지 끝 선 ↔ 후보 중심 각도, 가우시안 σ 15°), v4 lang(질문 속 색 · 종류 · 위치 · 크기, 속성 점수를 곱함), v5 ctx("그거" · "아까" 같은 이어 묻기에서 직전 대상과 같은 종류). v3 은 아직 | — |
| `resolver.py` | 단서 융합 → 1위 결정 · 되묻기 판단 | 대상 선택 정확도 · 되묻기 비율 |

## 알아둘 것

- **손은 MediaPipe Hands(Tasks API `HandLandmarker`)의 관절 21개로 본다** (2026-10-14 결정).
  손끝 박스 하나로는 손가락 방향을 구할 수 없어서, 검지 뿌리(5번) → 끝(8번)을 쓴다. 지표는 안경 시점의 손 검출률 · 방향 오차.
  키보드에 얹거나 물건을 든 손도 손으로 잡히므로 "가리키는 손"인지 3D 관절 굽힘으로 따로 가른다(나머지 손가락 평균 − 검지 > 15°).
  가리키는 손이 없으면 손 단서는 0점이 아니라 기권이다. `resolve_target(hand=None)` 이면 그 사진에서 POINT 를 빼고
  켜진 나머지 단서 가중치로 다시 나누므로, 손 없는 사진의 v2 결과는 v1 과 같다(valid 손 없음 30.0% 그대로, `docs/experiments.md` v2 절).
- **말 단서(v4)는 질문 원문을 낱말 단위로 읽는다**(`parse_lang`, LLM 안 씀 — 라우터와 같은 지연 이유).
  "왼쪽 컵"은 컵 중 왼쪽이고(속성 점수를 곱함, 위치 · 크기는 종류가 맞는 후보 안에서 순위), 색은 박스 안쪽 픽셀 중 그 색(HSV 범위)의 비율이다.
  관계 말("컵 옆에 있는 거") · 쓸 말이 없을 때 · 후보를 못 가를 때는 손 단서처럼 기권한다. 색 범위는 표준 HSV 출발점이고 평가셋(s04)을 보며 고치지 않는다.
  **실제 비서에서는 `JARVISEO_ENABLE_LANGUAGE_CUE` 로 꺼 둔 채다.** 말 세트(s04)로 잰 뒤 라우터 보강과 같이 켜고 플래그를 지운다.
  ablation 은 `enabled_cues` 로 직접 켜서 플래그와 상관없이 잰다.
- **이어 묻기 단서(v5)는 직전 대상의 이름만 본다**(앞 사진의 박스 자리는 고개를 돌리면 뜻이 없다). 질문에 되가리키는 말(그 · 그거 · 그것 · 아까 · 방금 등)이
  있을 때만 쓰고, 같은 이름의 후보에 1점이다. 직전 대상은 `workflow/nodes/pointing.py` 의 `detect` 가 같은 세션에서 60초 안의 가장 최근 가리킴 턴
  (`turn_inference.target_label`)을 DB 에서 한 번 읽는다 — 턴마다 그래프를 새로 불러 상태가 안 남고, 부르는 쪽(assistant.py · 웹)을 고치지 않으려는 것이다.
  DB 가 없거나 직전 대상이 없으면 기권. **`JARVISEO_ENABLE_CONTEXT_CUE` 로 꺼 둔 채다**(말 · 맥락 세션 s04 로 잰 뒤 켬). ablation 은 정답표의 `prev_label` 로 잰다.
- **파인튜닝은 측정으로 필요가 확인될 때만** 한다. 진열대 장면에서 과자 봉지가 후보에 안 잡히면(후보 재현율이 낮으면) `product` 클래스를 학습한다.
- 진열대에서 "어느 물건을 가리켰나"까지가 이 모듈의 일이다. 성분 판정은 여기서 만든 크롭을 받아 시작한다.

- `cues.py` 가 **ablation 실험의 축**이다. 단서를 하나씩 켜고 끄며 기여도를 잰다.
- `resolver.py` 의 출력이 VLM 에 넘길 **크롭**을 결정한다. 여기가 틀리면 답도 틀린다.
- 1위와 2위 점수차(`margin`)가 임계값 미만이면 답하지 않고 되묻는다.
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
- 학습한 가중치는 11종(`FINETUNE_CLASSES`)만 찾는다. COCO 의 나머지(의자 · 가방 등)는 후보에서 사라진다.
