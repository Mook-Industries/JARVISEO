# pointing — ① 지시 대상 특정 · 담당 최홍묵

**이 프로젝트의 핵심 기여.** 시야에 물건이 여러 개일 때 "저거"가 무엇인지 정한다.
착용형 기기에는 화면을 터치해 대상을 고를 방법이 없기 때문에 필요하다.

| 파일 | 하는 일 | 책임 지표 |
|---|---|---|
| `detector.py` | 물체 후보 검출 (사전학습 COCO YOLO, 필요할 때만 `product` 파인튜닝) | 후보 재현율 (식품 장면 따로) |
| `hands.py` | 손 관절 21점(MediaPipe `HandLandmarker`) → 가리키는 손인지 · 검지 방향(5 → 8) | 손 검출률 · 가리킴 판정 · 방향 오차 (`scripts/check_hands.py`) |
| `cues.py` | 후보별 단서 점수 (v1~v5) | — |
| `resolver.py` | 단서 융합 → 1위 결정 · 되묻기 판단 | 대상 선택 정확도 · 되묻기 비율 |

## 알아둘 것

- **손은 MediaPipe Hands(Tasks API `HandLandmarker`)의 관절 21개로 본다** (2026-10-14 결정).
  손끝 박스 하나로는 손가락 방향을 구할 수 없어서, 검지 뿌리(5번) → 끝(8번)을 쓴다. 지표는 안경 시점의 손 검출률 · 방향 오차.
  키보드에 얹거나 물건을 든 손도 손으로 잡히므로 "가리키는 손"인지 3D 관절 굽힘으로 따로 가른다(나머지 손가락 평균 − 검지 > 15°).
  가리키는 손이 없으면 손 단서는 0점이 아니라 기권이다.
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
