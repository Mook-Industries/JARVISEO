# notebooks — Colab 학습 노트북

YOLO 파인튜닝처럼 **GPU 가 필요한 작업**을 여기서 돌린다.
팀 세 명 모두 Colab Pro 를 쓰고 있다.

## 규칙

- 노트북은 **학습·실험 전용**이다. 서비스 코드는 `src/jarviseo/` 에 둔다.
  노트북에서 검증이 끝난 로직만 모듈로 옮긴다.
- 커밋 전에 **출력 셀을 지운다.** 이미지가 들어간 출력은 diff 를 못 읽게 만든다.
- 학습 결과(mAP 등)는 `docs/experiments.md` 에 한 줄로 남긴다.
- 학습된 가중치는 `data/models/` 로 내려받고 Git 에는 올리지 않는다.

## 노트북

| 이름 | 무엇 |
|---|---|
| `wakeword_train.ipynb` | "자비서" 호출어 openWakeWord 분류기 학습 |
| `detector_finetune.ipynb` | 물체 후보 검출기(YOLOv8n) 회차 학습 — R2 부터. 맥에서 만든 묶음 zip 을 받아 같은 설정으로 학습하고 `R2_report.txt` 를 남긴다(`tools/README.md` "R2 사람 검수") |
