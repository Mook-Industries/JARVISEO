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
| `detector_finetune.ipynb` | 물체 후보 검출기(YOLOv8n) 회차 학습 — R2 부터. 맥에서 만든 묶음 zip 을 받아 같은 설정으로 학습하고 `<ROUND>_report.txt` 를 남긴다(`tools/README.md` "R2 사람 검수"). 셀 1 의 묶음 · 시드 목록으로 여러 번 학습한다 — 지금은 R2 · R3 묶음 × 시드 3개(`tools/README.md` "시드 반복"). T4 + 고용량 RAM(CPU 8)이 아니면 셀 1 에서 멈춘다(고용량 RAM 은 Colab Pro 에서만 보인다). R4(5 epoch 마다 저장 → `tools/auto_label.py pick`) 판은 커밋 `4bba733` 이다 |
