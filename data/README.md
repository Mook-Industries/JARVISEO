# data — 학습 데이터 · 가중치 · 영상

**Git 에 올리지 않는다.** 공용 Drive 에서 받아서 여기에 둔다.
`.gitkeep` 만 커밋되어 있고 내용물은 `.gitignore` 로 제외된다.

| 폴더 | 무엇이 들어가나 |
|---|---|
| `datasets/` | YOLO 학습·평가용 이미지와 라벨 |
| `datasets/wakeword/` | 웨이크워드 녹음과 평가 결과 (`scripts/wakeword/`) |
| `models/` | 학습된 가중치 (`.pt`, `.onnx`) |
| `videos/` | 촬영한 원본 영상 |

## 알아둘 것

- 카메라 없이 개발할 때 `datasets/sample/` 을 프레임 입력원으로 쓴다.
  (`.env` 의 `JARVISEO_FRAME_SOURCE=folder`)
- 용량이 큰 파일을 실수로 커밋하지 않도록 `git status` 를 꼭 확인한다.

## 웨이크워드 녹음 보관 규칙

```
datasets/wakeword/
├── eval/       평가용. 학습에 절대 쓰지 않는다
│   ├── metadata.csv
│   ├── speech.csv   실제 "자비서" 녹음을 들어 보고 적은 말투(정상·끊음·늘임·잘림)
│   ├── wake/  similar/  background/
├── train/      학습용. 구조는 eval 과 같고, 일상 문장 합성(speech/)이 더 있다
├── oww/        prepare.py 가 train 을 openWakeWord 학습 폴더로 나눈 것과 그 zip. 다시 만들 수 있다
└── results/    evaluate.py 결과 CSV
```

- **평가용과 학습용은 녹음할 때부터 나눈다**(`record.py --split eval|train`).
  학습에 쓴 소리로 평가하면 FRR·FAR 이 실제보다 좋게 나와서, 커스텀 학습이 나아졌는지 판단할 수 없다.
- **한 번 `eval/` 에 넣은 파일은 `train/` 으로 옮기지 않는다.** 학습 데이터가 모자라면 새로 녹음한다.
- 가능하면 평가용 화자는 학습용에 쓰지 않는다. 겹치면 그 사실을 `docs/experiments.md` 에 적는다.
- 화자는 실명 대신 번호(`s01`, `s02` …)로 적는다.
- Drive 에 올릴 때도 이 폴더 구조와 `metadata.csv` 를 그대로 올린다. 메타데이터가 없으면 평가 스크립트가 읽지 못한다.
- 잘못 녹음한 파일은 wav 만 지우면 된다. `metadata.csv` 에 줄이 남아 있어도 평가할 때 건너뛴다.
  지우기 아까우면 `speech.csv` 에 `잘림` 으로 적는다. `evaluate.py` 가 평가에서 빼고 뺀 개수를 출력한다.
- 팀원 녹음은 `train` 폴더를 `train_s0X.zip` 으로 받아 `scripts/wakeword/merge.py` 로 합치고,
  합친 뒤와 TTS 로 합성한 뒤에는 `scripts/wakeword/check.py <split> --fix` 로 한 번 거른다.
