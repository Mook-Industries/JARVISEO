# data — 학습 데이터 · 가중치 · 영상

**Git 에 올리지 않는다.** 공용 Drive 에서 받아서 여기에 둔다.
`.gitkeep` 만 커밋되어 있고 내용물은 `.gitignore` 로 제외된다.

| 폴더 | 무엇이 들어가나 |
|---|---|
| `datasets/` | YOLO 학습·평가용 이미지와 라벨 |
| `datasets/wakeword/` | 웨이크워드 녹음과 평가 결과 (`scripts/wakeword/`) |
| `datasets/pointing/` | 가리킴 평가셋 — 촬영 사진 · 정답 표시 · 채점 결과 (`tools/capture_scenes.py` · `tools/label_targets.py` · `scripts/run_ablation.py`) |
| `datasets/detector/` | 검출기 파인튜닝 학습셋 — 안경 영상 · 뽑은 사진 · 라벨 · 회차별 묶음 (`tools/extract_frames.py` · `tools/auto_label.py`) |
| `models/` | 학습된 가중치 (`.pt`, `.onnx`) |
| `videos/` | 촬영한 원본 영상 |
| `eval/` | 작은 평가셋 CSV. 이것만 커밋한다 (`recall_set.csv`: 기억 질문, `scripts/eval_recall.py`) |

## 알아둘 것

- 카메라 없이 개발할 때 `datasets/sample/` 을 프레임 입력원으로 쓴다.
  (`.env` 의 `JARVISEO_FRAME_SOURCE=folder`)
- 용량이 큰 파일을 실수로 커밋하지 않도록 `git status` 를 꼭 확인한다.

## 가리킴 평가셋 보관 규칙

```
datasets/pointing/
├── raw/
│   ├── s01/            세션 하나 = 폴더 하나 (tools/capture_scenes.py)
│   │   ├── s01_desk_0001.jpg …
│   │   └── manifest.csv    file · session · place · utterance(찍을 때 한 말) · captured_at
│   └── s99/            연습 · 지연 측정용. 평가에 쓰지 않는다
├── labels.csv          정답 표시 (tools/label_targets.py). 한 줄 = 사진 한 장
└── runs/               채점 결과 사진별 CSV (scripts/run_ablation.py), <run_id>_<버전>.csv
```

`labels.csv` 열 (`run_ablation.py` 가 이 이름으로 읽는다):

| 열 | 뜻 |
|---|---|
| `file` | `raw/` 기준 경로 (`s01/s01_desk_0001.jpg`). 절대 경로는 다른 컴퓨터에서 깨진다 |
| `session` · `split` | 세션 번호와 `valid` / `test` |
| `utterance` | 찍을 때 한 말 (manifest 에서 옮김). 채점 때 질문 글자로 쓴다 |
| `gt_x1` ~ `gt_y2` | 정답 박스, 원본 해상도 픽셀. 제외한 사진은 빈칸 |
| `gt_label` | 정답 물건 이름. 영어 소문자 · 띄어쓰기, COCO 에 있으면 그 이름 그대로(`cell phone`) |
| `has_hand` | 가리킨 손이 화면에 보임 (`0`/`1`) |
| `ambiguous` | 사람이 봐도 무엇을 가리켰는지 모호함 — 되묻는 게 정답 (`0`/`1`) |
| `prev_label` | 앞 질문의 대상 라벨 (v5 맥락 단서용). 없으면 빈칸 |
| `excluded` | 흐림 등으로 쓸 수 없는 사진 (`0`/`1`). 채점에서 뺀다 |

- **정답은 이름이 아니라 박스로 남긴다.** 같은 컵이 둘이면 이름으로는 어느 쪽인지 구별이 안 된다. 채점은 고른 박스와 IoU 0.5 이상이면 맞음.
- **valid/test 는 세션 단위로 나눈다**(세션 1·2 valid, 3 test). test 는 최종 측정 전까지 채점하지 않고, 한 번 test 로 표시한 세션은 valid 로 옮기지 않는다.
- s99 를 연습으로 표시하면 그 줄도 `split=valid` 로 저장된다. `run_ablation.py` 는 세션을 가리지 않고 valid 줄을 전부 읽으므로, 기준선을 재기 전에 s99 줄을 지운다(`tools/README.md`).
- 참/거짓은 `True`/`False` 가 아니라 `0`/`1` 로 쓴다. 글자 `"False"` 를 `bool()` 로 읽으면 참이 되는 함정 때문이다.

## 검출기 학습셋 보관 규칙 (#167)

```
datasets/detector/
├── videos/             안경 카메라 영상. 이름은 d01_kitchen.mov (영상 번호 d + 두 자리 · 장소 영어 소문자)
├── frames/
│   └── d01/            영상 하나 = 폴더 하나. d01_0001.jpg … (tools/extract_frames.py)
├── frames.csv          file · video · t_sec — 사진이 어느 영상 몇 초에서 나왔나
├── auto/               자동 라벨(R1) — labels/*.txt · tasks.json(Label Studio) · label_config.xml
├── reviewed/           Label Studio 에서 검수하고 YOLO 로 내보낸 것(R2) — labels/ · classes.txt
└── yolo_r1/            회차 하나 = 폴더 하나 — images/{train,val} · labels/{train,val} · data.yaml
```

**종류 11개** (`src/jarviseo/pointing/detector.py` 의 `FINETUNE_CLASSES`, 순서 = 라벨 번호. 바꾸지 않고 새 종류는 맨 뒤에만)

| 번호 | 종류 | 경계 사례 |
|---|---|---|
| 0 | bottle | 병 모양이면 내용물과 상관없이(물병 · 텀블러 · 소스병) |
| 1 | cup | 마시는 컵 · 머그. 컵라면은 product |
| 2 | bowl | 그릇 · 접시 모양 용기 |
| 3 | laptop | 화면과 붙은 키보드까지 하나로 |
| 4 | mouse | 컴퓨터 마우스 |
| 5 | keyboard | 따로 놓인 키보드만 |
| 6 | cell phone | 휴대폰(COCO 이름 그대로 띄어쓰기) |
| 7 | remote | 리모컨 |
| 8 | book | 책 · 공책 |
| 9 | tv | TV · 모니터(COCO 관례) |
| 10 | product | COCO 에 없는 포장 식품 — 과자 봉지 · 컵라면 · 캔 · 햇반 · 병조림(잼 병) · 상자 포장 |

- **학습 사진에 보이는 11종은 전부 라벨한다.** 일부 종류만 라벨하면 나머지가 "배경"으로 학습돼 사라진다. 11종 밖 물건(안경 · 시계 등)은 라벨하지 않는다 → 학습한 가중치로는 안 잡힌다.
- 묶음(햇반 더미 · 책 여러 권)은 한 개씩. 절반 이상 보이면 라벨, 유리 · 화면에 비친 것은 안 한다. 손에 쥔 물건은 보이는 만큼.
- **학습 / 검증은 영상 단위로 나눈다**(`build --val d05`). 같은 영상에서 1초 차이로 뽑은 사진은 거의 같아서, 사진 단위로 섞으면 검증 점수가 부풀려진다.
  이 "검증"은 학습 중 best.pt 를 고르는 데만 쓴다. 회차 비교는 가리킴 valid(세션 1 · 2)의 후보 재현율로 한다.
- **가리킴 test 장소(라운지)와 valid 배치(s01 책상 · s02 선반 · s04 식탁)는 영상에 넣지 않는다.** 평가에 쓰는 장면을 학습하면 회차 비교와 최종 test 숫자가 부풀려진다.
- Colab 에는 `frames/` 와 가리킴 `raw/s01` · `raw/s02` · `labels.csv` 만 올린다. test(s03) 사진은 올리지 않는다.
- 얼굴 · 화면 속 메시지 같은 개인 정보는 찍지 않는다(사진이 Colab · Drive 로 간다).

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

## 호출어 모델 (`models/jarviseo.onnx`)

"자비서"로 학습한 openWakeWord 분류기다. `config.WAKEWORD_MODEL` 의 기본값이라 없으면 호출어 감지기가 시작하지 않는다.

- **`python scripts/download_models.py` 가 받는다.** 레포 GitHub 릴리스 `models-v1` 에 첨부한 파일이다.
  openWakeWord 특징 모델(melspectrogram · embedding)도 같이 받는다. 커스텀 모델(`.onnx` 경로)을 쓰면
  `voice/wakeword.py` 가 이 둘을 받지 않아서, 없으면 `NO_SUCHFILE ... melspectrogram.onnx` 로 멈춘다.
- 손으로 받을 때는 공용 Drive 의 `jarviseo/jarviseo-r6.onnx`(학습 6회차, 205,069 바이트)를 `models/jarviseo.onnx` 로 저장한다.
- 받은 파일이 맞는지는 SHA-256 앞자리로 확인한다: `4686db10ae2febb8` (`download_models.py` 는 전체 값을 확인한다)
  (`python -c "import hashlib;print(hashlib.sha256(open('data/models/jarviseo.onnx','rb').read()).hexdigest()[:16])"`)
- 학습 노트북은 `notebooks/wakeword_train.ipynb`, 평가 결과와 한계는 `docs/experiments.md` 학습 기록에 있다.
- 다시 학습해서 바꾸면 새 릴리스(`models-v2` …)에 올리고, 이 절과 `scripts/download_models.py` 의 `RELEASE_URL` · `WAKEWORD_SHA256` 을 같이 고친다.
