# tools — 개발용 도구 (장치 점검 · 가리킴 평가셋 · 검출기 학습셋)

개발용 도구. 서비스 코드가 아니라서 `src/` 와 분리해 둔다.

| 이름 | 무엇 |
|---|---|
| `device_check/` | 카메라·마이크가 제대로 물렸는지 확인하는 5종 |
| `capture_scenes.py` | 가리킴 평가 장면 촬영 — 안경 카메라로 찍고, 그때 한 말을 `manifest.csv` 에 같이 남긴다 |
| `label_targets.py` | 가리킴 정답 표시 — 찍은 사진에 YOLO 후보를 번호로 띄우고, 정답을 골라(없으면 박스를 그려) `labels.csv` 에 남긴다 |
| `extract_frames.py` | 검출기 학습 사진 — 안경 카메라 영상에서 3초에 1장(`--every`, 기본 3) 뽑고, 거의 같은 장면은 버린다 |
| `auto_label.py` | 검출기 학습 라벨 — 선생 모델 비교(`compare`), 박스 초안(`label`), 사람 검수 작업(`review`), 회차별 학습 묶음(`build`) |

## device_check

```bash
python tools/device_check/find_camera.py     # 카메라 인덱스 찾기
python tools/device_check/preview.py         # 미리보기 (YOLO 없이)
python tools/device_check/detect_overlay.py  # 실시간 객체 검출 오버레이
python tools/device_check/mic_test.py        # 마이크 입력 레벨
python tools/device_check/mic_snr_test.py    # 신호 대 잡음비(SNR) 측정
```

- 맥/윈도우/리눅스 모두에서 돈다. 백엔드는 스크립트가 알아서 고른다.
- 산출물(캡처 이미지, wav)은 `device_check/_captures/` 에 생기고 Git 에서 제외된다.
- **처음 실행 시 카메라·마이크 권한 팝업**이 뜬다. 맥은 허용 후 앱을 완전히 재시작해야 한다.

## 가리킴 평가셋 — 촬영 → 정답 표시 → 채점

```bash
python tools/capture_scenes.py --session 1 --place desk   # 1. 촬영: 스페이스 → 터미널에 그때 한 말
python tools/label_targets.py --session 1                 # 2. 정답 표시 (test 세션은 --split test)
python scripts/run_ablation.py --note "v1 기준선"          # 3. v1~v5 채점 · 기록
```

- **valid/test 는 세션 단위로 나눈다.** 같은 자리에서 연달아 찍은 사진은 거의 같아서, 사진 단위로 섞으면 test 가 valid 를 베낀 셈이 된다.
  세션 1·2 는 valid, 3 은 test. test 는 최종 측정 전까지 채점하지 않는다. 세션 99 는 연습·지연 측정용이다.
  연습으로 표시한 s99 줄도 `split=valid` 로 저장돼서 그대로 두면 valid 숫자에 섞인다. 기준선을 재기 전에 지운다:
  `sed -i '' '/^s99\//d' data/datasets/pointing/labels.csv`
- 정답 표시는 사진 한 장마다 이 순서로 한다. 터미널에도 사진마다 키 안내가 찍힌다.
  1. 가리킨 손이 화면에 보이면 `h` (왼쪽 위 `hand=1`). 사람이 봐도 무엇을 가리켰는지 모르겠으면 `a`, 앞 질문에 이어진 질문이면 `p` 로 앞 대상 라벨
  2. 가리킨 물건에 초록 상자가 있으면 그 번호(`1`~`9`) → 터미널을 클릭해 정답 이름 확인 — 초록 글자의 YOLO 이름이 맞으면 `Enter`,
     틀리면 이름을 쳐서 `Enter` → 그림 창을 다시 클릭 → 끝. 박스는 맞아도 YOLO 이름은 틀릴 수 있다(안경 케이스를 `remote`, 텀블러를 `parking meter` 로 부름)
  3. 초록 상자가 없으면 `d` → 그림 창에서 대상 둘레를 끌고 `Enter`(다시 그리려면 다시 끌기, 취소 `c`) → 그림 창이 멈춘 것처럼 보이는 게 정상 →
     터미널을 클릭해 라벨 입력 → `저장:` 이 찍히면 그림 창을 다시 클릭
  4. 흐리거나 쓸 수 없는 사진은 `x`. 그만두려면 `q`, 같은 명령으로 이어서 한다(이미 표시한 사진은 다시 묻지 않는다)
- **후보에 없다고 비슷한 후보를 고르지 않는다.** 엉뚱한 박스가 정답으로 저장돼 채점이 틀어진다.
  직접 그린 박스는 `run_ablation.py` 가 "후보 없음"으로 세서, 검출기(COCO YOLO)가 가리킨 물건을 못 잡는 비율이 후보 재현율로 나온다.
- **라벨 이름은 영어 소문자, 단어 사이는 띄어쓰기.** COCO 에 있는 물건은 그 이름 그대로(`cell phone`, `cup`), 없는 물건도 같은 규칙(`air conditioner`, `cup noodle`). 같은 물건은 늘 같은 이름으로 쓴다.
- 터미널에 입력한 뒤에는 그림 창을 한 번 클릭해야 키가 다시 먹는다(두 도구 모두). `cv2.waitKey` 는 그림 창에 포커스가 있을 때만 키를 받는다.
- 잘못 표시한 사진은 `labels.csv` 에서 그 줄을 지우고 다시 실행하면 그 사진만 다시 묻는다.
  (`sed -i '' '/s99_spike_0002\.jpg/d' data/datasets/pointing/labels.csv`, 윈도우는 편집기로 그 줄을 지운다)
- 폴더 구조와 `labels.csv` 열은 `data/README.md`, 채점 방식과 결과 표는 `docs/experiments.md` "Ablation".

## 검출기 학습셋 — 영상 → 사진 → 자동 라벨 → 검수 → 학습 묶음 (#167)

```bash
# 맥: 영상(data/datasets/detector/videos/d01_kitchen.mov …)에서 사진 뽑기
python tools/extract_frames.py                      # 3초에 1장(기본). 영상마다 "남은 M장" — 3분 영상에 약 60장
# Colab: frames/ 와 pointing 의 s01 · s02 · labels.csv 를 Drive 에 올린 뒤
python tools/auto_label.py compare --pointing <Drive>/pointing         # 선생 비교 (YOLO-World vs yolo11x)
python tools/auto_label.py label --root <Drive>/detector --teacher world  # 박스 초안 → auto/
python tools/auto_label.py build --root <Drive>/detector --labels <Drive>/detector/auto/labels \
    --val d05 --out <Drive>/detector/yolo_r1                           # R1 학습 묶음
```

- **사진 양은 간격으로 정한다(기본 3초).** 안경 카메라는 고개를 따라 계속 움직여서 1초 간격 사진끼리도 지문이 19~26비트(64 중) 달라,
  중복 거르기(6비트 이하)로는 거의 안 줄었다(영상 6개 1133장). 3초 간격은 406장으로, R2 검수를 하루 안에 끝낼 수 있는 양이다.
  R3 보강 영상도 같은 간격으로 뽑아야 회차 비교가 맞는다.
- 뽑은 뒤 사람이 봐도 알아볼 수 없는 사진 · 얼굴이 찍힌 사진은 `excluded/` 로 옮긴다(`data/README.md` "검출기 학습셋 보관 규칙").
- **`compare` · `label` · `review` 는 CLIP 이 있는 곳에서만 돈다.** ultralytics 는 YOLO-World · YOLOE 의 글귀를 처음 읽을 때
  CLIP 을 git 에서 자동으로 설치해서, 팀 파이썬 환경에서 돌리면 승인 없이 패키지가 깔린다. 그래서 CLIP 이 없으면 도구가 미리 멈춘다.
  Colab 은 먼저 `pip install git+https://github.com/ultralytics/CLIP.git`, 맥은 아래 "R2 사람 검수"의 CLIP 가상환경에서 돌린다.
  `extract_frames.py` · `build` 는 모델을 안 써서 어디서든 돈다.
- **회차 폴더는 덮어쓰지 않는다.** `auto/`(R1 라벨) · `yolo_r1/` 이 이미 있으면 도구가 멈춘다. 다시 만들려면 이름을 바꿔 보관한다.
  회차를 같은 조건에서 다시 학습 · 비교하려면 그 회차의 라벨과 묶음이 그대로 남아 있어야 한다.
- R2 는 사람이 검수한 라벨로 학습한다. 검수 작업 만들기 · Label Studio · 검수 규칙은 아래 "R2 사람 검수".
  `build --labels` 에 준 폴더에 `classes.txt` 가 있으면 번호 순서가 우리 종류 표와 달라도 이름으로 다시 맞춘다(우리 표에 없는 이름이면 멈춘다).
- 학습한 가중치는 `data/models/detector_rN.pt` 로 받아 `scripts/run_ablation.py --weights yolov8n.pt data/models/detector_r1.pt` 처럼
  R0 와 한 실행에서 나란히 잰다(`scripts/README.md`). 날마다 따로 잰 숫자끼리 비교하면 그사이 바뀐 코드 효과가 섞인다.
- 종류 정의 · 경계 사례 · 분할 규칙 · 폴더 구조는 `data/README.md` "검출기 학습셋 보관 규칙".

## R2 사람 검수 — 검수 작업 → Label Studio (#167)

R1 은 자동 라벨을 그대로 학습해서 후보 재현율이 R0 보다 떨어졌다(50.0 → 37.1%, `docs/experiments.md`).
같은 물건이 어떤 사진엔 라벨이 있고 어떤 사진엔 없으면 모델이 그 물건에 확신을 낮추도록 배운다. 그래서 R2 는 사람이 고친 라벨로 학습한다.

### 1. CLIP 가상환경 (한 번만)

`review` 는 YOLOE 의 글귀를 읽으려고 CLIP 이 필요하다. 팀 conda 환경에 깔리지 않게 **프로젝트 밖 폴더**에 가상환경을 따로 만든다.
`--system-site-packages` 라서 torch · ultralytics 는 conda 환경 것을 그대로 쓰고, CLIP 만 이 가상환경 안에 깔린다.

```bash
/opt/miniconda3/envs/JARVISEO/bin/python -m venv --system-site-packages ~/jarviseo-clipenv
~/jarviseo-clipenv/bin/python -m pip install git+https://github.com/ultralytics/CLIP.git
```

### 2. 검수 작업 만들기

YOLOE 가중치(172MB)와 글귀 모델 MobileCLIP2(242MB)는 **명령을 실행한 폴더**에 받아진다. 레포에 남지 않게 프로젝트 밖 폴더에서 돌린다.
`-I` 는 지금 폴더의 파일을 파이썬 모듈로 읽지 않게 한다(받은 파일이 있는 폴더에서 돌리므로).

```bash
mkdir -p ~/jarviseo-models && cd ~/jarviseo-models
~/jarviseo-clipenv/bin/python -I <레포>/tools/auto_label.py review --root <레포>/data/datasets/detector
```

- 만드는 것: `review/tasks.json`(Label Studio 작업) · `review/label_config.xml`(프로젝트 설정). `review/` 가 이미 있으면 멈춘다.
  검수 도중 덮어쓰면 Label Studio 로 이미 가져간 작업과 어긋난다.
- 초안 = R1 초안 그대로 + YOLOE 가 찾은 식품(product) 박스를 **R1 박스와 IoU 0.5 미만인 빈 자리에만** 더한 것.
  YOLOE 의 "box" 글귀는 상자처럼 생긴 것을 다 잡아서, 점수로 겨루게 하면 키보드 · 노트북 초안이 product 로 덮인다.
- R1 의 `remote` 초안은 빼고, 그 사진 위에 "R1 초안에서 10종 밖 remote 1개를 뺐습니다"를 띄운다(작업의 `note`). 짧은 변 40px 미만 박스도 뺀다.
- 작업 순서(`order`, 가져온 순서 그대로): d05 짝수 번 → 학습 영상을 번갈아(d01 → d02 → d03 → d04 → d06 → …, 영상 안에서는 짝수 번 먼저) → d05 홀수 번.
  3시간이면 학습 사진의 절반쯤만 본다. 영상 순서대로 보면 앞 영상만 검수하게 되고, 3초 간격이라 이웃 사진이 거의 같아서 짝수 번을 먼저 본다.

### 3. Label Studio 띄우기 (레포 맨 위 폴더에서)

```bash
docker run -d --name JARVISEO-label-studio -p 127.0.0.1:8080:8080 \
  -v jarviseo-label-studio:/label-studio/data \
  -v "$PWD/data/datasets/detector/frames":/label-studio/files/frames:ro \
  -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
  -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \
  heartexlabs/label-studio:1.23.2
```

- `127.0.0.1:` 이라 이 컴퓨터에서만 열린다. 같은 와이파이의 다른 기기에서는 사진을 볼 수 없다.
- 사진은 `frames/` 만 **읽기 전용**(`:ro`)으로 붙인다. 검수 도구가 원본을 못 바꾸고, `excluded/`(얼굴 사진)는 컨테이너에 보이지도 않는다.
- 계정 · 프로젝트 · 검수한 것은 Docker 볼륨 `jarviseo-label-studio` 에 남는다. `docker stop JARVISEO-label-studio` 로 껐다가
  `docker start JARVISEO-label-studio` 로 켜도 그대로다.
- 이미지는 버전을 고정한다. `latest` 는 받는 날마다 바뀔 수 있다.

### 4. 프로젝트 설정 (브라우저 http://localhost:8080)

1. 계정을 만들고 프로젝트를 만든다.
2. 라벨 설정(Labeling Setup)에서 직접 쓰기(Custom template)를 고르고 `review/label_config.xml` 내용을 붙여넣는다.
   밝기 · 대비 · 확대 버튼이 켜져 있고 박스 회전은 꺼져 있다(YOLO 박스는 돌릴 수 없다).
3. 설정 → Cloud Storage → 원본 저장소(Source Storage) 추가 → Local files, 경로 `/label-studio/files/frames`.
   **이 저장소가 없으면 사진이 안 열린다.** Label Studio 는 저장소가 가리키는 폴더 안의 파일만 내준다(1.23 `io_storages/localfiles/views.py`).
   경로를 문서 루트(`/label-studio/files`) 그대로 쓰면 거부되므로 `frames` 까지 쓴다. **동기화(Sync)는 누르지 않는다** — 사진마다 초안 없는 작업이 하나씩 더 생긴다.
4. 가져오기(Import)로 `review/tasks.json` 을 올린다.
5. 검수가 끝나면 내보내기(Export) → **JSON**. JSON-MIN 은 취소 여부 · 검수 시간이 빠져서 쓸 수 없다.

### 5. 학습 묶음 만들기 → Colab (레포 맨 위 폴더에서)

검수 결과로 R2 묶음을, 같은 사진의 R1 자동 라벨로 대조군(`r2auto`) 묶음을 만든다. 모델을 안 쓰므로 팀 환경에서 돈다.
찍힌 기록은 `prep_report.txt` 에 모아 zip 에 같이 넣는다(Colab 보고서 `R2_report.txt` 에 그대로 들어간다).

```bash
python tools/auto_label.py import --export <내보낸 JSON> --out data/datasets/detector/review/labels | tee data/datasets/detector/prep_report.txt
python tools/auto_label.py import --predictions data/datasets/detector/auto/tasks.json --like data/datasets/detector/review/labels \
    --out data/datasets/detector/review/auto_labels | tee -a data/datasets/detector/prep_report.txt
python tools/auto_label.py build --labels data/datasets/detector/review/labels --val d05 \
    --out data/datasets/detector/yolo_r2 | tee -a data/datasets/detector/prep_report.txt
python tools/auto_label.py build --labels data/datasets/detector/review/auto_labels --val-labels data/datasets/detector/review/labels \
    --val d05 --out data/datasets/detector/yolo_r2auto | tee -a data/datasets/detector/prep_report.txt
cd data/datasets/detector && zip -qr JARVISEO_R2_data.zip yolo_r2 yolo_r2auto prep_report.txt
```

- `import --export`: 취소(Skip)하지 않은 마지막 제출만 쓴다. 제출하지 않은 사진은 라벨 파일을 만들지 않아 `build` 에서 빠진다
  (빈 파일로 두면 "물건 없음"으로 학습된다). 10종 밖 이름이 있으면 멈춘다. 검수 시간(`lead_time`) 합계 · 사진당 중앙값 · 영상별을 찍는다.
- `import --predictions --like`: 대조군 라벨. 사람이 검수한 사진과 **같은 사진만**, R1 자동 라벨 그대로(`remote` 는 빼고 센다).
  짧은 변 40px 미만은 사람 규칙과 같게 양쪽 다 뺀다.
- `build --val-labels`: 검증 사진의 라벨만 다른 폴더에서 읽는다. 대조군도 best.pt 를 사람 라벨 d05 로 고르게 해서, 두 학습의 차이를 학습 라벨 하나로 남긴다.
- zip 을 공용 Drive `JARVISEO/` 에 올리고 `notebooks/detector_finetune.ipynb` 를 Colab 에서 연다(T4 · 고용량 RAM → 모두 실행).
  결과 가중치는 `data/models/detector_r2.pt` · `detector_r2auto.pt` 로 받아 `run_ablation.py --weights` 로 R0 · R1 과 한 실행에서 잰다.

### 검수 규칙

| 그린다 | 예 |
|---|---|
| bottle | 페트병 · 소스병 · 텀블러 (병 모양이면 내용물 상관없이) |
| cup | 머그 · 종이컵 · 유리컵 |
| bowl | 그릇 · **햇반 용기** |
| product | 캔(참치 · 감자칩 통 · 음료) · 과자 봉지 · 컵라면 · 병조림 · 파우치 · 상자 포장 식품 |
| tv | 모니터 · TV |
| laptop · mouse · keyboard · cell phone · book | 이름 그대로 |

- **안 그린다**: 10종 밖 물건(안경 · 안경 케이스 · 시계 · 리모컨 · 사람 · 손 · 의자 · 칫솔 등), 짧은 변 40px 미만(1280×720 기준),
  절반 넘게 가려진 것, 애매한 것.
- **하나씩 그린다**: 쌓여 있어도 하나하나 구분되면 하나씩. "묶음 전체를 박스 하나로"는 가리킴 정답표에만 쓰는 규칙이다.
- **물건이 하나도 없는 사진도 제출(Submit)한다.** 빈 제출이 곧 "물건 없음" 라벨이다. 건너뛰기(Skip)는 쓰지 않는다 — 검수 안 한 사진으로 빠진다.
- 초안 박스는 지우거나 · 고치거나 · 이름을 바꾸고, 빠진 물건은 새로 그린다. 사진 위에 "remote … 뺐습니다"가 보이면 그 자리 물건을 다시 본다.
- 어두운 d04(밤 공원)는 밝기 · 대비 조절로 본다. 편의점 d06 은 한 장에 상품이 수십 개라 장당 2~3분 걸릴 수 있다.
- 검수 시간은 정해 두고 끊는다(R2 는 총 3시간). 다 못 봐도 제출한 사진만 학습에 쓴다.
