# tools — 개발용 도구 (장치 점검 · 가리킴 평가셋)

개발용 도구. 서비스 코드가 아니라서 `src/` 와 분리해 둔다.

| 이름 | 무엇 |
|---|---|
| `device_check/` | 카메라·마이크가 제대로 물렸는지 확인하는 5종 |
| `capture_scenes.py` | 가리킴 평가 장면 촬영 — 안경 카메라로 찍고, 그때 한 말을 `manifest.csv` 에 같이 남긴다 |
| `label_targets.py` | 가리킴 정답 표시 — 찍은 사진에 YOLO 후보를 번호로 띄우고, 정답을 골라(없으면 박스를 그려) `labels.csv` 에 남긴다 |

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
  2. 가리킨 물건에 초록 상자가 있으면 그 번호(`1`~`9`) → 끝
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
