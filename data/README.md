# data — 학습 데이터 · 가중치 · 영상

**Git 에 올리지 않는다.** 공용 Drive 에서 받아서 여기에 둔다.
`.gitkeep` 만 커밋되어 있고 내용물은 `.gitignore` 로 제외된다.

| 폴더 | 무엇이 들어가나 |
|---|---|
| `datasets/` | YOLO 학습·평가용 이미지와 라벨 |
| `models/` | 학습된 가중치 (`.pt`, `.onnx`) |
| `videos/` | 촬영한 원본 영상 |

## 알아둘 것

- 카메라 없이 개발할 때 `datasets/sample/` 을 프레임 입력원으로 쓴다.
  (`.env` 의 `JARVISEO_FRAME_SOURCE=folder`)
- 용량이 큰 파일을 실수로 커밋하지 않도록 `git status` 를 꼭 확인한다.
