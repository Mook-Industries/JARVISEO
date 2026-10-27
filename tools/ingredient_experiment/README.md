# 원재료 영역 결합 방식 로컬 실험

이 도구는 기존 OCR 규칙과 YOLO 후보를 결합합니다. 서비스 기본 동작을 변경하지 않습니다.

JARVISEO 폴더에서 PowerShell로 실행:

```powershell
.\.venv\Scripts\python.exe tools\ingredient_experiment\predict_hybrid.py data\datasets\ocr_eval\label_20.png --weights data\models\ingredient_panel_r2_experimental.pt --output data\datasets\ingredient_experiment\local_prediction
```

입력 이미지 경로를 다른 로컬 파일로 바꿀 수 있습니다. 결과는 crop.png, overlay.png, result.json입니다. GPU나 외부 API를 쓰지 않습니다. 설치된 EasyOCR 모델을 사용하며 자동 다운로드는 비활성화되어 있습니다.

고정 설정은 hybrid_config.json입니다. 다른 컴퓨터에서는 --repo로 JARVISEO 저장소 경로를 지정하세요.

측정 결과와 한계는 data/datasets/ingredient_experiment/HYBRID_REVIEW.md에 있습니다. 실서비스 또는 GitHub에 배포하기 전 사용자 승인이 필요합니다.
