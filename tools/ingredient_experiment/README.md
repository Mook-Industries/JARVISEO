# 원재료 영역 결합 방식 로컬 실험

이 도구는 기존 OCR 규칙과 YOLO 후보를 결합합니다. 서비스 기본 동작을 변경하지 않습니다.

JARVISEO 폴더에서 PowerShell로 실행:

```powershell
.\.venv\Scripts\python.exe tools\ingredient_experiment\predict_hybrid.py data\datasets\ocr_eval\label_20.png --weights data\models\ingredient_panel_r2_experimental.pt --output data\datasets\ingredient_experiment\local_prediction
```

입력 이미지 경로를 다른 로컬 파일로 바꿀 수 있습니다. 결과는 crop.png, overlay.png, result.json입니다. GPU나 외부 API를 쓰지 않습니다. 설치된 EasyOCR 모델을 사용하며 자동 다운로드는 비활성화되어 있습니다.

고정 설정은 hybrid_config.json입니다. 다른 컴퓨터에서는 --repo로 JARVISEO 저장소 경로를 지정하세요.

측정 결과와 한계는 data/datasets/ingredient_experiment/HYBRID_REVIEW.md에 있습니다. 실서비스 또는 GitHub에 배포하기 전 사용자 승인이 필요합니다.

## 함유 표시 보존과 팀 검토

- 원재료명뿐 아니라 `우유·대두 등 ○○ 함유` 표시까지 보존합니다.
- 별도 함유 표시를 읽으면 선택 영역을 확장합니다. 읽지 못하면 기존 영역으로 돌아갑니다.
- 기존 R2 가중치는 함유 표시를 제외한 정답으로 학습했습니다. 이 도구의 보완 로직 없이 단독으로 사용하지 마세요.
- 이전 원재료 전용 IoU/CER 수치는 이 변경의 성능 수치가 아닙니다. 새 기준 성능은 미확정입니다.
- 변경은 브랜치와 PR에서 검토하며, 승인 전 머지·릴리스를 하지 않습니다. 데이터와 가중치는 Git 추적에서 제외합니다.
