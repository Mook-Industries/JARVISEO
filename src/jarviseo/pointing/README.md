# pointing — ① 지시 대상 특정 · 담당 최홍묵

**이 프로젝트의 핵심 기여.** 시야에 물건이 여러 개일 때 "저거"가 무엇인지 정한다.
착용형 기기에는 화면을 터치해 대상을 고를 방법이 없기 때문에 필요하다.

| 파일 | 하는 일 | 책임 지표 |
|---|---|---|
| `detector.py` | 손끝·물체 검출 (YOLOv8n 파인튜닝) | fingertip mAP@0.5 |
| `cues.py` | 후보별 단서 점수 (v1~v5) | — |
| `resolver.py` | 단서 융합 → 1위 결정 · 되묻기 판단 | 대상 선택 정확도 |

## 알아둘 것

- `cues.py` 가 **ablation 실험의 축**이다. 단서를 하나씩 켜고 끄며 기여도를 잰다.
- `resolver.py` 의 출력이 VLM 에 넘길 **크롭**을 결정한다. 여기가 틀리면 답도 틀린다.
- 1위와 2위 점수차(`margin`)가 임계값 미만이면 답하지 않고 되묻는다.
- 실험 기록은 `docs/experiments.md`, 결과 적재는 `memory/models.py` 의
  `Detection` · `TargetResolution`.
