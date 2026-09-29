# capture — 프레임 입력

카메라에서 프레임을 받아오고, 그 프레임이 쓸 만한지 판단한다.
**카메라가 1대뿐**이라 그 한 대가 누구 손에 있든 나머지 사람의 개발이
멈추지 않도록 입력원을 인터페이스 뒤로 숨기는 것이 이 폴더의 존재 이유다.

| 파일 | 하는 일 |
|---|---|
| `frame_source.py` | 입력원 추상화 — `USBCamSource` / `ImageFolderSource` |
| `quality.py` | 선명도(Laplacian 분산) 측정 + 최근 프레임 버퍼 |

## 알아둘 것

- 카메라 없이 개발하려면 `.env` 에서 `JARVISEO_FRAME_SOURCE=folder` 로 바꾼다.
- 백엔드는 OS마다 다르다 (맥 AVFoundation / 윈도우 DSHOW). 직접 지정하지 말고
  `frame_source.py` 가 고르게 둔다.
- 흐린 프레임은 뒤 단계를 전부 오염시킨다. 여기서 거르는 게 가장 싸다.
