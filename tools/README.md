# tools — 하드웨어 점검 스크립트

개발용 도구. 서비스 코드가 아니라서 `src/` 와 분리해 둔다.

| 폴더 | 무엇 |
|---|---|
| `device_check/` | 카메라·마이크가 제대로 물렸는지 확인하는 5종 |

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
