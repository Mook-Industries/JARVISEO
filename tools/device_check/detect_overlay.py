"""
JARVISEO - 01 비전 모니터 (실시간 객체 검출 오버레이)

실행 전 준비:
    pip install ultralytics opencv-python

실행:
    python detect_overlay.py

조작:
    q : 종료
    s : 현재 화면 저장 (shot_xxxx.jpg)
"""

import sys
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

# ─────────────────────────────────────────────
# 설정 — 여기만 바꾸면 됩니다
# ─────────────────────────────────────────────
CAM_INDEX = 0  # 카메라 번호. find_camera.py 로 찾은 값
CONF = 0.5  # 이 확신도 미만은 화면에 안 그림 (0.0 ~ 1.0)
WIDTH = 1280  # 요청 해상도 (카메라가 지원 안 하면 알아서 바뀜)
HEIGHT = 720
MODEL = "yolov8n.pt"  # 첫 실행 때 자동으로 내려받습니다 (약 6MB)
DEVICE = None  # 애플 실리콘이면 "mps" 로 바꾸면 빨라집니다

BOX_COLOR = (0, 220, 120)  # BGR — 초록
TEXT_COLOR = (0, 0, 0)

# 산출물 저장 위치.
# Path(__file__) 은 '이 파일 자신의 경로'라서, 어느 폴더에서 실행하든
# 항상 스크립트 옆 _captures/ 에 떨어진다.
OUT_DIR = Path(__file__).resolve().parent / "_captures"


def pick_backend():
    """OS에 맞는 카메라 백엔드를 고른다."""
    if sys.platform == "darwin":  # macOS
        return cv2.CAP_AVFOUNDATION
    if sys.platform == "win32":  # Windows
        return cv2.CAP_DSHOW
    return cv2.CAP_V4L2  # Linux


def draw_box(frame, x1, y1, x2, y2, label):
    """박스 하나와 그 위의 라벨을 그린다."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # 박스가 화면 맨 위에 붙으면 라벨이 잘리므로 아래로 밀어준다
    ty = max(y1, th + 10)
    cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 6, ty), BOX_COLOR, -1)
    cv2.putText(frame, label, (x1 + 3, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)


def main():
    print(f"모델 불러오는 중... ({MODEL})")
    model = YOLO(MODEL)

    cap = cv2.VideoCapture(CAM_INDEX, pick_backend())
    if not cap.isOpened():
        print(f"카메라 {CAM_INDEX}번을 열 수 없습니다.")
        print("→ find_camera.py 로 번호를 다시 확인하고, 카메라 권한도 확인하세요.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"카메라 {CAM_INDEX}번 열림 — 실제 해상도 {real_w}x{real_h}")
    print("q 종료 / s 저장")

    fps = 0.0
    prev = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임을 읽지 못했습니다. 케이블이 빠졌는지 확인하세요.")
                break

            # ── 추론 ──────────────────────────────
            # ultralytics 의 반환 타입 힌트가 실제와 달라(Iterator[Results | Tensor])
            # 타입 검사기가 .boxes 를 못 찾는다. 실행에는 문제가 없으므로 Any 로 받는다.
            results: Any = model.predict(
                frame, conf=CONF, verbose=False, **({"device": DEVICE} if DEVICE else {})
            )
            boxes = results[0].boxes if len(results) else None
            found = len(boxes) if boxes is not None else 0

            # ── 오버레이 ──────────────────────────
            for box in boxes if boxes is not None else []:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                name = model.names[int(box.cls[0])]
                score = float(box.conf[0])
                draw_box(frame, x1, y1, x2, y2, f"{name} {score:.2f}")

            # ── 상태 표시줄 ───────────────────────
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            cv2.putText(
                frame,
                f"FPS {fps:4.1f}   objects {found}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                BOX_COLOR,
                2,
            )

            cv2.imshow("JARVISEO - vision monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                OUT_DIR.mkdir(exist_ok=True)
                fname = str(OUT_DIR / f"shot_{int(now)}.jpg")
                cv2.imwrite(fname, frame)
                print(f"저장: {fname}")

    except KeyboardInterrupt:
        print("\n사용자 중단")

    finally:
        # 여기가 중요 — 카메라를 반드시 놓아줘야 다음 실행에서 안 막힌다
        cap.release()
        cv2.destroyAllWindows()
        print("종료 완료")


if __name__ == "__main__":
    main()
