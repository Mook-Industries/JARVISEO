"""
카메라 최소 확인 — 그림이 나오는지, 해상도·FPS 가 얼마로 잡히는지만 본다.
YOLO 없이 도는 가장 가벼운 검사라, 화면이 안 나올 때 원인이
카메라인지 모델인지 가르는 용도다.

실행:
    python preview.py
    q : 종료
"""

import sys

import cv2

CAM = 0  # ← find_camera.py 로 찾은 번호로 바꾸세요


def pick_backend():
    """OS에 맞는 카메라 백엔드를 고른다."""
    if sys.platform == "darwin":  # macOS
        return cv2.CAP_AVFOUNDATION
    if sys.platform == "win32":  # Windows
        return cv2.CAP_DSHOW
    return cv2.CAP_V4L2  # Linux


cap = cv2.VideoCapture(CAM, pick_backend())
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print(
    "실제 적용된 값:",
    cap.get(cv2.CAP_PROP_FRAME_WIDTH),
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
    cap.get(cv2.CAP_PROP_FPS),
)

while True:
    ok, frame = cap.read()
    if not ok:
        print("프레임 읽기 실패")
        break
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # q 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
