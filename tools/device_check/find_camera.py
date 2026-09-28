import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok:
            print(f"[{i}] 열림 — {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"[{i}] 열리긴 했는데 프레임을 못 읽음")
        cap.release()
    else:
        print(f"[{i}] 없음")