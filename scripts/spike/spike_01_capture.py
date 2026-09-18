"""스파이크 1 — 프레임 캡처와 선명도 측정.

확인할 것: 프레임이 들어오는가, 흔들었을 때 선명도 숫자가 떨어지는가.

실행:
    python scripts/spike/spike_01_capture.py

카메라가 없으면 입력원을 바꿔서 돌린다:
    python scripts/spike/spike_01_capture.py data/videos/sample.mp4
"""

import sys
from pathlib import Path

from jarviseo.capture import FrameBuffer, VideoFileSource, WebcamSource


def main() -> None:
    if len(sys.argv) > 1:
        source = VideoFileSource(Path(sys.argv[1]))
    else:
        source = WebcamSource(index=0)

    buffer = FrameBuffer(maxlen=30)

    with source:
        for i, frame in enumerate(source.stream()):
            buffer.push(frame)
            print(f"[{i:3d}] {frame.size} t={frame.timestamp:6.2f}s 선명도={frame.sharpness:8.1f}")
            if i >= 60:
                break

    best = buffer.pick_best()
    print(f"\n버퍼 {len(buffer)}장 중 가장 선명한 프레임: {best.sharpness:.1f}")
    print("고개를 흔들면서 다시 돌려보고, 숫자가 떨어지는지 확인할 것.")


if __name__ == "__main__":
    main()
