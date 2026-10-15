"""VLM 에 사진 한 장과 질문을 보내 답과 왕복 시간을 찍는다.  담당: 문태현

OpenAI 키가 필요하다. 사진 경로를 안 주면 웹캠으로 한 장 찍는다.

실행:
    python scripts/check_vlm.py "이거 뭐야?"
    python scripts/check_vlm.py "이거 뭐야?" --image 사진.jpg
    python scripts/check_vlm.py "아까 본 가방?" --memory "질문: 이거 뭐야? / 대답: 백팩이에요."
    python scripts/check_vlm.py "아까 본 가방 어디 있었지?" --memory    # 찾은 기록이 없을 때
"""

import argparse
import time

import cv2

from jarviseo import config
from jarviseo.capture import WebcamSource
from jarviseo.types import Frame, MemoryHit
from jarviseo.vlm import VLMClient

WARMUP_FRAMES = 15  # 웹캠은 켜자마자 몇 장이 어둡다(자동 노출)


def snapshot() -> Frame:
    with WebcamSource(config.CAMERA_INDEX) as camera:
        frames = [camera.read() for _ in range(WARMUP_FRAMES)]
    picked = frames[-1]
    if picked is None:
        raise SystemExit("웹캠에서 사진을 못 읽었다")
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM 한 번 묻기")
    parser.add_argument("question", help="질문")
    parser.add_argument("--image", help="사진 경로. 없으면 웹캠으로 찍는다")
    parser.add_argument("--memory", nargs="*", help="기억 검색 결과로 넘길 대화 기록 (10분 전)")
    args = parser.parse_args()

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"사진을 못 읽었다: {args.image}")
        frame = Frame(image=image, timestamp=time.monotonic(), source_id=args.image)
    else:
        frame = snapshot()
    memories = None
    if args.memory is not None:
        ago = time.time() - 600
        memories = [
            MemoryHit(memory_id=str(i), kind="observation", text=text, score=1.0, observed_at=ago)
            for i, text in enumerate(args.memory)
        ]

    vlm = VLMClient()
    print(f"모델 {vlm.model}, 사진 {frame.size}, 질문 {args.question!r}")
    answer = vlm.ask(frame, args.question, memories)
    if answer is None:
        print("답 없음 (위 경고 참고)")
        return
    print(f"답 {answer!r}")
    print(f"왕복 {vlm.last_latency_ms:.0f}ms")


if __name__ == "__main__":
    main()
