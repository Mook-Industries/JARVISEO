"""스파이크 3 — 보내는 사진 크기가 VLM 왕복 시간을 얼마나 바꾸나.  담당: 최홍묵

같은 사진을 긴 변 1568 / 1024 / 768px 로 줄여 보내고, 크기별 왕복 중앙값 · p95 와
보낸 크기를 찍는다.
1568 은 지금 전체 사진이 실제로 나가는 크기다(vlm/client.py 의 encode_image 가 그 이상은 줄인다).
대상 크롭(#88)의 긴 변 768px 상한이 정말 시간을 줄이는지, 작게 보내도 답이 괜찮은지가
이 결과로 정해진다.
결과는 docs/experiments.md 에 남긴다.

키 확인 · 답 한 번 받아 보기는 scripts/check_vlm.py 가 한다.
실제 API 를 부른다. 호출 수 = 사진 수 × 3 크기 × 반복 횟수.

실행 (레포 맨 위 폴더에서):
    python scripts/spike/spike_03_vlm.py 사진1.jpg 사진2.jpg --repeat 3
    python scripts/spike/spike_03_vlm.py                    # 사진을 안 주면 웹캠으로 한 장
"""

import argparse
import statistics
import time

import cv2
import numpy as np

from jarviseo import config
from jarviseo.capture import WebcamSource
from jarviseo.types import Frame
from jarviseo.vlm import VLMClient
from jarviseo.vlm.client import encode_image

SIZES = (1568, 1024, 768)
WARMUP_FRAMES = 15  # 웹캠은 켜자마자 몇 장이 어둡다(자동 노출)


def snapshot() -> np.ndarray:
    with WebcamSource(config.CAMERA_INDEX) as camera:
        frames = [camera.read() for _ in range(WARMUP_FRAMES)]
    if frames[-1] is None:
        raise SystemExit("웹캠에서 사진을 못 읽었다")
    return frames[-1].image


def shrink(image: np.ndarray, long_side: int) -> np.ndarray:
    """긴 변을 long_side 로 줄인다. 이미 작으면 그대로 둔다."""
    height, width = image.shape[:2]
    scale = long_side / max(height, width)
    if scale >= 1:
        return image
    size = (round(width * scale), round(height * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def p95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM 사진 크기별 왕복 시간")
    parser.add_argument("images", nargs="*", help="사진 경로. 없으면 웹캠으로 한 장 찍는다")
    parser.add_argument("--repeat", type=int, default=3, help="사진 · 크기마다 몇 번 보낼지")
    parser.add_argument("--question", default="이거 뭐야?")
    args = parser.parse_args()

    photos = {}
    for path in args.images:
        image = cv2.imread(path)
        if image is None:
            raise SystemExit(f"사진을 못 읽었다: {path}")
        photos[path] = image
    if not photos:
        photos["webcam"] = snapshot()

    vlm = VLMClient()
    calls = len(photos) * len(SIZES) * args.repeat
    print(f"모델 {vlm.model}, 사진 {len(photos)}장, 반복 {args.repeat}, 호출 {calls}번")
    latency: dict[int, list[float]] = {size: [] for size in SIZES}
    sent_kb: dict[int, list[float]] = {size: [] for size in SIZES}
    failed = dict.fromkeys(SIZES, 0)
    done = 0
    for r in range(args.repeat):
        # 늘 같은 크기를 먼저 보내면 첫 요청의 연결 비용이 그 크기에만 붙는다. 순서를 돌린다.
        order = SIZES[r % len(SIZES) :] + SIZES[: r % len(SIZES)]
        for name, image in photos.items():
            for size in order:
                small = shrink(image, size)
                frame = Frame(image=small, timestamp=time.monotonic(), source_id=name)
                answer = vlm.ask(frame, args.question)
                done += 1
                if answer is None or vlm.last_latency_ms is None:
                    failed[size] += 1
                    print(f"  [{done}/{calls}] {name} {size}px 실패")
                    continue
                latency[size].append(vlm.last_latency_ms)
                sent_kb[size].append(len(encode_image(small)) / 1024)
                # 답 문장은 첫 반복만. 나머지는 진행만 찍어 멈춘 듯 보이지 않게
                shown = f" {answer!r}" if r == 0 else ""
                print(f"  [{done}/{calls}] {name} {size}px {vlm.last_latency_ms:.0f}ms{shown}")

    print("\n크기 | 성공 | 중앙값 ms | p95 ms | 보낸 크기 KB(중앙값) | 실패")
    for size in SIZES:
        values = latency[size]
        if not values:
            print(f"{size} | 0 | - | - | - | {failed[size]}")
            continue
        median_kb = statistics.median(sent_kb[size])
        print(
            f"{size} | {len(values)} | {statistics.median(values):.0f} | {p95(values):.0f}"
            f" | {median_kb:.0f} | {failed[size]}"
        )


if __name__ == "__main__":
    main()
