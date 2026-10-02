"""계약 파일이 깨지지 않았는지 확인한다.

types.py 는 세 사람이 전부 의존하는 파일이다. 필드 이름 하나가 바뀌면
남의 코드가 조용히 깨진다. 그래서 최소한의 검사를 테스트로 박아둔다.

실행: pytest
"""

import numpy as np

from jarviseo.types import (
    AllergenVerdict,
    BBox,
    CueKind,
    Detection,
    Frame,
    Intent,
)


def test_bbox_기본_계산():
    box = BBox(10, 20, 110, 220)
    assert box.width == 100
    assert box.height == 200
    assert box.area == 20000
    assert box.center == (60, 120)


def test_bbox_겹침():
    a = BBox(0, 0, 10, 10)
    b = BBox(5, 5, 15, 15)
    assert 0 < a.iou(b) < 1
    assert a.iou(a) == 1.0
    assert a.iou(BBox(100, 100, 110, 110)) == 0.0


def test_frame_크기는_width_height_순서():
    # OpenCV 의 shape 는 (H, W, C) 지만 size 는 (W, H) 로 뒤집어 돌려준다.
    # 이 순서를 헷갈리면 좌표 계산이 통째로 틀어진다.
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(image=image, timestamp=1.0, source_id="test")
    assert frame.size == (640, 480)


def test_detection_필드_이름():
    det = Detection(label="fingertip", confidence=0.9, bbox=BBox(0, 0, 1, 1))
    assert det.label == "fingertip"
    assert det.track_id is None


def test_알레르기_판정에_uncertain이_있다():
    # 애매할 때 SAFE 로 떨어뜨리면 안 된다. 재현율 우선 원칙.
    assert AllergenVerdict.UNCERTAIN.value == "uncertain"


def test_의도와_단서_목록():
    assert Intent.GENERAL in set(Intent)
    # ERD turn_candidate.cue_scores 의 키와 같아야 한다
    assert {c.value for c in CueKind} == {"center", "point", "gaze", "lang", "ctx"}
