"""가리킴 길(detect → resolve ⇄ clarify → crop_target)과 재촬영이 그래프에서 맞게 도는지.

여기서 보는 것은 "후보가 오면 그래프가 맞는 길로 가는가"다. 검출 정확도·대상 고르기 계산은
pointing 쪽 테스트가 본다. 그래서 검출기·대상 결정은 이 파일 안의 작은 가짜로 바꿔 끼운다.
CI 에는 YOLO(ultralytics)가 없어서 실제 검출기는 여기서 쓸 수 없다.
"""

import time

import cv2
import numpy as np
import pytest

from jarviseo.capture import FrameBuffer, ImageFolderSource
from jarviseo.memory import TurnCandidate, TurnInference
from jarviseo.pointing import resolve_target
from jarviseo.types import BBox, Detection, Frame, Intent, TargetCandidate, TargetResolution
from jarviseo.vlm import FakeVLMClient
from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech
from jarviseo.workflow import GraphDeps, build_graph
from jarviseo.workflow.graph import RECURSION_LIMIT
from jarviseo.workflow.messages import CLARIFY_GIVE_UP, RECAPTURE_GIVE_UP

FULL = (1080, 1920, 3)
CROP = (400, 300, 3)  # 아래 왼쪽 컵 박스 (400,500)-(700,900)
QUESTION = "왼쪽 컵이요, 오른쪽 컵이요?"
TWO_CUPS = [
    Detection("cup", 0.9, BBox(400, 500, 700, 900)),
    Detection("cup", 0.9, BBox(1200, 500, 1500, 900)),
]


class _Detector:
    """정해 둔 후보를 돌려준다. ObjectDetector.detect(image) 와 모양이 같다."""

    def __init__(self, detections=TWO_CUPS) -> None:
        self.detections = detections

    def detect(self, image: np.ndarray) -> list[Detection]:
        return list(self.detections)


class _Resolver:
    """정해 둔 1·2위 점수 차로 고른다. resolve_target 과 같은 키워드로 불린다.

    margins 를 부를 때마다 하나씩 꺼내 쓰고, 다 쓰면 마지막 값을 계속 쓴다.
    """

    def __init__(self, margins=(0.5,)) -> None:
        self.margins = list(margins)
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, candidates, text, frame_size, replies=(), **kwargs) -> TargetResolution:
        self.calls.append((text, list(replies)))
        margin = self.margins.pop(0) if len(self.margins) > 1 else self.margins[0]
        if not candidates:
            return TargetResolution([], chosen=None, margin=0.0, needs_clarify=True)
        ranked = [TargetCandidate(d, total_score=0.5) for d in candidates]
        if margin < 0.15:
            return TargetResolution(ranked, None, margin, True, QUESTION)
        return TargetResolution(ranked, ranked[0], margin, False)


def _sharp() -> np.ndarray:
    # 체커보드는 경계가 많아 선명도(Laplacian 분산)가 기준(100)보다 훨씬 높다.
    tile = np.indices(FULL[:2]).sum(axis=0) // 40 % 2
    return np.repeat((tile * 255).astype(np.uint8)[:, :, None], 3, axis=2)


def _blurry() -> np.ndarray:
    return np.full(FULL, 128, np.uint8)  # 한 가지 색이라 선명도 0


def _buffer(*images: np.ndarray) -> FrameBuffer:
    frames = FrameBuffer()
    for image in images:
        frames.push(Frame(image=image, timestamp=time.monotonic(), source_id="test"))
    return frames


def _deps(texts, frames=None, margins=(0.5,), detections=TWO_CUPS, **kwargs) -> GraphDeps:
    return GraphDeps(
        stt=FakeSpeechToText(texts),
        tts=FakeTextToSpeech(),
        vlm=FakeVLMClient(["컵이에요."]),
        frames=frames if frames is not None else _buffer(_sharp()),
        detector=_Detector(detections),
        resolve=_Resolver(margins),
        **kwargs,
    )


# --- 가리킴 길 ---------------------------------------------------------------


def test_pointing_question_sends_chosen_crop_to_vlm():
    """가리키는 질문은 고른 대상의 크롭을 VLM 에 보낸다."""
    deps = _deps(["자비서, 저거 뭐야?"])

    result = build_graph(deps).invoke({})

    assert result["intent"] is Intent.POINTING
    ((sent, question, _),) = deps.vlm.asked
    # 프로젝트 핵심 결정: VLM 에는 전체 프레임이 아니라 고른 대상의 크롭이 간다.
    assert sent.image.shape == CROP
    assert question == "저거 뭐야?"
    assert result["response"].used_frame.image.shape == FULL  # 기록에는 원본 장면
    assert result["response"].target is result["target"]
    assert deps.tts.spoken == ["컵이에요."]
    assert {"route", "detect", "resolve", "crop"} <= set(result["latency_ms"])


def test_no_detector_answers_with_whole_photo():
    """검출기가 없으면 가리키는 질문도 전체 사진으로 답한다."""
    vlm = FakeVLMClient(["컵이에요."])
    deps = GraphDeps(
        stt=FakeSpeechToText(["저거 뭐야?"]),
        tts=FakeTextToSpeech(),
        vlm=vlm,
        frames=_buffer(_sharp()),
    )

    result = build_graph(deps).invoke({})

    assert result["intent"] is Intent.GENERAL
    assert vlm.asked[0][0].image.shape == FULL


def test_no_frame_buffer_answers_with_text_only():
    """사진 버퍼가 없으면 가리키는 질문도 글자만으로 답한다."""
    vlm = FakeVLMClient(["무엇을 말씀하시는지 안 보여요."])
    deps = GraphDeps(
        stt=FakeSpeechToText(["저거 뭐야?"]),
        tts=FakeTextToSpeech(),
        vlm=vlm,
        detector=_Detector(),
        resolve=_Resolver(),
    )

    result = build_graph(deps).invoke({})

    assert result["intent"] is Intent.GENERAL
    assert vlm.asked[0][0] is None


def test_runs_end_to_end_with_image_folder(tmp_path):
    """이미지 폴더로 넣은 사진으로도 끝까지 돈다."""
    for i in range(2):
        ok, png = cv2.imencode(".png", _sharp())
        assert ok
        png.tofile(tmp_path / f"{i:03d}.png")
    frames = FrameBuffer()
    with ImageFolderSource(tmp_path) as source:
        for frame in source.stream():
            frames.push(frame)
    deps = _deps(["자비서, 이게 뭐야?"], frames=frames)

    build_graph(deps).invoke({})

    assert deps.vlm.asked[0][0].image.shape == CROP
    assert deps.tts.spoken == ["컵이에요."]


# --- 되묻기 ------------------------------------------------------------------


def test_ambiguous_asks_and_resolves_with_reply():
    """애매하면 되묻고 대답으로 다시 고른다."""
    deps = _deps(["저거 뭐야?", "왼쪽 거"], margins=[0.05, 0.5])

    result = build_graph(deps).invoke({})

    assert deps.tts.spoken == [QUESTION, "컵이에요."]
    # 다시 고를 때 처음 질문은 그대로, 대답은 따로 넘어간다(대답은 단서가 아니라 후보 좁히기)
    assert deps.resolve.calls == [("저거 뭐야?", []), ("저거 뭐야?", ["왼쪽 거"])]
    assert result["clarify_replies"] == ["왼쪽 거"]
    assert result["clarify_count"] == 1
    assert deps.vlm.asked[0][1] == "저거 뭐야?"  # VLM 질문은 처음 질문 그대로


class _HintSTT(FakeSpeechToText):
    """listen() 에 받은 힌트를 기록한다."""

    def __init__(self, texts) -> None:
        super().__init__(texts)
        self.hints: list[tuple[str, ...]] = []

    def listen(self, *args, hints=(), **kwargs):
        self.hints.append(tuple(hints))
        return super().listen(*args, hints=hints, **kwargs)


def test_clarify_hints_include_candidate_names():
    """되묻기 대답을 받아쓸 때 위치 말과 장면 후보의 한글 이름을 중복 없이 힌트로 준다."""
    deps = _deps(["저거 뭐야?", "컵"], margins=[0.05, 0.5])
    deps.stt = _HintSTT(["저거 뭐야?", "컵"])

    build_graph(deps).invoke({})

    assert deps.stt.hints[-1] == ("왼쪽", "오른쪽", "가운데", "컵")


def test_real_resolver_asks_and_crops_the_answered_side():
    """실제 resolve_target 으로: 컵 둘이 비슷하면 되묻고, '왼쪽 거'면 왼쪽 컵의 크롭을 보낸다."""
    deps = _deps(["저거 뭐야?", "왼쪽 거"])
    deps.resolve = resolve_target

    result = build_graph(deps).invoke({})

    assert deps.tts.spoken == ["왼쪽 컵이요, 오른쪽 컵이요?", "컵이에요."]
    assert result["target"].chosen.detection is TWO_CUPS[0]
    assert deps.vlm.asked[0][0].image.shape == CROP


def test_accepts_one_letter_reply():
    """한 글자 대답도 받는다."""
    # 질문은 두 글자 미만이면 버리지만, 되묻기 대답 "컵"은 받아야 한다(listen min_letters=1)
    deps = _deps(["저거 뭐야?", "컵"], margins=[0.05, 0.5])

    result = build_graph(deps).invoke({})

    assert result["clarify_replies"] == ["컵"]
    assert deps.tts.spoken == [QUESTION, "컵이에요."]


def test_no_reply_asks_to_point_again():
    """되묻기에 대답이 없으면 다시 가리켜 달라고 한다."""
    deps = _deps(["저거 뭐야?"], margins=[0.05])

    build_graph(deps).invoke({})

    assert deps.tts.spoken == [QUESTION, CLARIFY_GIVE_UP]
    assert deps.vlm.asked == []


def test_clarifies_at_most_twice():
    """되묻기는 두 번까지만 한다."""
    deps = _deps(["저거 뭐야?", "왼쪽 거", "왼쪽 거", "왼쪽 거"], margins=[0.05])

    result = build_graph(deps).invoke({})

    assert deps.tts.spoken == [QUESTION, QUESTION, CLARIFY_GIVE_UP]
    assert result["clarify_count"] == 2
    assert len(deps.resolve.calls) == 3


def test_no_candidates_gives_notice_without_asking():
    """후보가 없으면 되묻지 않고 안내한다."""
    deps = _deps(["저거 뭐야?"], detections=[])

    build_graph(deps).invoke({})

    assert deps.tts.spoken == [CLARIFY_GIVE_UP]


# --- 재촬영 ------------------------------------------------------------------


def test_no_sharpness_threshold_keeps_blurry_frame():
    """선명도 기준을 안 주면 흐려도 다시 고르지 않는다."""
    # 안경 카메라로 기준을 재기 전까지는 꺼 둔다(GraphDeps.sharpness_threshold 기본값 None)
    deps = _deps(["저거 뭐야?"], frames=_buffer(_blurry()))

    result = build_graph(deps).invoke({})

    assert "recapture_count" not in result
    assert deps.tts.spoken == ["컵이에요."]


def test_only_blurry_frames_recapture_twice_then_notice():
    """흐린 사진뿐이면 두 번 다시 고르고 안내한다."""
    deps = _deps(["저거 뭐야?"], frames=_buffer(_blurry()), sharpness_threshold=100.0)

    result = build_graph(deps).invoke({})

    assert result["recapture_count"] == 2
    assert deps.tts.spoken == [RECAPTURE_GIVE_UP]
    assert deps.vlm.asked == []


def test_sharp_frame_after_blurry_continues():
    """흐린 사진 뒤에 선명한 사진이 들어오면 이어서 답한다."""
    # 듣기에서 고른 사진은 흐리고, 다시 고르기 전에 카메라 스레드가 선명한 사진을 넣은 경우
    frames = _buffer(_blurry())
    pick = frames.pick_best

    def pick_then_capture(around=None, window=0.5):
        frame = pick(around=around, window=window)
        frames.push(Frame(image=_sharp(), timestamp=time.monotonic(), source_id="test"))
        return frame

    frames.pick_best = pick_then_capture
    deps = _deps(["저거 뭐야?"], frames=frames, sharpness_threshold=100.0)

    result = build_graph(deps).invoke({})

    assert result["recapture_count"] == 1
    assert deps.tts.spoken == ["컵이에요."]


def test_empty_buffer_recaptures_even_without_threshold():
    """버퍼가 비어 있으면 기준이 없어도 다시 고르고 안내한다."""
    deps = _deps(["저거 뭐야?"], frames=FrameBuffer())

    result = build_graph(deps).invoke({})

    assert result["recapture_count"] == 2
    assert deps.tts.spoken == [RECAPTURE_GIVE_UP]


# --- 기록 · 안전장치 ---------------------------------------------------------


def test_pointing_result_is_logged(store):
    """가리킴 결과가 기록에 남는다."""
    user_id = store.ensure_user()
    turn = {"session_id": store.start_session(user_id), "user_id": user_id}
    deps = _deps(["저거 뭐야?"], store=store)

    result = build_graph(deps).invoke(turn)

    with store.session() as db:
        inference = db.get(TurnInference, result["turn_id"])
        assert (inference.target_label, inference.detected_count) == ("cup", 2)
        assert inference.resolved_by == "MODEL"
        rows = db.query(TurnCandidate).filter_by(turn_id=result["turn_id"]).all()
        assert sorted(r.is_chosen for r in rows) == [False, True]


def test_gave_up_clarify_turn_has_no_resolver(store):
    """되묻다 끝난 턴은 고른 사람이 비어 있다."""
    # MODEL 로 남으면 실패 분석 때 "모델이 고른 턴"에 아무도 안 고른 턴이 섞인다
    user_id = store.ensure_user()
    turn = {"session_id": store.start_session(user_id), "user_id": user_id}
    deps = _deps(["저거 뭐야?"], margins=[0.05], store=store)

    result = build_graph(deps).invoke(turn)

    with store.session() as db:
        inference = db.get(TurnInference, result["turn_id"])
        assert inference.resolved_by is None
        assert inference.target_label is None


def test_recursion_limit_is_set():
    """재귀 한도가 걸려 있다."""
    assert build_graph(_deps([])).config["recursion_limit"] == RECURSION_LIMIT


@pytest.mark.parametrize("texts", [[], ["음."]])
def test_no_question_skips_pointing_path(texts):
    """질문이 없으면 가리킴 길도 타지 않는다."""
    deps = _deps(texts)

    build_graph(deps).invoke({})

    assert deps.resolve.calls == []
    assert deps.tts.spoken == []
