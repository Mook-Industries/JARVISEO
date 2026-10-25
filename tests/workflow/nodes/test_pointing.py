"""가리킴 길(detect → resolve ⇄ clarify → crop_target)과 재촬영이 그래프에서 맞게 도는지.

여기서 보는 것은 "후보가 오면 그래프가 맞는 길로 가는가"다. 검출 정확도·대상 고르기 계산은
pointing 쪽 테스트가 본다. 그래서 검출기·손 검출기·대상 결정은 이 파일 안의 작은 가짜로 바꿔 끼운다.
CI 에는 YOLO(ultralytics) · MediaPipe 가 없어서 실제 검출기는 여기서 쓸 수 없다.
"""

import time

import cv2
import numpy as np
import pytest

from jarviseo import config
from jarviseo.capture import FrameBuffer, ImageFolderSource
from jarviseo.memory import SessionTurn, TurnCandidate, TurnInference, TurnVoice
from jarviseo.pointing import resolve_target
from jarviseo.pointing.hands import HandPose
from jarviseo.types import BBox, Detection, Frame, Intent, TargetCandidate, TargetResolution
from jarviseo.vlm import FakeVLMClient
from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech
from jarviseo.workflow import GraphDeps, build_graph
from jarviseo.workflow.graph import RECURSION_LIMIT
from jarviseo.workflow.messages import CLARIFY_GIVE_UP, RECAPTURE_GIVE_UP
from jarviseo.workflow.nodes.answer import POINTING_CONTEXT

FULL = (1080, 1920, 3)
# 아래 왼쪽 컵 박스 (400,500)-(700,900) 300×400 에 사방 15% 여백. 긴 변 520 이라 VLM 에도 그대로
CROP = (520, 390, 3)
QUESTION = "왼쪽 컵이요, 오른쪽 컵이요?"
# 가리킨 질문이 VLM 에 갈 때의 모양(크롭이라는 맥락 + 검출기 추정 이름 + 원래 질문)
PROMPT = POINTING_CONTEXT.format(name="컵", question="저거 뭐야?")
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
    assert question == PROMPT  # 잘린 사진이라는 맥락을 붙여 보낸다
    assert result["response"].used_frame.image.shape == FULL  # 기록에는 원본 장면
    assert result["response"].target is result["target"]
    assert deps.tts.spoken == ["컵이에요."]
    assert {"route", "detect", "resolve", "crop"} <= set(result["latency_ms"])


def test_vlm_gets_crop_shrunk_to_768():
    """큰 물건의 크롭은 VLM 으로 보낼 때 긴 변 768px 로 줄인다. 기록에는 원본 장면이 남는다."""
    big = Detection("cup", 0.9, BBox(100, 100, 1800, 1000))  # 여백을 넣으면 사진 전체가 된다
    deps = _deps(["저거 뭐야?"], detections=[big])

    result = build_graph(deps).invoke({})

    assert deps.vlm.asked[0][0].image.shape == (432, 768, 3)
    assert result["crop"].shape == FULL  # 상태의 크롭(성분 판정이 받는 것)은 줄이지 않는다
    assert result["response"].used_frame.image.shape == FULL


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


@pytest.mark.parametrize("called_since", [None, lambda t: False])
def test_ambiguous_asks_and_resolves_with_reply(called_since):
    """애매하면 되묻고 대답으로 다시 고른다. 다시 부르지 않았으면 called_since 가 있어도 같다."""
    deps = _deps(["저거 뭐야?", "왼쪽 거"], margins=[0.05, 0.5], called_since=called_since)

    result = build_graph(deps).invoke({})

    assert deps.tts.spoken == [QUESTION, "컵이에요."]
    # 다시 고를 때 처음 질문은 그대로, 대답은 따로 넘어간다(대답은 단서가 아니라 후보 좁히기)
    assert deps.resolve.calls == [("저거 뭐야?", []), ("저거 뭐야?", ["왼쪽 거"])]
    assert result["clarify_replies"] == ["왼쪽 거"]
    assert result["clarify_count"] == 1
    assert deps.vlm.asked[0][1] == PROMPT  # 되묻기 대답은 빼고 처음 질문에 맥락만


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


def test_failed_clarify_question_gives_up_without_listening():
    """되묻는 말을 합성하지 못해 소리가 안 났으면 대답을 기다리지 않고 다시 가리켜 달라고 한다."""
    deps = _deps([], margins=[0.05, 0.5])
    deps.stt = _ReplySTT(["저거 뭐야?", "왼쪽 거"])
    deps.tts = FakeTextToSpeech(fail=[QUESTION])  # 서버가 느려 되묻는 말이 안 나갔다

    result = build_graph(deps).invoke({})

    assert deps.stt.calls == 1  # 질문만 듣고 대답(5초)은 기다리지 않았다
    assert deps.tts.failed == [QUESTION]
    assert deps.tts.spoken == [CLARIFY_GIVE_UP]
    assert result["clarify_replies"] == []
    assert not result.get("interrupted")  # 다시 부른 게 아니다. 다음 턴은 평소처럼 WAKEWORD
    assert deps.vlm.asked == []


def test_no_candidates_gives_notice_without_asking():
    """후보가 없으면 되묻지 않고 안내한다."""
    deps = _deps(["저거 뭐야?"], detections=[])

    build_graph(deps).invoke({})

    assert deps.tts.spoken == [CLARIFY_GIVE_UP]


def test_clarify_plays_start_chime_before_reply():
    """되묻는 말을 다 읽은 뒤, 대답을 듣기 전에 시작음을 낸다."""
    deps = _deps([], margins=[0.05, 0.5])
    deps.stt = _ReplySTT(["저거 뭐야?", "왼쪽 거"])
    chimed: list[tuple[list[str], int]] = []
    deps.start_chime = lambda: chimed.append((list(deps.tts.spoken), deps.stt.calls))

    build_graph(deps).invoke({})

    # 되묻는 말은 읽었고 대답은 아직 안 들었을 때 한 번만
    assert chimed == [([QUESTION], 1)]


@pytest.mark.parametrize(
    ("detections", "tts"),
    [([], {}), (TWO_CUPS, {"fail": [QUESTION]}), (TWO_CUPS, {"interrupt": [QUESTION]})],
    ids=["no_candidates", "question_failed", "called_again"],
)
def test_no_start_chime_without_waiting_for_reply(detections, tts):
    """후보가 없거나, 되묻는 말이 안 나갔거나, 되묻다 다시 불려 대답을 안 기다리면 시작음도 없다."""
    chimed: list[str] = []
    deps = _deps(
        ["저거 뭐야?", "왼쪽 거"],
        margins=[0.05],
        detections=detections,
        start_chime=lambda: chimed.append("start"),
    )
    deps.tts = FakeTextToSpeech(**tts)

    build_graph(deps).invoke({})

    assert chimed == []


# --- 되묻는 중에 다시 부름 ---------------------------------------------------


class _ReplySTT(FakeSpeechToText):
    """listen() 을 부른 횟수를 센다. 두 번째(되묻기 대답)를 듣는 동안 다시 부른 척할 수 있다."""

    def __init__(self, texts, call_while_replying=False) -> None:
        super().__init__(texts)
        self.calls = 0
        self.call_while_replying = call_while_replying
        self.called_again = False

    def listen(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 2 and self.call_while_replying:
            self.called_again = True
        return super().listen(*args, **kwargs)


def test_clarify_cut_off_by_new_call_ends_turn_without_reply(store):
    """되묻는 중에 다시 부르면 대답을 듣지 않고, 빈 대답·끊김·되묻기로 기록만 하고 끝낸다."""
    user_id = store.ensure_user()
    turn = {"session_id": store.start_session(user_id), "user_id": user_id}
    deps = _deps([], margins=[0.05], store=store, dialog=_Dialog())
    deps.stt = _ReplySTT(["저거 뭐야?", "이건 뭐야?"])
    deps.tts = FakeTextToSpeech(interrupt=[QUESTION])

    result = build_graph(deps).invoke(turn)

    assert deps.stt.calls == 1  # 새 질문("이건 뭐야?")은 다음 턴이 받는다
    assert deps.tts.spoken == [QUESTION]  # "다시 가리켜 주세요"를 읽지 않는다
    assert result["interrupted"] is True  # 다음 턴은 BARGE_IN
    assert result["clarify_replies"] == []  # 받아쓴 말을 대답으로 넣지 않았다
    assert deps.dialog.remembered == []
    with store.session() as db:
        logged = db.get(SessionTurn, result["turn_id"])
        assert logged.answer_text == ""
        assert logged.total_ms is None  # 대답 소리가 안 났다. 단계 합으로 채우지 않는다
        assert db.get(TurnVoice, result["turn_id"]).is_interrupted is True
        assert db.get(TurnInference, result["turn_id"]).is_reask is True


def test_new_call_while_waiting_for_reply_ends_turn():
    """되묻기 대답을 기다리는 동안 다시 불렀으면 받아쓴 말을 대답으로 쓰지 않고 끝낸다."""
    deps = _deps([], margins=[0.05])
    deps.stt = _ReplySTT(["저거 뭐야?", "이건 뭐야?"], call_while_replying=True)
    seen: list[float] = []
    deps.called_since = lambda t: seen.append(t) or deps.stt.called_again

    result = build_graph(deps).invoke({})

    assert result["interrupted"] is True
    assert result["clarify_replies"] == []  # 받아쓴 말을 대답으로 넣지 않았다
    assert len(deps.resolve.calls) == 1  # 받아쓴 말로 다시 고르지 않는다
    assert deps.tts.spoken == [QUESTION]
    # 질문을 다 말한 뒤에 들린 호출만 센다
    assert set(seen) == {result["utterance"].ended_at}
    # 대답 듣기를 그 자리에서 멈춰서(stop) 새 질문은 다음 턴의 listen 이 받는다
    assert deps.stt.listen().text == "이건 뭐야?"


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


class _Dialog:
    """DialogMemory.remember 와 같은 모양. 기억에 넣으라고 받은 질문 · 대답을 모은다."""

    def __init__(self) -> None:
        self.remembered: list[tuple[str, str]] = []

    def remember(self, user_id, turn_id, question, answer) -> None:
        self.remembered.append((question, answer))


def test_pointing_context_is_not_saved_to_dialog_memory(store):
    """맥락 문장은 VLM 에만 가고, 대화 기억에는 사용자가 한 원래 질문이 들어간다."""
    user_id = store.ensure_user()
    turn = {"session_id": store.start_session(user_id), "user_id": user_id}
    deps = _deps(["저거 뭐야?"], store=store, dialog=_Dialog())

    build_graph(deps).invoke(turn)

    assert deps.vlm.asked[0][1] == PROMPT
    assert deps.dialog.remembered == [("저거 뭐야?", "컵이에요.")]


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


# --- 손 단서 (v2) ------------------------------------------------------------


class _Hands:
    """정해 둔 손을 돌려준다. HandDetector.detect(image) 와 모양이 같고, 부른 횟수를 센다."""

    def __init__(self, hands=()) -> None:
        self.hands = list(hands)
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[HandPose]:
        self.calls += 1
        return list(self.hands)


def _hand(pointing: bool, base=(965.0, 1030.0), tip=(1000.0, 1000.0)) -> HandPose:
    """pointing 이면 검지만 펴고 나머지를 접은 손, 아니면 다 편 손(키보드에 얹은 손).

    방향은 base → tip. 기본값은 화면 아래에서 오른쪽 컵 중심(1350, 700)을 똑바로 가리킨다.
    """
    folded = [(0.0, 0.0, 0.0), (0.0, 0.02, 0.0), (0.0, 0.02, 0.02), (0.0, 0.0, 0.02)]
    straight = [(0.0, 0.0, 0.0), (0.0, 0.02, 0.0), (0.0, 0.04, 0.0), (0.0, 0.06, 0.0)]
    world = [(0.0, 0.0, 0.0)] * 5 + straight + (folded if pointing else straight) * 3
    landmarks = [(0.0, 0.0)] * 21
    landmarks[5], landmarks[8] = base, tip
    return HandPose(tuple(landmarks), tuple(world), "Right", 0.9)


def test_detect_keeps_only_the_pointing_hand():
    """detect 가 손도 찾아 상태에 둔다. 키보드에 얹은 손은 빼고 가리키는 손만 남긴다."""
    pointing = _hand(pointing=True)
    deps = _deps(["저거 뭐야?"], hands=_Hands([_hand(pointing=False), pointing]))

    result = build_graph(deps).invoke({})

    assert result["hand"] is pointing


def test_real_resolver_uses_hand_without_asking():
    """실제 resolve_target 으로: 컵 둘이 비슷해도 손이 오른쪽 컵을 가리키면 되묻지 않고 고른다."""
    deps = _deps(["저거 뭐야?"], hands=_Hands([_hand(pointing=True)]))
    deps.resolve = resolve_target

    result = build_graph(deps).invoke({})

    assert deps.tts.spoken == ["컵이에요."]  # 되묻는 말 없이 바로 답
    assert result["target"].chosen.detection is TWO_CUPS[1]
    assert [s.kind.value for s in result["target"].chosen.cue_scores] == ["center", "point"]


def test_hand_is_found_once_even_when_clarifying():
    """되묻기 대답으로 resolve 가 다시 돌아도 손은 detect 에서 한 번만 찾는다(같은 사진)."""
    hands = _Hands()
    deps = _deps(["저거 뭐야?", "왼쪽 거"], margins=[0.05, 0.5], hands=hands)

    result = build_graph(deps).invoke({})

    assert len(deps.resolve.calls) == 2
    assert hands.calls == 1
    assert result["hand"] is None  # 손이 안 보이면 None(손 단서 기권)


def test_no_hand_detector_means_no_hand():
    """손 검출기가 없으면(모델 파일 없음 등) 손은 None 이고 가리킴 길은 그대로 돈다."""
    deps = _deps(["저거 뭐야?"])

    result = build_graph(deps).invoke({})

    assert result["hand"] is None
    assert deps.vlm.asked[0][0].image.shape == CROP


# --- 말 단서 (v4) -------------------------------------------------------------------


def test_resolve_passes_the_photo_for_color_words():
    """resolve 노드가 사진을 넘긴다. 말 단서의 색("노란 캔")은 사진이 있어야 잰다."""
    seen = {}

    def resolve(**kwargs) -> TargetResolution:
        seen.update(kwargs)
        return resolve_target(**kwargs)

    deps = _deps(["저거 뭐야?"])
    deps.resolve = resolve

    build_graph(deps).invoke({})

    assert seen["image"].shape == FULL


def test_real_resolver_uses_language_cue_when_switched_on(monkeypatch):
    """말 단서 플래그를 켜면 실제 resolve_target 이 "왼쪽 컵"으로 되묻지 않고 왼쪽 컵을 고른다."""
    monkeypatch.setattr(config, "ENABLE_LANGUAGE_CUE", True)
    deps = _deps(["저거 왼쪽 컵 뭐야?"])
    deps.resolve = resolve_target

    result = build_graph(deps).invoke({})

    assert deps.tts.spoken == ["컵이에요."]  # 되묻는 말 없이 바로 답
    assert result["target"].chosen.detection is TWO_CUPS[0]
    assert [s.kind.value for s in result["target"].chosen.cue_scores] == ["center", "lang"]
