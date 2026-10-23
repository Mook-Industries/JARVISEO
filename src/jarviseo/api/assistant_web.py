"""웹에서 자비서 전체 흐름을 한 번 돌린다.  담당: 권용현

안경의 한 턴(호출어 → 질문 → 의도 → 가리킴 · 기억 · 성분 · VLM → 읽기 → 기록)을
팀원이 만든 그래프(workflow/graph.py, 최홍묵) 그대로 돌린다. 다른 모듈은 고치지 않고,
입출력 부품만 웹에 맞게 바꿔 끼운다.

    마이크 + STT     →  입력한 질문 글자 (voice.FakeSpeechToText, 문태현)
    스피커 + TTS     →  읽을 문장을 모아 화면에 보여 준다 (voice.FakeTextToSpeech, 문태현)
    웹캠 FrameBuffer →  올린 사진 한 장
    나머지          →  그대로: 가리킴(pointing, 최홍묵) · VLM · 기억(vlm · memory, 문태현)
                       · 성분(nutrition)

손에 든 제품 (성분 질문만, 웹에서만)
-----------------------------------
가리킴 검출기(사전학습 COCO YOLO)에는 과자 봉지 · 성분표 분류가 없다. 손(person)은 후보에서 뺀다.
그래서 손에 든 과자를 못 찾거나 뒤의 노트북 · 책을 고른다(2026-10-21 웹캠 실측, 크런키).
그래프 흐름(가리킴 → 크롭 → 성분 판정)은 그대로 두고, 웹에 끼우는 부품에서만 둘을 보완한다.
- 물체를 하나도 못 찾으면 사진 전체를 대상 하나로 준다 (_WholeFrameIfNothing)
- 크롭에서 성분표를 못 읽어 다시 찍어 달라고 할 참이면, 사진 전체를 한 번 더 읽는다
  (_RetryWholeFrame). 손이 가린 데가 있으면 찾은 경고 + 가려졌다는 안내가 나간다(service.py).
진짜 안경에서도 같은 문제라, 가리킴 쪽 해결은 담당(최홍묵)과 따로 정한다.

턴마다 원본 · 후보 박스 · 크롭을 data/captures/web/ 에 남긴다(어디서 막혔는지 보려고).

OPENAI_API_KEY 가 없으면 VLM 은 가짜(vlm.FakeVLMClient)로 답하고 기억 저장 · 검색은 끈다.
의도 분류 · 가리킴 · 크롭 · 성분 판정은 키 없이도 진짜로 돈다.
"""

from __future__ import annotations

import logging
import inspect
import time
from pathlib import Path

import cv2
import numpy as np

from jarviseo import config
from jarviseo.capture import FrameBuffer
from jarviseo.memory import MemoryStore
from jarviseo.nutrition.quality import best_frame
from jarviseo.nutrition.service import IngredientChecker
from jarviseo.types import BBox, Detection, Frame, Intent
from jarviseo.voice import FakeSpeechToText, FakeTextToSpeech
from jarviseo.workflow import GraphDeps, JarviseoState, build_graph
from jarviseo.workflow.router import classify_intent

__all__ = ["NO_KEY_ANSWER", "WebAssistant", "draw_boxes"]

NO_KEY_ANSWER = "VLM 키(OPENAI_API_KEY)가 없어서 사진을 보고 답할 수 없어요. 이건 가짜 답이에요."
WHOLE_LABEL = "사진 전체"  # 물체를 못 찾아 사진 전체를 대상으로 줄 때의 이름
PHOTO_DIR = Path("data/captures/web")


log = logging.getLogger(__name__)


class _WholeFrameIfNothing:
    """물체를 하나도 못 찾으면 사진 전체를 대상 하나로 준다. 성분 질문에만 끼운다."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def detect(self, image: np.ndarray) -> list[Detection]:
        found = self.inner.detect(image)
        if found:
            return found
        h, w = image.shape[:2]
        return [Detection(label=WHOLE_LABEL, confidence=1.0, bbox=BBox(0, 0, w, h))]


def _accepts(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


class _RetryWholeFrame:
    """크롭에서 성분표를 못 읽어 재촬영을 청할 참이면, 사진 전체로 한 번 더 판정한다."""

    def __init__(self, inner: IngredientChecker, frame: np.ndarray) -> None:
        self.inner, self.frame = inner, frame
        self.synonyms = getattr(inner, "synonyms", {})
        self.retried = False  # 사진 전체로 다시 읽었나
        self.used: np.ndarray | None = None  # 마지막 판정에 쓴 사진(크롭 또는 사진 전체)

    def check(self, image: np.ndarray, allergens: list[str], *args, **kwargs):
        # 크롭에 바코드가 안 걸리면 사진 전체에서 찾게 한다.
        # 판정기가 그 인자를 받을 때만 넘긴다(테스트용 가짜 판정기는 안 받는다).
        if image.shape != self.frame.shape and _accepts(self.inner.check, "barcode_image"):
            kwargs.setdefault("barcode_image", self.frame)
        result = self.inner.check(image, allergens, *args, **kwargs)
        kwargs.pop("barcode_image", None)  # 사진 전체로 다시 읽을 때는 필요 없다
        self.used = image
        if result.retake and image.shape != self.frame.shape:
            self.retried = True
            whole = self.inner.check(self.frame, allergens, *args, **kwargs)
            # 사진 전체로 읽었거나, 둘 다 못 읽었어도 사진 전체에서 바코드를 읽었으면 그쪽을 쓴다.
            # 크롭이 바코드를 잘라 먹는 일이 있다(웹캠 실측). 바코드 단계와 표시가 맞게 나간다.
            read = whole.barcode is not None and whole.barcode.barcode is not None
            crop_read = result.barcode is not None and result.barcode.barcode is not None
            if not whole.retake or (read and not crop_read):
                self.used = self.frame
                return whole
        return result


class WebAssistant:
    """웹 요청 하나 = 그래프 한 턴. 오래 사는 부품(검출기 · VLM · 기억)은 처음 한 번만 만든다."""

    def __init__(
        self,
        store: MemoryStore,
        checker: IngredientChecker | None,
        user_id: int,
        vlm=None,
        detector=None,
        resolve=None,
        hands=None,
        photo_dir: Path | None = PHOTO_DIR,
    ) -> None:
        self.store, self.checker, self.user_id = store, checker, user_id
        self.photo_dir = photo_dir  # None 이면 사진을 남기지 않는다(테스트)
        self.vlm_real = vlm is None and bool(config.OPENAI_API_KEY)
        self._vlm, self._detector, self._resolve = vlm, detector, resolve
        # 손 검출기(pointing.hands.HandDetector). 있어야 손끝 방향(v2)으로 가리킨 물건을 고른다.
        # 테스트처럼 detector 를 직접 넣으면 손은 넣은 것만 쓴다(없으면 v1 = 가운데 우선).
        self._hands = hands
        self._load_hands = detector is None and hands is None
        self.pointing = "v1"  # 지난 턴에 쓴 가리킴 방식(v1 가운데 우선 · v2 가운데 + 손 방향)
        self._dialog = None
        self.retried_whole = False  # 지난 턴에 크롭을 못 읽어 사진 전체로 다시 읽었나
        self.judged_image: np.ndarray | None = None  # 지난 턴 성분 판정에 쓴 사진
        self.picked: np.ndarray | None = None  # 지난 턴에 여러 장 중 고른 장면
        self.picked_index = 0  # 몇 번째 장면을 골랐나(0 = 말 시작 순간)
        self.session_id = store.start_session(user_id, title="web")

    def _parts(self) -> None:
        """무거운 부품(YOLO · OpenAI 클라이언트)은 첫 질문 때 만든다. 페이지는 바로 뜨게."""
        if self._vlm is None:
            if self.vlm_real:
                from jarviseo.memory import DialogMemory, VectorMemory
                from jarviseo.vlm import VLMClient

                self._vlm = VLMClient()
                self._dialog = DialogMemory(VectorMemory(self.store))
            else:
                from jarviseo.vlm.client import FakeVLMClient

                self._vlm = FakeVLMClient([NO_KEY_ANSWER])
        if self._load_hands:
            self._load_hands = False
            # scripts/assistant.py 와 같게: 손 모델이나 mediapipe 가 없으면 손 단서 없이 돈다
            try:
                from jarviseo.pointing.hands import HandDetector

                self._hands = HandDetector()
            except (FileNotFoundError, ImportError) as e:
                log.warning("손 단서 꺼짐(가리킴 v1): %s", e)
        if self._detector is None:
            from jarviseo.pointing.detector import ObjectDetector

            self._detector = ObjectDetector()
        if self._resolve is None:
            from jarviseo.pointing import resolve_target

            self._resolve = resolve_target

    def ask(
        self,
        question: str,
        image: np.ndarray | None,
        more: list[tuple[np.ndarray, float]] | None = None,
    ) -> tuple[JarviseoState, list[str]]:
        """질문 한 번을 그래프로 돌리고 (끝난 상태, 읽은 문장들)을 돌려준다.

        Args:
            image: 말을 시작한 순간의 장면.
            more: 그 뒤 잇달아 찍은 장면과 말 시작으로부터 지난 초. 흔들림 · 반사가 가장 적은
                장면을 고른다(nutrition.quality.best_frame). 한 장만 찍으면 그 순간 손이 흔들리거나
                빛이 반사됐을 때 그대로 실패한다(멘토 피드백 2026-10-22).
        """
        self._parts()
        now = time.monotonic()
        frames = None
        if image is not None:
            # 말 시작 뒤 잇달아 찍은 장면 중 품질(선명도 × (1 - 반사))이 가장 좋은 한 장을 쓴다.
            # 그래프의 pick_best 는 선명도만 보므로, 반사까지 보는 성분 쪽 기준으로 먼저 고른다.
            shots = [image, *(shot for shot, _ in more or [])]
            index = best_frame(shots)[0] if len(shots) > 1 else 0
            image = shots[index]
            frames = FrameBuffer(4)
            frames.push(Frame(image=image, timestamp=now, source_id="web"))
            self.picked_index = index
        self.picked = image
        ingredient = classify_intent(question) is Intent.INGREDIENT
        detector, checker = self._detector, self.checker
        if ingredient and image is not None:
            detector = _WholeFrameIfNothing(detector)
            if checker is not None:
                checker = _RetryWholeFrame(checker, image)
        tts = FakeTextToSpeech()
        deps = GraphDeps(
            stt=FakeSpeechToText([question], starts=[now]),
            tts=tts,
            vlm=self._vlm,
            frames=frames,
            detector=detector if image is not None else None,
            hands=self._hands if image is not None else None,
            resolve=self._resolve,
            checker=checker,
            store=self.store,
            memory=self._dialog.memory if self._dialog is not None else None,
            dialog=self._dialog,
            recall_enabled=config.ENABLE_RECALL and self._dialog is not None,
        )
        turn = {"session_id": self.session_id, "user_id": self.user_id, "trigger_type": "WAKEWORD"}
        self.pointing = "v2" if image is not None and self._hands is not None else "v1"
        state = build_graph(deps).invoke(turn)
        self.retried_whole = isinstance(checker, _RetryWholeFrame) and checker.retried
        self.judged_image = checker.used if isinstance(checker, _RetryWholeFrame) else None
        if self.photo_dir is not None:
            save_photos(state, self.photo_dir)
        return state, list(tts.spoken)

    def close(self) -> None:
        if self._dialog is not None:
            self._dialog.close()
        self.store.end_session(self.session_id)


def draw_boxes(state: JarviseoState) -> np.ndarray | None:
    """가리킴 후보 박스를 그린다. 초록 = 고른 대상, 노랑 = 나머지(순위·이름·점수), 십자 = 가운데.

    scripts/assistant.py 가 사진으로 남기는 그림과 같다.
    """
    frame, target = state.get("frame"), state.get("target")
    if frame is None or (target is None and not state.get("detections")):
        return None
    img = frame.image.copy()
    h, w = img.shape[:2]
    cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 40, 2)
    if target is not None:
        chosen = target.chosen.detection if target.chosen is not None else None
        ranked = [(c.detection, c.total_score) for c in target.candidates]
    else:
        chosen, ranked = None, [(d, d.confidence) for d in state["detections"]]
    for rank, (det, score) in reversed(list(enumerate(ranked, 1))):  # 1위를 맨 위에 그린다
        color, thick = ((0, 200, 0), 4) if det is chosen else ((0, 220, 255), 2)
        b = det.bbox
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, thick)
        label = f"{rank} {det.label} {score:.2f}"
        org = (int(b.x1) + 4, max(int(b.y1) + 24, 24))
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


def save_photos(state: JarviseoState, folder: Path = PHOTO_DIR) -> None:
    """턴의 원본 · 후보 박스 · 크롭을 jpg 로 남긴다. 이름은 turn 번호."""
    frame = state.get("frame")
    if frame is None:
        return
    name = f"turn{state.get('turn_id', 'x')}"
    folder.mkdir(parents=True, exist_ok=True)
    shots = (("", frame.image), ("_boxes", draw_boxes(state)), ("_crop", state.get("crop")))
    for suffix, image in shots:
        if image is not None:
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:  # cv2.imwrite 는 윈도우 한글 경로를 못 쓴다
                (folder / f"{name}{suffix}.jpg").write_bytes(buf.tobytes())
