"""호출어부터 음성 대답까지 도는 비서.  담당: 문태현

웹캠·마이크·스피커를 열고 "자비서"를 기다린다. 부르면 그래프(src/jarviseo/workflow/graph.py)로
질문 한 번을 돌린다: 받아쓰기 → 의도 → (기억 검색 · 가리킴 · 성분 판정) → VLM → 읽기 → 기록.
대답을 읽거나 기다리는 중에 "자비서"라고 부르면 끊고 새 질문을 받는다(끼어들기).
부르면 시작음을, 질문하는 말이 끝나면(받아쓰기 전) 종료음을 낸다(JARVISEO_CHIME_VOLUME=0 이면 끔).
OpenAI 키와 호출어 모델(data/models/jarviseo.onnx, data/README.md)이 필요하다.

실행 (레포 맨 위 폴더에서):
    python scripts/assistant.py                  # 기본 마이크·스피커, 0번 웹캠
    python scripts/assistant.py 1 4 --camera 1   # 1번 마이크, 4번 스피커, 1번 웹캠
    python scripts/assistant.py --db postgresql+psycopg://jarviseo:jarviseo@localhost:5432/jarviseo

턴은 data/assistant.db(SQLite)에 남는다. --db 로 Postgres 를 줄 수 있다.
턴마다 고른 사진을 data/captures/session_<번호>/ 에 남긴다(--no-photos 로 끔).
    <턴>.jpg        말을 시작한 순간 고른 사진(VLM·가리킴이 받은 원본)
    <턴>_boxes.jpg  가리킴 후보 박스. 초록 = 고른 대상, 노랑 = 나머지(순위·이름·점수), 십자 = 가운데
    <턴>_crop.jpg   VLM·성분 판정에 실제로 간 크롭(대상을 고른 턴만)
기억 검색("아까 본 거 뭐였지?")은 .env 에 JARVISEO_ENABLE_RECALL=true 가 있을 때만 한다.
Ctrl+C 로 끝내면 단계별 지연 중앙값을 찍는다.

찍는 값 (ms)
    stt     받아쓰기 왕복
    recall  질문 임베딩 + 기억 검색
    vlm     VLM 왕복
    tts     대답을 읽기 시작하고 첫 조각이 나오기까지. 답이 "잠깐만요"를 읽는 동안 오면 그때 합성을
            미리 시작하므로, 이 값이 필러가 끝난 뒤 대답 첫 소리까지 기다린 시간이 된다
    total   말이 끝나고 대답 첫 소리가 나오기까지 (스피커 출력 지연 약 0.2초는 빠짐)
"""

import argparse
import statistics
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import make_url
from sqlalchemy.exc import OperationalError

from jarviseo import config
from jarviseo.capture import FrameBuffer, FrameSource, WebcamSource
from jarviseo.memory import DialogMemory, MemoryStore, SessionTurn, VectorMemory
from jarviseo.memory.when import time_range
from jarviseo.nutrition.barcode import make_food_qr_client
from jarviseo.nutrition.service import IngredientChecker
from jarviseo.pointing import resolve_target
from jarviseo.pointing.detector import ObjectDetector
from jarviseo.pointing.hands import HandDetector
from jarviseo.types import MemoryHit
from jarviseo.vlm import VLMClient
from jarviseo.voice import (
    BargeIn,
    Chime,
    ResponseCollector,
    SpeechToText,
    TextToSpeech,
    WakeWordDetector,
)
from jarviseo.voice.mic import MicStream
from jarviseo.workflow import GraphDeps, JarviseoState, build_graph

STAGES = ("stt", "recall", "vlm", "tts", "total")
DEFAULT_DB = "sqlite:///data/assistant.db"
PHOTO_DIR = Path("data/captures")


class Camera:
    """카메라를 계속 읽어 FrameBuffer 에 넣는 스레드. with 안에서만 돈다."""

    def __init__(self, source: FrameSource, frames: FrameBuffer) -> None:
        self.source = source
        self.frames = frames
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "Camera":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.source.release()

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self.source.read()
            if frame is not None:
                self.frames.push(frame)


def show(state: JarviseoState) -> None:
    """턴 하나의 결과를 찍는다."""
    utterance = state.get("utterance")
    if utterance is None:
        print("  질문 없음")
        return
    raw = state.get("stt_raw_text")
    # 재촬영을 포기한 턴은 의도를 정하기 전에 끝나서 intent 가 없다
    print(f"  질문 {utterance.text!r} (원문 {raw!r}) → {state.get('intent')}")
    memories = state.get("memories")
    if memories is not None:
        show_memories(utterance.text, memories)
    flags = []
    if state.get("filler_sent"):
        flags.append("잠깐만요")
    if state.get("interrupted"):
        flags.append("끊김")
    print(f"  대답 {state['response'].text!r}" + (f" ({', '.join(flags)})" if flags else ""))
    latency = state.get("latency_ms", {})
    print("  " + ", ".join(f"{stage} {latency[stage]:.0f}" for stage in STAGES if stage in latency))


def show_memories(question: str, memories: list[MemoryHit]) -> None:
    """VLM 에 넘긴 기록을 한 줄에 하나씩, 본 시각·점수와 함께 찍는다.

    기록 글은 "질문: … / 대답: …" 두 줄이라 그대로 이어 찍으면 옛 대답이 여러 개 나온 것처럼
    보인다. 때 낱말("아까", "어제")로 찾을 범위를 좁혔으면 그 범위도 찍는다.
    """
    window = time_range(question)
    scope = ""
    if window is not None:
        start, end = window
        scope = (
            f" ({_clock(start.timestamp())} ~ {_clock(end.timestamp()) if end else '지금'} 기록만)"
        )
    print(f"  기억 {len(memories)}개{scope}" if memories else f"  기억 없음{scope}")
    for hit in memories:
        when = _clock(hit.observed_at) if hit.observed_at is not None else "시각 없음"
        print(f"    {when}  {hit.score:.2f}  {hit.text.replace(chr(10), ' / ')}")


def _clock(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%m/%d %H:%M")


def save_photos(state: JarviseoState, folder: Path, number: int, store: MemoryStore) -> None:
    """턴에서 쓴 사진·후보 박스·크롭을 jpg 로 남기고 원본 경로를 session_turn.image_path 에."""
    frame = state.get("frame")
    if frame is None:
        return
    turn_id = state.get("turn_id")
    name = f"{number:02d}" + (f"_turn{turn_id}" if turn_id is not None else "")
    folder.mkdir(parents=True, exist_ok=True)
    original = folder / f"{name}.jpg"
    _write_jpg(original, frame.image)
    saved = [original.name]

    target = state.get("target")
    if target is not None or state.get("detections"):
        boxes = frame.image.copy()
        height, width = boxes.shape[:2]
        cv2.drawMarker(boxes, (width // 2, height // 2), (255, 255, 255), cv2.MARKER_CROSS, 40, 2)
        if target is not None:
            chosen = target.chosen.detection if target.chosen is not None else None
            ranked = [(c.detection, c.total_score) for c in target.candidates]
        else:
            chosen, ranked = None, [(d, d.confidence) for d in state["detections"]]
        for rank, (det, score) in reversed(list(enumerate(ranked, 1))):  # 1위를 맨 위에 그린다
            picked = det is chosen
            color, thick = ((0, 200, 0), 4) if picked else ((0, 220, 255), 2)
            b = det.bbox
            top_left, bottom_right = (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2))
            cv2.rectangle(boxes, top_left, bottom_right, color, thick)
            label = f"{rank} {det.label} {score:.2f}"
            org = (int(b.x1) + 4, max(int(b.y1) + 24, 24))
            cv2.putText(boxes, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(boxes, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        _write_jpg(folder / f"{name}_boxes.jpg", boxes)
        saved.append("boxes")
    crop = state.get("crop")
    if crop is not None:
        _write_jpg(folder / f"{name}_crop.jpg", crop)
        saved.append("crop")
    print(f"  사진 {original} (+{', '.join(saved[1:])})" if saved[1:] else f"  사진 {original}")

    if turn_id is not None:
        with store.session() as db:
            row = db.get(SessionTurn, turn_id)
            if row is not None:
                row.image_path = original.as_posix()


def _write_jpg(path: Path, image: np.ndarray) -> None:
    # cv2.imwrite 는 윈도우에서 한글 경로를 못 쓴다. 바이트로 만들어 파이썬이 쓴다.
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        path.write_bytes(jpeg.tobytes())


def run(
    graph,
    barge: BargeIn,
    chime: Chime,
    session_id: int,
    user_id: int,
    turns: list,
    store: MemoryStore,
    photo_dir: Path | None,
) -> None:
    """호출을 기다렸다가 그래프로 질문 한 번씩 돌린다. Ctrl+C 로 끝난다."""
    trigger = "WAKEWORD"
    number = 0
    while True:
        wake = barge.wait(timeout=0.5)
        if wake is None:
            continue
        chime.start()  # 듣고 있다는 신호. 바로 그래프가 듣기 시작해서 이어 말한 질문도 받는다
        print(f"\n{'끼어들기' if trigger == 'BARGE_IN' else '호출어'} (점수 {wake.score:.3f})")
        turn = {
            "session_id": session_id,
            "user_id": user_id,
            "wake_at": wake.t,
            "trigger_type": trigger,
        }
        state = graph.invoke(turn)
        show(state)
        number += 1
        if photo_dir is not None:
            save_photos(state, photo_dir, number, store)
        if "turn_id" in state:
            turns.append(state.get("latency_ms", {}))
        # 대답을 읽다 끊겼거나 기다리는 동안(되묻는 중도) 다시 불렀으면 그 호출이 쌓여 있고,
        # 다음 턴이 된다.
        # 끝까지 읽었으면 질문하는 동안 다시 잡힌 호출어 같은 지난 감지는 버린다.
        trigger = "BARGE_IN" if state.get("interrupted") else "WAKEWORD"
        if trigger == "WAKEWORD":
            barge.clear()


def open_store(url: str) -> MemoryStore:
    """DB 를 열고 테이블을 만든다. 못 붙으면 할 일을 알려 주고 끝낸다.

    마이크·카메라를 열기 전에 한다. 도커를 안 켠 채 시연을 시작해도 몇 초 안에 이유를 안다(#131).
    """
    store = MemoryStore(database_url=url)
    try:
        store.init_schema()
    except OperationalError as e:
        where = make_url(url).render_as_string(hide_password=True)
        reason = str(e.orig or e).strip().splitlines()[0]
        raise SystemExit(
            f"DB 에 연결하지 못했습니다: {where}\n  {reason}\n"
            "  Postgres 면 도커를 켜고 `docker compose up -d db` 를 하거나, "
            f"--db 를 빼고 SQLite({DEFAULT_DB})로 돌리세요."
        ) from None
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="호출어 → 질문 → VLM → 음성 대답")
    parser.add_argument("mic", type=int, nargs="?", help="sounddevice 입력 장치 번호")
    parser.add_argument("speaker", type=int, nargs="?", help="sounddevice 출력 장치 번호")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="웹캠 번호")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"턴을 남길 DB (기본 {DEFAULT_DB})")
    parser.add_argument("--no-photos", action="store_true", help=f"{PHOTO_DIR}/ 에 사진 안 남김")
    args = parser.parse_args()

    store = open_store(args.db)
    user_id = store.ensure_user()
    session_id = store.start_session(user_id, title="assistant")
    vlm = VLMClient()
    dialog = DialogMemory(VectorMemory(store))
    tts = TextToSpeech(device=args.speaker)
    chime = Chime(device=args.speaker)
    barge = BargeIn(tts)
    collector = ResponseCollector(tts, called_since=barge.called_since)  # "잠깐만요"를 미리 합성
    frames = FrameBuffer(config.FRAME_BUFFER_SIZE)
    turns: list[dict[str, float]] = []
    # YOLO·동의어 사전을 읽는 데 몇 초 걸린다. 마이크·카메라를 열기 전에 읽어야 그동안 쌓인
    # 소리·사진이 첫 질문에 섞이지 않는다. 아래 WakeWordDetector 와 헷갈리지 않게 이름을 달리 둔다.
    object_detector = ObjectDetector()
    # 손 모델 파일이나 mediapipe 가 없으면 손 단서 없이(v1 처럼) 돈다. 팀원 PC 에서도 떠야 한다.
    try:
        hand_detector = HandDetector()
    except (FileNotFoundError, ImportError) as e:
        print(f"손 단서 꺼짐: {e}")
        hand_detector = None
    # 성분 질문: 바코드 → 캐시 → 푸드QR → 성분표 OCR → 재촬영 안내.
    # 식품안전나라는 쓰지 않는다. 원재료가 '면류, 분말스프'처럼 상위 분류만 오고
    # 바코드가 다른 제품으로 잡히는 일이 있었다(2026-10-16 실측, #81).
    # 성분표 OCR 은 .env 에 JARVISEO_OCR_REMOTE_URL 이 있으면 Colab GPU 서버에서 돈다(#82).
    qr = make_food_qr_client()
    checker = IngredientChecker(store=store, food_qr=qr)

    with MicStream(args.mic) as mic, Camera(WebcamSource(args.camera), frames):
        deps = GraphDeps(
            stt=SpeechToText(mic=mic, on_speech_end=chime.end),  # 질문 말이 끝나면 종료음
            tts=tts,
            vlm=vlm,
            frames=frames,
            store=store,
            memory=dialog.memory,
            dialog=dialog,
            collector=collector,
            called_since=barge.called_since,  # 되묻는 동안 다시 불렀는지
            detector=object_detector,
            hands=hand_detector,
            resolve=resolve_target,
            checker=checker,
        )
        graph = build_graph(deps)
        detector = WakeWordDetector(mic, on_detect=barge.on_detect)
        detector.start()
        photo_dir = None if args.no_photos else PHOTO_DIR / f"session_{session_id}"
        print(f"기록 {args.db} (session_id {session_id}), VLM {vlm.model}")
        if photo_dir is not None:
            print(f"사진 {photo_dir}/")
        print(f"기억 검색 {'켜짐' if deps.recall_enabled else '꺼짐 (JARVISEO_ENABLE_RECALL)'}")
        pointing = "v2(가운데 + 손 방향)" if hand_detector else "v1(가운데 우선)"
        print(f"가리킴 {pointing} · 성분 판정 켜짐")
        qr_state = "푸드QR" if qr else "푸드QR 꺼짐(JARVISEO_FOOD_QR_API_KEY 없음)"
        ocr_state = "Colab GPU" if config.OCR_REMOTE_URL else "이 PC CPU"
        print(f"성분 판정: 캐시 → {qr_state} → 성분표 OCR({ocr_state})")
        print("\n'자비서'를 부르고 이어서 말하세요. 끝내려면 Ctrl+C")
        try:
            run(graph, barge, chime, session_id, user_id, turns, store, photo_dir)
        except KeyboardInterrupt:
            pass
        finally:
            detector.stop()

    print("\n마무리하는 중 (받던 답·대화 기억 저장)...")
    collector.close()
    dialog.close()
    if qr is not None:
        qr.close()
    store.end_session(session_id)
    if turns:
        print(f"\n중앙값 ({len(turns)}턴, ms)")
        for stage in STAGES:
            values = [turn[stage] for turn in turns if stage in turn]
            if values:
                print(f"  {stage} {statistics.median(values):.0f} ({len(values)}턴)")


if __name__ == "__main__":
    main()
