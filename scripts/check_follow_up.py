"""되물은 뒤 호출어 없이 이어 듣기를 실제 부품으로 확인한다 (#253).  담당: 문태현

브라우저 음성(/api/voice/ws, api/browser_voice.py) 연결 하나를 열고 브라우저처럼 마이크 소리(80ms)와
카메라 장면(250ms)을 실시간으로 계속 보낸다. 사용자 말은 OpenAI 목소리로 만든 녹음이다. 그 밖의
받아쓰기 · 읽기 · VLM · 성분 판정(Google Vision 또는 Colab OCR)은 서버가 실제로 돌린다.
사용자는 "자비서, …"를 이어 말하고, 호출은 "자비서"가 끝나고 감지가 늦는 만큼(WAKE_AT) 뒤에
보낸다. 호출어 모델 대신이라 2단계 확인은 건너뛴다(button). 기록 · 턴 사진은 임시 폴더에 남는다.

    A  "자비서, 이거 먹어도 돼?" + 흔들려 흐린 성분표 사진 → 재촬영 안내 → (호출어 없이)
       성분표 사진으로 바꾸고 "다시 확인해줘" → 성분 판정
    B  A 의 첫 질문과 같다 → 재촬영 안내 → 아무 말 없음 → 대기로 돌아감
    C  "자비서, 오늘 날씨 어때?" + 성분표 사진 → 그냥 대답 → 이어 듣지 않고 대기로

OpenAI 키(.env)와 OCR(GOOGLE_VISION_API_KEY 또는 JARVISEO_OCR_REMOTE_URL)이 있어야 한다.
성분표 사진은 원재료명 글을 그려서 만들고, 첫 장면은 그 사진을 흔들린 것처럼 흐리게 한다.
--retake-photo 로 실제 사진을 줄 수 있다(바코드가 보이면 바로 판정해 재촬영 안내가 안 나온다).

실행 (레포 맨 위 폴더에서):
    python scripts/check_follow_up.py
    python scripts/check_follow_up.py --retake-photo 사진.jpg --only A
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

SAMPLE_RATE = 16000
BLOCK = 1280  # 80ms
FRAME_EVERY = 4000  # 250ms
# 말을 시작하고 이만큼 뒤에 호출을 보낸다. "자비서,"(약 0.6초) + 감지 지연
WAKE_AT = 0.9
LABEL_LINES = [
    "고소한 땅콩과자",
    "원재료명: 밀가루(밀:미국산), 설탕, 땅콩(국산) 15%,",
    "대두유, 정제소금, 탄산수소나트륨, 합성향료",
    "알레르기 유발물질: 밀, 땅콩, 대두 함유",
]


def speech(client, text: str) -> np.ndarray:
    """OpenAI 목소리로 사용자 말을 만든다. 24kHz → 16kHz int16."""
    pcm = client.audio.speech.create(
        model="tts-1", voice="echo", input=text, response_format="pcm"
    ).content
    audio = np.frombuffer(pcm, "<i2").astype(np.float32)
    n = int(len(audio) * SAMPLE_RATE / 24000)
    out = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return out.astype(np.int16)


def label_photo() -> np.ndarray:
    """포장지 색 바탕에 원재료명을 크게 쓴 1280×720 사진."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1280, 720), (214, 200, 170))  # 포장지 색. 흰 바탕은 반사로 걸린다
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 44)
    for i, line in enumerate(LABEL_LINES):
        draw.text((60, 140 + i * 90), line, fill="black", font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class Browser:
    """대시보드 브라우저 흉내. 마이크 소리 · 장면을 실시간으로 끊지 않고 보낸다."""

    def __init__(self, ws, image: np.ndarray) -> None:
        from jarviseo.api.browser_voice import AUDIO, FRAME, HEADER

        self.ws, self.header, self.kinds = ws, HEADER, (AUDIO, FRAME)
        self.pos = 0
        self.image = image
        self.said: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, audio: np.ndarray) -> None:
        self.said.put(np.concatenate([audio, np.zeros(SAMPLE_RATE // 2, np.int16)]))

    def show(self, image: np.ndarray) -> None:
        self.image = image

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        audio_kind, frame_kind = self.kinds
        speaking = np.zeros(0, np.int16)
        next_tick = time.monotonic()
        while not self._stop.is_set():
            if not len(speaking) and not self.said.empty():
                speaking = self.said.get()
            block, speaking = speaking[:BLOCK], speaking[BLOCK:]
            block = np.pad(block, (0, BLOCK - len(block)))
            try:
                self.ws.send_bytes(self.header.pack(audio_kind, self.pos) + block.tobytes())
                if self.pos % FRAME_EVERY < BLOCK:
                    jpeg = cv2.imencode(".jpg", self.image)[1].tobytes()
                    self.ws.send_bytes(self.header.pack(frame_kind, self.pos) + jpeg)
            except Exception:
                return
            self.pos += BLOCK
            next_tick += BLOCK / SAMPLE_RATE
            time.sleep(max(0.0, next_tick - time.monotonic()))


def receive(ws, until: set[str], t0: float, log: list[dict]) -> dict:
    """until 중 하나가 올 때까지 알림을 받는다. 소리(바이트)는 버린다."""
    while True:
        m = ws.receive()
        if m.get("bytes") or m.get("text") is None:
            continue
        msg = json.loads(m["text"])
        msg["at"] = round(time.monotonic() - t0, 2)
        log.append(msg)
        if msg["type"] in until:
            return msg


def run(c, name: str, question, reply, first: np.ndarray, second: np.ndarray) -> dict:
    """시나리오 하나. reply 가 None 이면 이어 듣는 동안 말하지 않는다."""
    log: list[dict] = []
    with c.websocket_connect("/api/voice/ws") as ws:
        receive(ws, {"ready"}, time.monotonic(), log)
        browser = Browser(ws, first)
        time.sleep(1.0)  # 직전 소리를 조금 쌓는다
        t0 = time.monotonic()
        browser.say(question)
        time.sleep(WAKE_AT)
        ws.send_json({"type": "wake", "id": 1, "pos": browser.pos, "button": True, "camera": True})
        receive(ws, {"result", "no_question", "error"}, t0, log)
        end = receive(ws, {"idle", "wake"}, t0, log)
        if end["type"] == "wake":  # 호출어 없이 이어 듣는다
            browser.show(second)
            if reply is not None:
                time.sleep(0.7)  # 시작음을 듣고 말한다
                browser.say(reply)
            receive(ws, {"idle"}, t0, log)
        browser.close()
    results = [m["data"] for m in log if m["type"] == "result"]
    follow = next((m for m in log if m["type"] == "wake" and m.get("follow_up")), None)
    done = next(m for m in log if m["type"] == "idle")
    print(f"\n[{name}]")
    for r in results:
        tag = "이어 들음" if r.get("follow_up") else "호출"
        print(f"  {tag}: {r['question']!r} → {r['intent']} → {r['answer']!r}")
    for m in log:
        if m["type"] == "no_question":
            print(f"  {m['at']:5.1f}s 이어 듣는 동안 말 없음 (follow_up={m.get('follow_up')})")
    print(f"  이어 듣기 시작 {follow['at'] if follow else '-'}s · 대기로 {done['at']}s")
    return {"name": name, "results": results, "follow_up_at": follow and follow["at"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--retake-photo", type=Path, help="첫 장면 사진(없으면 흐린 성분표)")
    ap.add_argument("--only", choices=["A", "B", "C"], help="시나리오 하나만")
    args = ap.parse_args()

    label = label_photo()
    if args.retake_photo is None:
        blind = cv2.GaussianBlur(label, (0, 0), 9)
    elif (blind := cv2.imread(str(args.retake_photo))) is None:
        raise SystemExit(f"사진을 못 읽었습니다: {args.retake_photo}")
    scratch = Path(tempfile.mkdtemp())
    os.environ["JARVISEO_API_DB"] = f"sqlite:///{(scratch / 'check_follow_up.db').as_posix()}"
    from fastapi.testclient import TestClient
    from openai import OpenAI

    from jarviseo import config
    from jarviseo.api import browser_voice
    from jarviseo.api.main import create_app
    from jarviseo.memory import SessionTurn

    # 턴 사진도 임시 폴더에. 임시 DB 의 턴 번호(1, 2, …)로 data/captures/web 에 쓰면
    # 실제 웹 턴 사진(turn1.jpg …)을 덮어쓴다(10/28 처음 돌렸을 때 덮어씀).
    browser_voice.PHOTO_DIR = scratch / "captures"

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    eat, again, what = (
        speech(client, t)
        for t in ("자비서, 이거 먹어도 돼?", "다시 확인해줘", "자비서, 오늘 날씨 어때?")
    )
    app = create_app()
    runs = []
    with TestClient(app) as c:
        app.state.store.set_user_allergens(app.state.user_id, ["땅콩"])
        plan = {
            "A": (eat, again, blind, label),
            "B": (eat, None, blind, label),
            "C": (what, None, label, label),
        }
        for name, (q, r, first, second) in plan.items():
            if args.only in (None, name):
                runs.append(run(c, name, q, r, first, second))
        with app.state.store.session() as s:
            rows = s.query(SessionTurn).order_by(SessionTurn.turn_id).all()
            print("\n기록(session_turn)")
            for t in rows:
                print(
                    f"  세션 {t.session_id} 턴 {t.turn_id} {t.trigger_type:13} {t.question_text!r}"
                )


if __name__ == "__main__":
    main()
