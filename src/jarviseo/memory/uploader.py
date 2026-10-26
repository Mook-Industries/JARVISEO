"""끝난 턴을 클라우드 대시보드로 보내기 (#144).  담당: 문태현

노트북의 assistant.py 는 턴을 로컬 DB(SQLite)에 먼저 쓴다. 이 모듈은 그 뒤에 다른 스레드에서
새 턴을 읽어 클라우드 서버(``POST /api/turns``)로 보낸다. 대답은 보내기를 기다리지 않는다.
노트북이 클라우드 Postgres 에 바로 쓰면 인터넷이 느릴 때 턴 기록 · 기억 검색이 그만큼 늦고,
끊기면 assistant.py 가 멈춘다(#131 과 같은 종류). 발표장 인터넷은 믿기 어렵다.

어디까지 보냈는지는 ``data/cloud_upload.json`` 에 마지막으로 보낸 turn_id 로 남긴다.
못 보낸 턴은 그 뒤에 그대로 있으니, 연결되면(다음 턴 · 다시 시도 · 다음에 켤 때) 이어서 보낸다.
처음 켜면 그때까지의 턴은 보내지 않고 이후 턴부터 보낸다(예전 측정 턴은 노트북에만 둔다).
사진 · 녹음은 보내지 않는다(store.UPLOAD_FIELDS).

    uploader = TurnUploader(store, config.CLOUD_URL, config.CLOUD_TOKEN)
    uploader.start()
    ...                      # 턴마다
    uploader.notify()        # 바로 돌아온다
    ...
    uploader.close()         # 끝낼 때 한 번 더 보내 본다
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path

import httpx

from jarviseo import config
from jarviseo.memory.store import MemoryStore, turn_payload

__all__ = ["TurnUploader"]

log = logging.getLogger(__name__)

STATE_PATH = config.DATA_DIR / "cloud_upload.json"
BATCH = 20  # 한 번에 보내는 턴 수. 오래 끊겼다 붙어도 요청 하나가 너무 커지지 않게
RETRY_SEC = (2.0, 60.0)  # 실패하면 2초부터 두 배씩, 60초까지 기다렸다 다시 보낸다
IDLE_SEC = 30.0  # 턴이 없어도 이만큼마다 한 번 본다(notify 를 못 받은 턴이 있어도 결국 간다)


class TurnUploader:
    """로컬 DB 의 새 턴을 클라우드로 보내는 백그라운드 스레드."""

    def __init__(
        self,
        store: MemoryStore,
        url: str,
        token: str,
        state_path: Path = STATE_PATH,
        timeout: float = 5.0,
        transport: httpx.BaseTransport | None = None,  # 테스트에서 가짜 서버를 끼운다
        retry_sec: tuple[float, float] = RETRY_SEC,
        idle_sec: float = IDLE_SEC,
    ) -> None:
        self.store = store
        self.url = url.rstrip("/")
        self.state_path = state_path
        self.retry_sec = retry_sec
        self.idle_sec = idle_sec
        self._http = httpx.Client(
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {token}"},
        )
        self._state = self._load_state()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="turn-uploader", daemon=True)
        self._notified_at: float | None = None
        # 턴이 끝나고(notify) 서버가 받기까지 걸린 시간. assistant.py 가 마치며 중앙값을 찍는다
        self.delays_ms: list[float] = []
        self.last_error: str | None = None

    @property
    def last_sent_id(self) -> int:
        """서버가 받은 마지막 로컬 turn_id."""
        return self._state["last_turn_id"]

    def pending(self) -> int:
        """아직 못 보낸 턴 수."""
        return len(self.store.turns_after(self.last_sent_id, limit=10_000))

    def start(self) -> None:
        self._thread.start()

    def notify(self) -> None:
        """턴 하나가 기록됐다. 바로 돌아오고, 보내기는 스레드가 한다."""
        self._notified_at = time.perf_counter()
        self._wake.set()

    def close(self, wait: float = 5.0) -> None:
        """스레드를 멈춘다. 멈추기 전에 한 번 더 보내 본다. 못 보낸 턴은 다음에 켤 때 보낸다."""
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=wait)
        self._http.close()

    def flush(self) -> bool:
        """밀린 턴을 다 보낸다. 다 보냈으면 True, 서버가 안 받았으면 False."""
        while True:
            turns = self.store.turns_after(self.last_sent_id, limit=BATCH)
            if not turns:
                return True
            body = {"source": self._state["source"], "turns": [turn_payload(t) for t in turns]}
            try:
                r = self._http.post(f"{self.url}/api/turns", json=body)
                r.raise_for_status()
            except httpx.HTTPError as e:
                self._fail(e)
                return False
            if self._notified_at is not None:
                self.delays_ms.append((time.perf_counter() - self._notified_at) * 1000)
                self._notified_at = None
            try:
                failed = r.json().get("failed", 0)
            except ValueError:
                failed = 0
            if failed:
                log.warning("클라우드가 턴 %d개를 남기지 못함 (서버 로그 확인)", failed)
            if self.last_error is not None:
                log.warning("클라우드에 다시 연결됨")
                self.last_error = None
            self._state["last_turn_id"] = turns[-1].turn_id
            self._save_state()

    def _fail(self, e: httpx.HTTPError) -> None:
        if isinstance(e, httpx.HTTPStatusError):
            reason = f"{e.response.status_code} {e.response.text[:100]}"
        else:
            reason = type(e).__name__
        if reason != self.last_error:  # 같은 이유로 계속 실패하면 한 번만 알린다
            log.warning("클라우드로 턴을 보내지 못함: %s (연결되면 다시 보낸다)", reason)
        self.last_error = reason

    def _run(self) -> None:
        delay = self.retry_sec[0]
        while True:
            self._wake.clear()
            try:
                ok = self.flush()
            except Exception:
                # 다른 스레드라 예외가 묻힌다. 남기고 계속 돈다(대답에는 영향이 없다).
                log.exception("턴 보내기 실패")
                ok = False
            if self._stop.is_set():
                return
            if ok:
                delay = self.retry_sec[0]
                self._wake.wait(self.idle_sec)
            else:
                self._wake.wait(delay)
                delay = min(delay * 2, self.retry_sec[1])

    def _load_state(self) -> dict:
        """어디까지 보냈는지 읽는다. 처음이거나 다른 서버면 지금까지의 턴은 건너뛴다."""
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        if state.get("url") != self.url or not isinstance(state.get("last_turn_id"), int):
            # source 는 이 노트북을 가리키는 임의 이름. 컴퓨터 이름(사람 이름이 들어가곤 한다)
            # 대신 쓴다.
            state = {
                "url": self.url,
                "source": state.get("source") or uuid.uuid4().hex[:8],
                "last_turn_id": self.store.last_turn_id(),
            }
            self._state = state
            self._save_state()
        return state

    def _save_state(self) -> None:
        # 쓰다가 꺼져도 반쪽 파일이 남지 않게 옆에 쓰고 바꿔 끼운다.
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)
