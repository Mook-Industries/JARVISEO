"""웹 · 음성 턴 사진 남기기(api/assistant_web.save_photos) — 기억 질문이 그때 사진을 쓴다(#260).

fastapi 없이 돈다(CI 에서도 돈다).
"""

import time

import numpy as np
import pytest

pytest.importorskip("langgraph")
from jarviseo.api.assistant_web import save_photos  # noqa: E402
from jarviseo.memory import SessionTurn  # noqa: E402
from jarviseo.types import AssistantResponse, Frame, Intent, Utterance  # noqa: E402


def logged_turn(store) -> int:
    session = store.start_session(store.ensure_user())
    now = time.monotonic()
    utterance = Utterance("이거 뭐야?", now - 1.0, now)
    return store.log_turn(session, utterance, AssistantResponse("컵이에요.", Intent.POINTING))


def turn_state(turn_id, crop=True) -> dict:
    image = np.zeros((48, 64, 3), np.uint8)
    state = {"turn_id": turn_id, "frame": Frame(image=image, timestamp=0.0, source_id="web")}
    if crop:
        state["crop"] = image[:20, :20]
    return state


def test_saves_original_path_for_recall_photo(store, tmp_path):
    """원본 · 크롭을 남기고 원본 경로를 session_turn.image_path 에 적는다."""
    turn_id = logged_turn(store)

    save_photos(turn_state(turn_id), tmp_path, store)

    assert (tmp_path / f"turn{turn_id}_crop.jpg").is_file()
    with store.session() as db:
        assert (
            db.get(SessionTurn, turn_id).image_path == (tmp_path / f"turn{turn_id}.jpg").as_posix()
        )


def test_removes_stale_crop_of_reused_turn_number(store, tmp_path):
    """같은 번호의 예전 크롭이 남아 있으면 지운다. 이 턴의 사진으로 쓰이면 안 된다."""
    turn_id = logged_turn(store)
    (tmp_path / f"turn{turn_id}_crop.jpg").write_bytes(b"old")

    save_photos(turn_state(turn_id, crop=False), tmp_path, store)

    assert not (tmp_path / f"turn{turn_id}_crop.jpg").exists()
    assert (tmp_path / f"turn{turn_id}.jpg").is_file()
