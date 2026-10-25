"""가리킴 노드 — 검출 · 대상 결정 · 되묻기 · 크롭.  담당: 최홍묵

detect      : 사진에서 물체 후보와 가리키는 손을 찾는다 (pointing/detector.py · hands.py).
              이어 묻기 단서가 쓸 직전 대상(같은 세션 바로 전 가리킴 턴)도 DB 에서 한 번 읽는다
resolve     : 후보 중 가리킨 대상을 고른다 (pointing/resolver.py)
clarify     : 1·2위가 비슷하면 되묻고 대답을 듣는다. 사진은 다시 고르지 않는다.
              되묻는 말 뒤에 시작음, 대답이 끝나면 종료음을 낸다(처음 질문할 때와 같다)
              2번 넘거나 대답이 없거나 후보가 없으면 "다시 가리켜 주세요"로 끝낸다.
              되묻는 중이나 대답을 기다리는 중에 "자비서"라고 다시 부르면 그 말은 대답이 아니라
              새 질문이다. 빈 대답으로 기록만 하고 끝내서, 새 질문은 다음 턴(BARGE_IN)이 받는다.
              되묻는 말을 합성하지 못해 소리가 안 났으면 대답을 기다리지 않고 바로 끝낸다
crop_target : 고른 대상을 여백 15% 넣어 잘라낸다 (pointing/crop.py). 성분 판정은 이 크롭 그대로,
              VLM 에는 generate 가 크기를 맞춰 보낸다
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from jarviseo.memory import SessionTurn, TurnInference
from jarviseo.pointing.crop import crop_target as cut_target
from jarviseo.pointing.hands import pick_pointing_hand
from jarviseo.pointing.labels import korean_name
from jarviseo.workflow.deps import GraphDeps
from jarviseo.workflow.messages import CLARIFY_GIVE_UP, canned_response
from jarviseo.workflow.nodes.voice import REPLY_SEC
from jarviseo.workflow.state import MAX_CLARIFY, JarviseoState

__all__ = [
    "REPLY_HINTS",
    "after_clarify",
    "after_resolve",
    "clarify",
    "crop_target",
    "detect",
    "resolve",
]

log = logging.getLogger(__name__)

# 되묻기 대답("왼쪽 거")을 받아쓸 때 STT 에 주는 힌트. 짧은 대답이 덜 틀린다(voice/stt.py).
# 장면에 있는 후보 이름(한글)은 clarify 가 같이 더한다.
REPLY_HINTS = ("왼쪽", "오른쪽", "가운데")
# 이어 묻기 단서가 "그거"를 앞 대상으로 읽는 시간(초). 이보다 오래된 가리킴 턴은 직전 대상이 아니다.
# 출발점이다(말 · 맥락 세션 s04 의 이어 묻기는 10~20초 간격이었다).
PREV_TARGET_SEC = 60


def detect(state: JarviseoState, deps: GraphDeps) -> dict:
    image = state["frame"].image
    # 손도 여기서 한 번만 찾는다. resolve 는 되묻기 대답마다 다시 돌지만 사진은 그대로다.
    hand = pick_pointing_hand(deps.hands.detect(image)) if deps.hands is not None else None
    return {
        "detections": deps.detector.detect(image),
        "hand": hand,
        # 직전 대상도 사진 · 손처럼 턴마다 한 번만 정한다. 되묻기 대답마다 DB 를 다시 읽지 않는다.
        "prev_target_label": _previous_target(deps, state),
    }


def _previous_target(deps: GraphDeps, state: JarviseoState) -> str | None:
    """같은 세션에서 바로 전 가리킴 턴(PREV_TARGET_SEC 안)에 고른 대상의 이름. 없으면 None.

    턴마다 그래프를 새로 불러 상태가 남지 않으므로, 이미 기록하는 turn_inference.target_label 을
    읽는다. 그래프를 부르는 쪽(assistant.py · 웹)을 고치지 않으려는 것이다. 지금 턴은 log 노드가
    끝에서 기록하므로 여기서는 안 읽힌다. DB 가 없거나 오류가 나면 None(이어 묻기 기권)이고 턴은
    그대로 돈다.
    """
    session_id = state.get("session_id")
    if deps.store is None or session_id is None:
        return None
    stmt = (
        select(TurnInference.target_label, SessionTurn.asked_at)
        .join(SessionTurn, SessionTurn.turn_id == TurnInference.turn_id)
        .where(SessionTurn.session_id == session_id, TurnInference.target_label.is_not(None))
        # 같은 초에 기록된 턴은 asked_at 이 같을 수 있어(SQLite 는 초 단위) 번호로 한 번 더 가른다.
        .order_by(SessionTurn.asked_at.desc(), SessionTurn.turn_id.desc())
        .limit(1)
    )
    try:
        with deps.store.session() as db:
            row = db.execute(stmt).first()
    except SQLAlchemyError as e:
        log.warning("previous target lookup failed: %s", e)
        return None
    if row is None:
        return None
    label, asked_at = row
    # asked_at 은 DB 시계(func.now())다. SQLite 는 시간대 없는 UTC 로, Postgres 는 시간대를 붙여
    # 돌려준다. 시간대 없는 값을 그대로 지금 시각과 빼면 TypeError 가 나거나 9시간이 틀어진다.
    if asked_at.tzinfo is None:
        asked_at = asked_at.replace(tzinfo=UTC)
    age = (datetime.now(UTC) - asked_at).total_seconds()
    return label if age <= PREV_TARGET_SEC else None


def resolve(state: JarviseoState, deps: GraphDeps) -> dict:
    # 키워드로 불러야 가짜와 실제 resolve_target 을 바꿔 끼워도 안 깨진다.
    target = deps.resolve(
        candidates=state["detections"],
        hand=state.get("hand"),
        gaze_vector=None,
        text=state["utterance"].text,
        frame_size=state["frame"].size,
        # 되묻기 대답("왼쪽 거")은 단서가 아니라 후보를 좁히는 데 쓴다. 질문과 따로 넘긴다.
        replies=state.get("clarify_replies", []),
        # 말 단서의 색("노란 캔")을 보려고 사진도 넘긴다. 말 단서가 꺼져 있으면 쓰이지 않는다.
        image=state["frame"].image,
        # 이어 묻기 단서가 "그거"를 읽을 직전 대상. 이어 묻기 단서가 꺼져 있으면 쓰이지 않는다.
        prev_target_label=state.get("prev_target_label"),
    )
    return {"target": target}


def after_resolve(state: JarviseoState) -> Literal["ambiguous", "chosen"]:
    return "ambiguous" if state["target"].needs_clarify else "chosen"


def clarify(state: JarviseoState, deps: GraphDeps) -> dict:
    target, count = state["target"], state.get("clarify_count", 0)
    if not target.candidates or count >= MAX_CLARIFY:
        return {"response": canned_response(state, CLARIFY_GIVE_UP)}
    took = deps.tts.speak(target.clarify_question or CLARIFY_GIVE_UP)
    # 질문을 다 말한 뒤에 들린 호출어는 새로 부른 것이다. 질문 첫머리의 호출어가
    # 한 번 더 잡힌 것은 이보다 이르다.
    question_end = state["utterance"].ended_at

    def called_again() -> bool:
        return _called_since(deps, question_end)

    if deps.tts.last_interrupted or called_again():
        return _called_again(state)  # 되묻는 중에 다시 불러서 되묻기가 끊겼다
    if took is None:
        # 되묻는 말을 합성하지 못해 소리가 하나도 안 났다. 사용자는 못 들은 질문에 대답하지
        # 않으니 5초를 기다려도 소용없다(#105 세션 9·10, 서버가 느려 약 9초 뒤 실패, total 16~18초).
        # 이미 수 초를 썼으니 다시 합성하지 않고 "다시 가리켜 주세요"로 끝낸다.
        log.warning("되묻는 말을 합성하지 못해 대답을 기다리지 않는다")
        return {"response": canned_response(state, CLARIFY_GIVE_UP)}
    # 사진을 새로 고르지 않는다(on_speech_start 없음). 대답하는 사이 고개를 돌렸어도
    # 처음 가리킨 장면에서 다시 골라야 한다.
    # 대답은 "컵"처럼 한 글자일 수 있다. 기본(두 글자 미만은 버림)대로면 대답이 없던 것이 된다.
    # "컵" 한 글자는 "급"·"켭"으로 틀리기 쉽다. 장면에 있는 후보 이름을 힌트로 같이 준다.
    names = (korean_name(c.detection.label) for c in target.candidates)
    hints = tuple(dict.fromkeys((*REPLY_HINTS, *names)))
    # 대답을 기다리다 다시 부르면 그 자리에서 듣기를 멈춘다. 5초를 다 채우면 사용자는 신호(시작음)를
    # 기다리며 말을 멈추고, 다음 턴은 "자비서"만 다시 읽어 "네, 말씀하세요"를 해야 했다(세션 11).
    # 되묻는 말 끝에 걸쳐 부르면 감지가 재생 뒤로 밀리는데(감지 지연 최대 0.33초), 그 감지도 듣기를
    # 시작한 뒤에 들어오므로 여기서 잡힌다. 듣기 전에 그만큼 쉬지 않는 것은, 쉬는 동안 시작한
    # 대답("왼쪽") 앞부분이 잘리기 때문이다(since 없는 listen 은 부른 뒤 소리부터 읽는다).
    # 되묻는 말을 다 읽었다. 시작음으로 이제 대답하라고 알린다. 대답이 끝나면 stt 가 종료음을 낸다.
    # 시작음 구간은 VAD 가 말로 안 볼 뿐 녹음에는 남아서, 시작음과 겹쳐 대답해도 앞이 안 잘린다.
    if deps.start_chime is not None:
        deps.start_chime()
    reply = deps.stt.listen(timeout=REPLY_SEC, min_letters=1, hints=hints, stop=called_again)
    if called_again():
        return _called_again(state)  # 대답을 기다리는 동안 다시 불렀다. 그 말은 새 질문이다
    if reply is None:
        return {"response": canned_response(state, CLARIFY_GIVE_UP)}
    # 한 개짜리 리스트로 줘야 리듀서(operator.add)가 앞의 대답 뒤에 이어 붙인다.
    return {"clarify_replies": [reply.text], "clarify_count": count + 1}


def after_clarify(state: JarviseoState) -> Literal["called_again", "gave_up", "replied"]:
    if state.get("interrupted"):
        return "called_again"  # 읽을 것 없이 기록만
    return "gave_up" if "response" in state else "replied"  # 포기 안내를 만들었으면 읽으러


def _called_since(deps: GraphDeps, t: float) -> bool:
    return deps.called_since is not None and deps.called_since(t)


def _called_again(state: JarviseoState) -> dict:
    """되묻다가 다시 불렸을 때. 답을 버린 generate 처럼 빈 대답으로 기록만 한다.

    빈 대답이라 대화 기억에는 안 들어가고, interrupted 라서 쌓인 호출이 다음 턴(BARGE_IN)이 된다.
    """
    return {"response": canned_response(state, ""), "interrupted": True}


def crop_target(state: JarviseoState, deps: GraphDeps) -> dict:
    bbox = state["target"].chosen.detection.bbox
    return {"crop": cut_target(state["frame"].image, bbox)}
