"""클라우드 VLM 호출.  담당: 문태현

하이브리드 온디바이스의 경계선이 이 파일이다
---------------------------------------------
이미지가 네트워크로 나가는 곳은 여기뿐이다. 그리고 나가는 것은
**질문한 시점의 1프레임**뿐이다. 영상 스트림이 아니다.
웨이크워드·객체검출처럼 상시 동작하는 것은 전부 로컬에서 돈다.
이 원칙을 깨면 프로젝트의 프라이버시 근거가 통째로 사라진다.

모델 이름은 ``config.VLM_MODEL`` 하나만 본다. 팀 공용 API 키 1개를 쓰므로 월 사용 한도를 걸어 둔다.

``FakeVLMClient`` 는 키 없이 쓰는 가짜다. 메서드 모양이 실제와 같아서
그래프 테스트에서 바꿔 끼우면 된다(``tests/test_vlm.py`` 가 확인한다).
"""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Iterable
from datetime import datetime

import cv2
import numpy as np
from openai import OpenAI, OpenAIError

from jarviseo import config
from jarviseo.types import Frame, MemoryHit

__all__ = ["FakeVLMClient", "VLMClient", "encode_image"]

log = logging.getLogger(__name__)

# 한 번 기다리는 최대 시간(초)과 다시 보내는 횟수. 그동안 사용자가 기다리므로 짧게 끊는다.
TIMEOUT_SEC, RETRIES = 10.0, 1
# 사진 긴 변(px). 크면 업로드 시간과 이미지 토큰(비용)이 는다.
MAX_SIDE = 1568
JPEG_QUALITY = 85

INSTRUCTIONS = """\
너는 사용자가 쓰고 있는 스마트 안경의 음성 비서 '자비서'다.
사진은 사용자가 질문한 순간 안경 카메라에 보인 장면이다.
대답은 스피커로 읽어 주므로 한국어 존댓말 한두 문장으로 짧게 말한다.
목록, 기호, 이모지, 마크다운은 쓰지 않는다.
사진으로 알 수 없는 것은 추측하지 말고 모른다고 말한다."""

MEMORY_RULE = """\
아래는 사용자가 전에 본 장면 기록이다. 기억을 묻는 질문이면 이 기록만 근거로 답한다.
기록에 없는 내용은 지어내지 말고 기억에 없다고 말한다."""

SCENE_INSTRUCTIONS = """\
사진은 스마트 안경 카메라에 보인 장면이다. 나중에 사용자가 "아까 본 그거"를 물으면
찾을 수 있게, 눈에 띄는 물건과 그 색·위치·적힌 글자를 한국어 한두 문장으로 적는다.
사람의 얼굴 생김새나 신원은 적지 않는다."""


class VLMClient:
    """사진 한 장과 질문을 보내고 답 문장을 받는다."""

    def __init__(
        self, model: str = config.VLM_MODEL, max_tokens: int = 300, client: OpenAI | None = None
    ) -> None:
        """max_tokens 는 답 길이 상한이다. 한두 문장이면 100 토큰 안팎이다.

        client 는 테스트에서 가짜 API 를 끼울 때만 넘긴다.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or OpenAI(
            api_key=config.OPENAI_API_KEY or None, timeout=TIMEOUT_SEC, max_retries=RETRIES
        )
        # 마지막 호출의 왕복 시간(ms). 실패했으면 None.
        # 그래프가 latency_ms["vlm"] 로 넘기면 log_turn 이 turn_inference.llm_ms 에 적는다.
        self.last_latency_ms: float | None = None

    def ask(
        self, frame: Frame | None, question: str, memories: list[MemoryHit] | None = None
    ) -> str | None:
        """질문에 대한 답 문장을 돌려준다. 실패하거나 답이 비었으면 None.

        frame 이 없으면 글자만 보낸다. memories 는 기억 검색(RECALL) 결과다.
        None 이면 기억을 안 찾은 질문이고, 빈 목록이면 찾았는데 없던 것이다.
        빈 목록일 때도 "찾은 기록 없음"을 알려야 모델이 사진을 보고 지어내지 않는다.
        """
        self.last_latency_ms = None
        answer, ms = self._send(INSTRUCTIONS, _prompt(question, memories), frame)
        self.last_latency_ms = ms
        return answer

    def describe(self, frame: Frame) -> str | None:
        """기억 검색에 쓸 장면 설명을 한두 문장으로 돌려준다. 실패하면 None.

        대답을 읽은 뒤 다른 스레드에서 부르므로 ``last_latency_ms`` 는 건드리지 않는다.
        건드리면 그사이 시작된 다음 질문의 지연이 이 값으로 덮인다.
        """
        scene, _ = self._send(SCENE_INSTRUCTIONS, "이 장면을 기록해 줘.", frame)
        return scene

    def _send(
        self, instructions: str, text: str, frame: Frame | None
    ) -> tuple[str | None, float | None]:
        """요청 한 번을 보내고 (답, 왕복 ms) 를 돌려준다. 실패하거나 답이 비었으면 답은 None."""
        content: list[dict] = [{"type": "input_text", "text": text}]
        if frame is not None:
            content.append(
                {"type": "input_image", "image_url": encode_image(frame.image), "detail": "auto"}
            )
        # monotonic 은 윈도우에서 15.6ms 단위로만 움직여서, 왕복은 perf_counter 로 잰다.
        asked_at = time.perf_counter()
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                input=[{"role": "user", "content": content}],
                reasoning={"effort": "none"},  # 생각하는 단계를 건너뛰어 첫 답을 빨리 받는다
                max_output_tokens=self.max_tokens,
            )
        except OpenAIError as e:
            # 대답을 못 만들어도 음성 루프는 멈추지 않는다. 그래프가 대신 사과 문장을 읽는다.
            log.warning("VLM 호출 실패: %s", e)
            return None, None
        ms = (time.perf_counter() - asked_at) * 1000
        return response.output_text.strip() or None, ms


class FakeVLMClient:
    """정해 둔 답을 차례로 주고, 받은 질문을 ``asked`` 에 모아 둔다.

    답이 하나 남으면 그 답을 계속 준다. None 을 넣어 두면 실패한 호출을 흉내 낸다.
    delay 초만큼 기다렸다가 답한다("잠깐만요" 시험용). describe() 는 늘 scene 을 준다.
    """

    def __init__(
        self,
        answers: Iterable[str | None] = ("컵이에요.",),
        scene: str | None = "책상 위에 흰 컵이 있다.",
        delay: float = 0.0,
    ) -> None:
        self._answers = list(answers)
        self.scene = scene
        self.delay = delay
        self.asked: list[tuple[Frame | None, str, list[MemoryHit] | None]] = []
        self.described: list[Frame] = []
        self.last_latency_ms: float | None = None

    def ask(
        self, frame: Frame | None, question: str, memories: list[MemoryHit] | None = None
    ) -> str | None:
        self.asked.append((frame, question, memories))
        time.sleep(self.delay)
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        self.last_latency_ms = None if answer is None else self.delay * 1000
        return answer

    def describe(self, frame: Frame) -> str | None:
        self.described.append(frame)
        return self.scene


def encode_image(image: np.ndarray, max_side: int = MAX_SIDE) -> str:
    """BGR 사진을 긴 변 max_side 이하의 JPEG data URL 로 바꾼다.

    성분표 OCR 은 로컬에서 원본으로 끝내고 결과만 넘기므로, 여기서 줄여도 글자 인식에는
    영향이 없다.
    """
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1:
        size = (round(width * scale), round(height * scale))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("사진을 JPEG 로 바꾸지 못했다")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")


def _prompt(question: str, memories: list[MemoryHit] | None) -> str:
    """질문 글자를 만든다. 기억 검색을 했으면 기록과 지금 시각을 앞에 붙인다."""
    if memories is None:
        return question
    lines = [f"- {_when(hit.observed_at)}{hit.text}" for hit in memories]
    records = "\n".join(lines) or "- (찾은 기록 없음)"
    now = _when(time.time()).strip()
    return f"{MEMORY_RULE}\n{records}\n\n지금 시각: {now}\n질문: {question}"


def _when(timestamp: float | None) -> str:
    """기록 앞에 붙일 시각. 모델이 "아까"와 "어제"를 가릴 수 있게 한다. 모르면 빈 글자."""
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(timestamp).strftime("%m/%d %H:%M ")
