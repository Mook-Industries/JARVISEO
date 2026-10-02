"""클라우드 VLM 호출.

하이브리드 온디바이스의 경계선이 이 파일이다
---------------------------------------------
이미지가 네트워크로 나가는 곳은 여기뿐이다. 그리고 나가는 것은
**질문한 시점의 1프레임**뿐이다. 영상 스트림이 아니다.

STT 도 OpenAI 로 나가지만, 호출어 뒤의 발화 한 번만 보낸다.
웨이크워드·객체검출처럼 상시 동작하는 것은 전부 로컬에서 돈다.
상시 동작하는 것을 전부 로컬에 두었기 때문에 "안경이 계속 찍어서
어디론가 보내는 것 아니냐"는 질문에 구조로 답할 수 있다.
이 원칙을 깨면 프로젝트의 프라이버시 근거가 통째로 사라진다.

비용 주의: 팀 공용 API 키 1개를 쓴다. 월 사용 한도를 반드시 걸어둘 것.
"""

from __future__ import annotations

import numpy as np

from jarviseo.types import AllergenJudgement, Frame, Intent, MemoryHit, TargetResolution

__all__ = ["VLMClient"]


class VLMClient:
    """이미지 한 장과 질문을 보내고 답변 문장을 받는다."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 512) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        raise NotImplementedError

    def ask(
        self,
        frame: Frame,
        question: str,
        intent: Intent = Intent.GENERAL,
        target: TargetResolution | None = None,
        allergen: AllergenJudgement | None = None,
        memories: list[MemoryHit] | None = None,
    ) -> str:
        """질문에 답한다. 특화 모듈 결과가 있으면 맥락으로 덧붙인다.

        target 이 있으면 "사용자가 가리킨 것은 화면 왼쪽의 가방입니다"처럼
        좁혀준다. 이게 없으면 VLM 은 화면에 뭐가 여러 개 있을 때
        무엇을 묻는지 판단할 수 없다.

        allergen 이 있으면 판정 결과를 사실로 주고, VLM 에게 재판정을
        시키지 않는다. 알레르기 판정은 규칙이 내리고 VLM 은 문장만 만든다.
        모델이 지어낸 알레르기 판단이 나가면 안 된다.
        """
        raise NotImplementedError

    @staticmethod
    def _encode(image: np.ndarray, max_side: int = 1568) -> str:
        """이미지를 API 에 보낼 수 있게 인코딩한다.

        너무 크면 비용과 지연이 늘어난다. 긴 변 기준으로 줄인 뒤 보낸다.
        단, 성분표 OCR 은 로컬에서 이미 끝난 상태로 결과만 넘기므로
        여기서 줄여도 글자 인식에는 영향이 없다.
        """
        raise NotImplementedError
