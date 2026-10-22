"""마이크·스피커를 번호나 장치 이름 일부로 고른다.  담당: 문태현 (배포)

sounddevice 장치 번호는 노트북마다, USB 를 꽂는 순서마다 바뀐다. 그래서
.env 에 "USB"·"Buds" 처럼 이름 일부를 적어 두고, 켤 때마다 지금 번호를 찾는다.

    JARVISEO_MIC=USB          # 이름에 USB 가 들어간 마이크
    JARVISEO_SPEAKER=3        # 숫자면 그 번호 그대로
    (비우면 윈도우·맥 기본 장치)

윈도우는 같은 장치가 MME·DirectSound·WASAPI·WDM-KS 로 네 번씩 나온다. 이름이 같으면
sounddevice 기본 호스트 API(윈도우는 MME) 쪽 번호를 고른다. 번호로 넘기던 지금까지와 같은 쪽이다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

Kind = Literal["input", "output"]

_CHANNELS = {"input": "max_input_channels", "output": "max_output_channels"}
_LABEL = {"input": "마이크", "output": "스피커"}


def find_device(
    query: str | int | None,
    kind: Kind,
    devices: Sequence[dict[str, Any]] | None = None,
    default_hostapi: int | None = None,
) -> int | None:
    """query 에 맞는 장치 번호를 돌려준다. 비어 있으면 None(기본 장치).

    query 가 숫자면 그 번호가 kind 방향(입력/출력)으로 쓸 수 있는지만 확인한다.
    글자면 이름에 그 글자가 들어간(대소문자 무시) 장치 중 기본 호스트 API 쪽 첫 번째를 고른다.
    못 찾으면 고를 수 있는 이름을 담아 ValueError 를 낸다.
    devices·default_hostapi 는 테스트에서만 넘긴다. 없으면 sounddevice 에 묻는다.
    """
    if query is None or str(query).strip() == "":
        return None
    if devices is None:
        import sounddevice as sd

        devices = list(sd.query_devices())
        default_hostapi = sd.default.hostapi
    usable = [d for d in devices if d[_CHANNELS[kind]] > 0]
    label = _LABEL[kind]
    text = str(query).strip()

    if text.isdigit():
        index = int(text)
        if any(d["index"] == index for d in usable):
            return index
        raise ValueError(f"{label} {index}번이 없습니다. {_names(usable, default_hostapi)}")

    matches = [d for d in usable if text.lower() in d["name"].lower()]
    if not matches:
        choices = _names(usable, default_hostapi)
        raise ValueError(f"이름에 '{text}' 가 들어간 {label}가 없습니다. {choices}")
    preferred = [d for d in matches if d.get("hostapi") == default_hostapi]
    return (preferred or matches)[0]["index"]


def device_name(index: int | None, kind: Kind) -> str:
    """시작할 때 어느 장치를 열었는지 찍으려고 쓴다. None 이면 기본 장치 이름."""
    import sounddevice as sd

    info = sd.query_devices(index, kind=kind)
    return f"{info['index']}번 {info['name']}"


def _names(devices: Sequence[dict[str, Any]], default_hostapi: int | None) -> str:
    """고를 수 있는 이름. 같은 장치가 호스트 API 마다 되풀이되지 않게 기본 쪽만 보여 준다."""
    preferred = [d for d in devices if d.get("hostapi") == default_hostapi] or devices
    shown = sorted({" ".join(d["name"].split()) for d in preferred})
    return "고를 수 있는 것: " + ", ".join(shown) if shown else "고를 수 있는 장치가 없습니다."
