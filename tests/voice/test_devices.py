"""find_device 가 번호·이름 일부로 마이크·스피커를 고르는지 확인한다. 실제 장치 없이 돈다."""

import pytest

from jarviseo.voice.devices import find_device

MME, WASAPI = 0, 2


def device(index, name, hostapi, inputs, outputs):
    """sounddevice.query_devices() 가 주는 모양."""
    return {
        "index": index,
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
    }


# 윈도우처럼 같은 장치가 호스트 API 마다 다른 번호로 나온다.
DEVICES = [
    device(0, "Microsoft 사운드 매퍼 - Input", MME, 2, 0),
    device(1, "마이크(USB PnP Sound Device)", MME, 1, 0),
    device(3, "스피커(Synaptics SmartAudio HD)", MME, 0, 2),
    device(4, "헤드폰(Buds2 Pro)", MME, 0, 2),
    device(17, "마이크(USB PnP Sound Device)", WASAPI, 1, 0),
    device(20, "Speakers (Realtek)", WASAPI, 0, 2),
]


def pick(query, kind):
    return find_device(query, kind, devices=DEVICES, default_hostapi=MME)


@pytest.mark.parametrize("query", [None, "", "  "])
def test_empty_query_means_default_device(query):
    """비어 있으면 None(윈도우·맥 기본 장치)을 돌려준다."""
    assert pick(query, "input") is None


def test_number_is_used_as_is():
    """숫자는 그 번호 그대로 쓴다. 글자로 적은 숫자도 같다(.env 값은 글자다)."""
    assert pick(4, "output") == 4
    assert pick("4", "output") == 4


def test_number_with_wrong_direction_is_rejected():
    """스피커 번호를 마이크로 주면 없는 번호로 본다."""
    with pytest.raises(ValueError, match="마이크 4번이 없습니다"):
        pick(4, "input")


def test_name_part_matches_ignoring_case():
    """이름 일부(대소문자 무시)로 고른다."""
    assert pick("buds", "output") == 4
    assert pick("usb", "input") == 1


def test_same_name_prefers_default_hostapi():
    """같은 이름이 여러 호스트 API 에 있으면 기본 호스트 API(MME) 쪽 번호를 고른다."""
    assert pick("USB PnP", "input") == 1
    assert find_device("USB PnP", "input", devices=DEVICES, default_hostapi=WASAPI) == 17


def test_falls_back_to_other_hostapi():
    """기본 호스트 API 에 없는 이름이면 다른 호스트 API 에서라도 고른다."""
    assert pick("Realtek", "output") == 20


def test_unknown_name_lists_choices():
    """못 찾으면 고를 수 있는 이름을 알려 준다."""
    with pytest.raises(ValueError, match="고를 수 있는 것: .*Buds2 Pro"):
        pick("에어팟", "output")
