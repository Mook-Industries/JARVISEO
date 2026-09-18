"""프레임 입력원 추상화.

카메라가 1대뿐이다. 팀장이 그 카메라를 쓰는 동안 팀원 두 명은 개발을 못 하는
상황을 막으려고, 프레임을 어디서 얻는지를 이 인터페이스 뒤로 숨긴다.

- 팀장은 USBCamSource 로 실물 카메라를 쓴다.
- 팀원은 VideoFileSource / ImageFolderSource 로 공용 Drive 에서 받은
  녹화 영상·이미지를 쓴다.
- 코드는 어느 쪽이든 똑같이 동작한다.

카메라를 기다릴 필요가 없고, 고장 나도 개발이 멈추지 않고, 테스트 재현성도
확보된다. 카메라 도착이 9/29~10/5 로 밀린 상황에서 특히 중요하다.
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from jarviseo.types import Frame

__all__ = [
    "FrameSource",
    "WebcamSource",
    "USBCamSource",
    "VideoFileSource",
    "ImageFolderSource",
]

# 이미지 폴더를 읽을 때 인정하는 확장자
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _open_capture(index_or_path: int | str) -> cv2.VideoCapture:
    """OS 별로 다른 카메라 백엔드를 골라서 연다.

    윈도우에서 기본 백엔드로 열면 카메라가 잡히기까지 몇 초씩 걸리거나
    해상도 설정이 먹지 않는다. CAP_DSHOW(DirectShow)를 명시해야 한다.
    맥/리눅스에는 CAP_DSHOW 가 없으므로 기본값을 쓴다.

    OS 분기는 반드시 이 파일 안에서만 한다. 크로스 플랫폼 규칙.
    """
    if sys.platform == "win32" and isinstance(index_or_path, int):
        return cv2.VideoCapture(index_or_path, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index_or_path)


class FrameSource(ABC):
    """프레임을 한 장씩 내주는 것들의 공통 인터페이스.

    ``with`` 문으로 쓰면 자원 해제를 신경 쓰지 않아도 된다::

        with USBCamSource(0) as src:
            frame = src.read()
    """

    source_id: str = "unknown"

    @abstractmethod
    def read(self) -> Frame | None:
        """다음 프레임 한 장. 더 읽을 게 없으면 None."""

    def stream(self) -> Iterator[Frame]:
        """프레임이 끝날 때까지 계속 내주는 반복자."""
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def release(self) -> None:
        """열어둔 자원을 정리한다. 하위 클래스에서 필요하면 재정의."""

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


class WebcamSource(FrameSource):
    """노트북 내장 카메라. 안경 카메라가 없을 때의 임시 입력원."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        self.source_id = f"webcam:{index}"
        self._cap = _open_capture(index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"웹캠 {index} 번을 열지 못했습니다. "
                "맥이라면 시스템 설정 > 개인정보 보호 > 카메라 권한을 확인하세요."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok:
            return None
        return Frame(image=image, timestamp=time.monotonic(), source_id=self.source_id)

    def release(self) -> None:
        self._cap.release()


class USBCamSource(FrameSource):
    """안경에 붙인 UVC 규격 USB 카메라. 팀에 1대뿐이고 팀장이 전담한다.

    해상도 기본값이 1920x1080 인 이유가 있다. 640x480 모듈을 기각한 근거가
    "성분표 글자(약 2mm)가 30cm 거리에서 약 4픽셀로 잡혀 OCR 이 물리적으로
    불가능"이었다. 여기서 해상도를 낮추면 그 기각 사유로 되돌아간다.
    """

    def __init__(
        self, index: int = 0, width: int = 1920, height: int = 1080, fps: int = 30
    ) -> None:
        self.source_id = f"usbcam:{index}"
        self._cap = _open_capture(index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"USB 카메라 {index} 번을 열지 못했습니다. "
                "맥북은 USB-A 포트가 없으므로 허브 연결을 먼저 확인하세요."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok:
            return None
        return Frame(image=image, timestamp=time.monotonic(), source_id=self.source_id)

    def release(self) -> None:
        self._cap.release()


class VideoFileSource(FrameSource):
    """녹화된 영상 파일을 프레임으로 풀어준다. 팀원 개발용 필수 입력원.

    timestamp 는 영상 안에서의 경과 시간을 쓴다. 실제 시계가 아니라
    "영상 시작 후 몇 초"이므로, 같은 파일을 몇 번 돌려도 같은 값이 나온다.
    테스트 재현성이 여기서 나온다.
    """

    def __init__(self, path: str | Path, loop: bool = False) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"영상 파일이 없습니다: {self.path}")
        self.source_id = f"video:{self.path.name}"
        self.loop = loop
        self._cap = _open_capture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"영상을 열지 못했습니다: {self.path}")

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok:
            if not self.loop:
                return None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._cap.read()
            if not ok:
                return None
        # 영상 내 경과 시간(ms) → 초
        elapsed = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return Frame(image=image, timestamp=elapsed, source_id=self.source_id)

    def release(self) -> None:
        self._cap.release()


class ImageFolderSource(FrameSource):
    """폴더 안 이미지를 파일명 순으로 한 장씩 내준다.

    평가셋을 돌릴 때 쓴다. 500장을 순회하며 지시 대상 특정 정확도를 재는
    식의 실험이 이 입력원 위에서 돌아간다.
    """

    def __init__(self, folder: str | Path, fps: float = 30.0) -> None:
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(f"폴더가 없습니다: {self.folder}")
        self.paths = sorted(
            p for p in self.folder.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not self.paths:
            raise FileNotFoundError(f"이미지가 한 장도 없습니다: {self.folder}")
        self.source_id = f"folder:{self.folder.name}"
        self._interval = 1.0 / fps if fps > 0 else 0.0
        self._index = 0

    def read(self) -> Frame | None:
        if self._index >= len(self.paths):
            return None
        path = self.paths[self._index]
        # 한글 경로에서 cv2.imread 가 실패하는 것을 피하려고 바이트로 읽어 디코딩한다.
        # 윈도우에서 특히 문제가 된다.
        buffer = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"이미지를 디코딩하지 못했습니다: {path}")
        timestamp = self._index * self._interval
        self._index += 1
        return Frame(image=image, timestamp=timestamp, source_id=f"{self.source_id}/{path.name}")

    @property
    def current_path(self) -> Path | None:
        """방금 읽은 이미지의 경로. 평가 결과를 파일명과 묶을 때 쓴다."""
        if self._index == 0:
            return None
        return self.paths[self._index - 1]

    def release(self) -> None:
        self._index = 0
