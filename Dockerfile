# 팀 개발 환경 통일용 이미지
#
# 무엇이 통일되고 무엇이 안 되는가
# --------------------------------
# 통일되는 것 : 파이썬 버전, pip 패키지, OpenCV·Whisper 가 필요로 하는
#               시스템 라이브러리. "내 노트북에선 되는데"의 대부분이 여기서 사라진다.
# 안 되는 것  : USB 카메라 · 마이크 · 스피커.
#               맥과 윈도우의 Docker 는 리눅스 VM 안에서 돌기 때문에
#               USB 장치를 컨테이너로 넘길 수 없다. 이건 설정 문제가 아니라 구조다.
#
# 그래서 이렇게 나눈다
# -------------------
#   팀장 A  : 네이티브 실행. 안경 카메라와 마이크를 실물로 쓴다.
#   팀원 B·C: 이 컨테이너에서 개발. VideoFileSource / ImageFolderSource 로
#             공용 Drive 에서 받은 영상·이미지를 입력원으로 쓴다.
#
# FrameSource 추상화가 처음부터 이걸 위해 있는 것이다.
# 인수인계서의 "캡처는 네이티브, 처리 서버만 컨테이너" 방침과 같은 이야기다.

FROM python:3.12-slim

# OpenCV 는 libGL, Whisper 는 ffmpeg 바이너리, sounddevice 는 portaudio 가
# 있어야 import 부터 된다. slim 이미지에는 없어서 직접 넣는다.
# 팀원이 윈도우에서 겪던 설치 실패가 대부분 이 줄로 해결된다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libportaudio2 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch 를 CPU 판으로 **먼저** 설치한다.
# 그냥 두면 easyocr·whisper·ultralytics 가 CUDA 판을 끌어오는데,
# 컨테이너에서는 GPU 를 쓰지도 않으면서 이미지만 2GB 넘게 커진다.
# 학습은 Colab 에서 한다.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# opencv-python 과 opencv-python-headless 는 둘 다 cv2 를 설치해서 서로 덮어쓴다.
# requirements.txt 에는 전자가, requirements-dev.txt 에는 후자가 들어 있어
# 그대로 두면 어느 쪽이 남는지가 설치 순서에 달린다.
# 컨테이너에는 화면이 없으므로 headless 로 못박는다.
RUN pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir opencv-python-headless

# src 레이아웃이라 PYTHONPATH 를 잡아준다.
# pip install -e . 를 쓰지 않는 이유는, 소스를 바인드 마운트로 덮어쓸 때
# .egg-info 가 꼬이는 일이 있어서다.
ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["bash"]
