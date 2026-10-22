# 팀 개발 환경 통일용 이미지.  담당: 문태현 (배포)
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
#   네이티브  : 안경 카메라와 마이크를 실물로 쓸 때.
#   컨테이너  : 그 외. VideoFileSource / ImageFolderSource 로
#              공용 Drive 에 올라온 영상·이미지를 입력원으로 쓴다.
#
# FrameSource 추상화가 처음부터 이걸 위해 있는 것이다.
# 인수인계서의 "캡처는 네이티브, 처리 서버만 컨테이너" 방침과 같은 이야기다.

FROM python:3.12-slim

# OpenCV·MediaPipe 는 libGL, sounddevice 는 portaudio 가 있어야 import 부터 된다.
# slim 이미지에는 없어서 직접 넣는다. (ffmpeg 는 로컬 Whisper 를 뺀 뒤로 쓰는 곳이 없어 뺐다.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 버전은 requirements.lock(requirements.txt + requirements-dev.txt 를 uv 로 푼 것) 그대로 깐다.
# requirements.txt 는 하한(>=)만 있어서 빌드할 때마다 최신 버전이 들어와, 어제 되던 이미지가
# 오늘 깨질 수 있다. 9/19 에 만든 뒤 requirements 가 여러 번 바뀌어도 아무도 빌드를 확인하지 않았다.
#
# torch 만 CPU 판으로 **먼저** 따로 깐다. 잠금 파일은 리눅스에서 PyPI 의 CUDA 판 torch 와
# nvidia-*·cuda-* 패키지(수 GB)를 고르는데, 컨테이너는 GPU 를 쓰지 않는다. 학습은 Colab 에서 한다.
# 그래서 같은 버전의 CPU 판을 pytorch 저장소에서 받고, 잠금 파일에서는 그 줄들을 빼고 깐다.
COPY requirements.lock ./
RUN set -eux; \
    torch="$(grep -E '^torch==' requirements.lock | cut -d' ' -f1)"; \
    vision="$(grep -E '^torchvision==' requirements.lock | cut -d' ' -f1)"; \
    pip install --no-cache-dir "$torch" "$vision" --index-url https://download.pytorch.org/whl/cpu; \
    grep -vE "^(torch|torchvision)==|^cuda-|^triton|^nvidia-.*sys_platform == 'linux'" requirements.lock \
        > /tmp/requirements-cpu.lock; \
    pip install --no-cache-dir -r /tmp/requirements-cpu.lock; \
    rm /tmp/requirements-cpu.lock

# opencv-python · opencv-python-headless · opencv-contrib-python 은 셋 다 cv2 를 설치해서 서로 덮어쓴다.
# 잠금 파일에 셋이 다 있다(앱 · CI · mediapipe 가 각각 끌어온다). 어느 쪽이 남는지가 설치 순서에
# 달리지 않게, 셋 중 가장 넓은 contrib 를 마지막에 같은 버전으로 다시 깐다. 위에서 libGL 을 넣었으므로
# 화면 없는 컨테이너에서도 import 된다.
RUN contrib="$(grep -E '^opencv-contrib-python==' requirements.lock | cut -d' ' -f1)" \
    && pip install --no-cache-dir --no-deps --force-reinstall "$contrib"

# src 레이아웃이라 PYTHONPATH 를 잡아준다.
# pip install -e . 를 쓰지 않는 이유는, 소스를 바인드 마운트로 덮어쓸 때
# .egg-info 가 꼬이는 일이 있어서다.
ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["bash"]
