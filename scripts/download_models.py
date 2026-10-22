"""처음 켜기 전에 모델 파일을 미리 받아 둔다.  담당: 문태현 (배포)

assistant.py 가 처음 실행 때 인터넷으로 받는 파일이 있다. 인터넷이 느린 곳이면 첫 질문이
그만큼 늦어지고, 끊겨 있으면 켜지지 않는다. 설치할 때 집에서 한 번 받아 두려고 둔다.
이미 있는 파일은 건너뛴다. 몇 번을 돌려도 된다.

실행 (레포 맨 위 폴더에서):
    python scripts/download_models.py

받는 것
    호출어 모델          data/models/jarviseo.onnx — 레포 GitHub 릴리스(models-v1) 첨부 파일
    손 관절 모델         data/models/hand_landmarker.task — MediaPipe 공식 주소
    openWakeWord 특징    melspectrogram.onnx · embedding_model.onnx — openwakeword 패키지 폴더.
                         커스텀 .onnx 를 쓰면 openWakeWord 가 알아서 받지 않는다
    EasyOCR 한국어·영어  ~/.EasyOCR/model — 성분표 OCR
    yolov8n.pt           레포 맨 위 폴더 — 가리킴 물체 검출

다 받은 뒤 python scripts/check_setup.py 로 확인한다.
"""

import hashlib
import sys
from pathlib import Path

import httpx

from jarviseo import config

# 호출어 모델은 Git 에 없다(*.onnx 는 .gitignore). 바꿀 때마다 옛 버전이 Git 기록에 쌓이지 않게
# 레포 릴리스에 첨부해 두고 여기서 받는다. git clone 은 릴리스 파일을 받지 않는다.
# 모델을 바꾸면 새 태그(models-v2 …)로 올리고 아래 주소와 SHA256 을 같이 바꾼다.
RELEASE_URL = "https://github.com/Mook-Industries/JARVISEO/releases/download/models-v1"
WAKEWORD_FILE = "jarviseo.onnx"
# r6 모델(205,069바이트). 받다 끊기거나 다른 파일이 올라가 있으면 여기서 걸린다.
WAKEWORD_SHA256 = "4686db10ae2febb837cadccf0e51e10b3bb9bccfdcc4b4820164fb73dd272b46"
HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
    "float16/latest/hand_landmarker.task"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, dest: Path, expected_sha256: str | None = None) -> None:
    """url 을 dest 로 받는다. 끝까지 받은 뒤에만 이름을 바꿔, 끊겨도 반쪽 파일이 남지 않게 한다."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=30.0) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    if expected_sha256 and sha256(part) != expected_sha256:
        part.unlink()
        raise RuntimeError(f"받은 파일이 예상과 다름(SHA256): {url}")
    part.replace(dest)


def get_wakeword() -> str:
    model = Path(config.WAKEWORD_MODEL)
    if model.suffix != ".onnx":
        return f"건너뜀 — 기성 모델 {model} 은 켤 때 openWakeWord 가 받는다"
    if model.is_file():
        return f"있음 {model}"
    if model.name != WAKEWORD_FILE:
        raise RuntimeError(f"릴리스에 없는 모델: {model} (.env 의 JARVISEO_WAKEWORD_MODEL 확인)")
    try:
        fetch(f"{RELEASE_URL}/{WAKEWORD_FILE}", model, WAKEWORD_SHA256)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise RuntimeError(f"릴리스에 {WAKEWORD_FILE} 가 없음: {RELEASE_URL}") from None
        raise
    return f"받음 {model}"


def get_hand_landmarker() -> str:
    dest = config.HAND_LANDMARKER_PATH
    if dest.is_file():
        return f"있음 {dest}"
    fetch(HAND_URL, dest)
    return f"받음 {dest}"


def get_openwakeword_features(target: Path | None = None) -> str:
    """openWakeWord 가 소리를 특징으로 바꿀 때 쓰는 모델 두 개. target 은 테스트에서만 넘긴다."""
    import openwakeword

    target = target or Path(openwakeword.__file__).parent / "resources" / "models"
    got = []
    for feature in openwakeword.FEATURE_MODELS.values():
        # 목록에는 .tflite 주소만 있다. 우리는 onnxruntime 으로 돌리므로 .onnx 를 받는다.
        url = feature["download_url"].replace(".tflite", ".onnx")
        dest = target / url.rsplit("/", 1)[-1]
        if not dest.is_file():
            fetch(url, dest)
            got.append(dest.name)
    return f"받음 {', '.join(got)}" if got else f"있음 {target}"


def get_easyocr(target: Path | None = None) -> str:
    """nutrition/ocr.py 와 같은 언어(ko, en)로 Reader 를 한 번 만들면 없는 모델을 받는다."""
    target = target or Path.home() / ".EasyOCR" / "model"
    # 있으면 Reader 를 만들지 않는다. 모델을 읽는 데 몇 초 걸려 start.py 가 매번 기다리게 된다.
    if all((target / name).is_file() for name in ("craft_mlt_25k.pth", "korean_g2.pth")):
        return f"있음 {target}"
    import easyocr

    easyocr.Reader(["ko", "en"], gpu=False, verbose=False, model_storage_directory=str(target))
    return f"받음 {target}"


def get_yolo() -> str:
    dest = Path("yolov8n.pt")  # pointing/detector.py 가 지금 폴더 기준으로 읽는다
    if dest.is_file():
        return f"있음 {dest}"
    from ultralytics import YOLO

    YOLO(str(dest))  # 없으면 ultralytics 가 공식 릴리스에서 받아 지금 폴더에 둔다
    return f"받음 {dest}"


STEPS = [
    ("호출어 모델", get_wakeword),
    ("손 관절 모델", get_hand_landmarker),
    ("openWakeWord 특징 모델", get_openwakeword_features),
    ("EasyOCR 모델", get_easyocr),
    ("yolov8n.pt", get_yolo),
]


def download_all() -> int:
    """없는 모델을 모두 받고 받지 못한 개수를 돌려준다. start.py 도 이걸 부른다."""
    failed = 0
    for name, step in STEPS:
        print(f"{name} ...", flush=True)
        try:
            print(f"  {step()}")
        except Exception as e:  # noqa: BLE001 — 하나가 실패해도 나머지는 받는다
            failed += 1
            print(f"  실패: {e}")
    return failed


def main() -> None:
    if not Path("scripts/assistant.py").is_file():
        sys.exit("레포 맨 위 폴더(JARVISEO)로 cd 한 뒤 돌리세요.")
    failed = download_all()
    if failed:
        sys.exit(f"\n{failed}개를 받지 못했습니다. 인터넷을 확인하고 다시 돌리세요.")
    print("\n다 받았습니다. python scripts/check_setup.py 로 확인하세요.")


if __name__ == "__main__":
    main()
