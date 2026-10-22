"""assistant.py 를 켜기 전에 필요한 것을 한 번에 점검한다.  담당: 문태현 (배포)

빠진 것마다 무엇이 문제이고 무엇을 하면 되는지 한 줄씩 알려 준다. 고칠 것이 하나라도 있으면
끝 코드 1 로 끝나서, start.py 가 assistant.py 를 켜지 않는다.

실행 (레포 맨 위 폴더에서, assistant.py 에 줄 값 그대로):
    python scripts/check_setup.py                    # .env 의 마이크·스피커·카메라 (비면 기본 장치)
    python scripts/check_setup.py 1 3 --camera 0     # 1번 마이크, 3번 스피커, 0번 웹캠
    python scripts/check_setup.py USB Buds           # 이름에 USB 가 든 마이크, Buds 가 든 스피커
    python scripts/check_setup.py --db postgresql+psycopg://jarviseo:jarviseo@localhost:5432/jarviseo
    python scripts/check_setup.py --offline          # OpenAI 연결은 건너뜀 (인터넷 없이 확인할 때)

보는 것
    파이썬    3.12 인지, 레포 맨 위 폴더에서 돌렸는지
    패키지    requirements.txt 의 패키지를 불러올 수 있는지
    .env      파일이 있는지, OPENAI_API_KEY 가 있는지
    모델      호출어 모델, openWakeWord 특징 모델, EasyOCR 한국어 모델, yolov8n.pt, 손 관절 모델
    장치      마이크·스피커 번호가 맞는지, 카메라에서 사진 한 장이 나오는지
    연결      OpenAI 키로 VLM 모델을 부를 수 있는지
    DB        --db 에 붙을 수 있는지 (SQLite 면 파일이 없을 때 새로 만든다)

[통과] 문제 없음  [주의] 없어도 켜지지만 기능 하나가 빠짐  [실패] 고쳐야 켜짐
"""

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# assistant.py 의 기본값과 같아야 한다. 점검한 DB 와 실제로 쓰는 DB 가 달라지면 점검이 의미가 없다.
DEFAULT_DB = "sqlite:///data/assistant.db"
DOWNLOAD = "python scripts/download_models.py"

OK, WARN, FAIL = "통과", "주의", "실패"


@dataclass
class Check:
    """점검 하나의 결과. fix 는 실패·주의일 때 사용자가 할 일 한 줄."""

    name: str
    level: str
    detail: str = ""
    fix: str = ""


def check_python() -> list[Check]:
    results = []
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info[:2] == (3, 12):
        results.append(Check("파이썬", OK, version))
    else:
        results.append(
            Check(
                "파이썬",
                FAIL,
                version,
                "conda create -n JARVISEO python=3.12 로 만든 환경에서 돌리세요",
            )
        )
    # data/assistant.db, yolov8n.pt 는 지금 폴더 기준 경로라 다른 폴더에서 켜면 엉뚱한 곳을 본다.
    here = str(Path.cwd())
    if Path("pyproject.toml").is_file() and Path("scripts/assistant.py").is_file():
        results.append(Check("실행 폴더", OK, here))
    else:
        results.append(Check("실행 폴더", FAIL, here, "레포 맨 위 폴더(JARVISEO)로 cd 하세요"))
    return results


# (불러올 이름, pip 이름). 불러오는 이름과 설치 이름이 다른 것이 있어 둘 다 적는다.
PACKAGES = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("dotenv", "python-dotenv"),
    ("ultralytics", "ultralytics"),
    ("onnxruntime", "onnxruntime"),
    ("easyocr", "easyocr"),
    ("httpx", "httpx"),
    ("langgraph", "langgraph"),
    ("openai", "openai"),
    ("sqlalchemy", "sqlalchemy"),
    ("psycopg", "psycopg[binary]"),
    ("pgvector", "pgvector"),
    ("sounddevice", "sounddevice"),
    ("soundfile", "soundfile"),
    ("webrtcvad", "webrtcvad-wheels"),
    ("openwakeword", "openwakeword"),
]
# 없어도 assistant.py 는 켜지고 기능 하나가 약해지는 것. (불러올 이름, pip 이름, 없으면 어떻게 되나)
OPTIONAL_PACKAGES = [
    ("mediapipe", "mediapipe", "손 방향 단서 없이 가운데 우선으로 고른다"),
    ("zxingcpp", "zxing-cpp", "바코드를 cv2 로만 읽어 잘 못 읽는다"),
]


def check_packages() -> list[Check]:
    missing = []
    for module, pip_name in PACKAGES:
        try:
            importlib.import_module(module)
        except Exception:  # noqa: BLE001 — DLL 못 찾음 같은 ImportError 아닌 오류도 설치 문제다
            missing.append(pip_name)
    results = []
    if missing:
        results.append(
            Check("패키지", FAIL, f"없음: {', '.join(missing)}", "pip install -r requirements.txt")
        )
    else:
        results.append(Check("패키지", OK, f"{len(PACKAGES)}개"))
    try:
        importlib.import_module("jarviseo")
        results.append(Check("jarviseo 패키지", OK))
    except ImportError:
        results.append(Check("jarviseo 패키지", FAIL, "불러올 수 없음", "pip install -e ."))
    for module, pip_name, without in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(module)
            results.append(Check(pip_name, OK))
        except Exception:  # noqa: BLE001
            results.append(Check(pip_name, WARN, f"없음 — {without}", f"pip install {pip_name}"))
    return results


def check_env() -> list[Check]:
    from jarviseo import config

    results = []
    env = config.PROJECT_ROOT / ".env"
    if env.is_file():
        results.append(Check(".env", OK))
    else:
        results.append(Check(".env", FAIL, "파일 없음", "copy .env.example .env 뒤 값을 채우세요"))
    if config.OPENAI_API_KEY:
        results.append(Check("OPENAI_API_KEY", OK, f"...{config.OPENAI_API_KEY[-4:]}"))
    else:
        results.append(
            Check("OPENAI_API_KEY", FAIL, "비어 있음", ".env 에 OPENAI_API_KEY=sk-... 를 적으세요")
        )
    if not config.ENABLE_RECALL:
        results.append(
            Check(
                "기억 검색",
                WARN,
                "꺼짐 — '아까 본 거 뭐였지?'에 답하지 못한다",
                ".env 에 JARVISEO_ENABLE_RECALL=true",
            )
        )
    if not config.FOOD_QR_API_KEY:
        results.append(
            Check(
                "푸드QR",
                WARN,
                "키 없음 — 바코드 조회 없이 성분표 OCR 로 간다",
                ".env 에 JARVISEO_FOOD_QR_API_KEY",
            )
        )
    return results


def check_models() -> list[Check]:
    from jarviseo import config

    results = []
    wake = Path(config.WAKEWORD_MODEL)
    if wake.suffix != ".onnx" or wake.is_file():
        results.append(Check("호출어 모델", OK, str(wake)))
    else:
        results.append(Check("호출어 모델", FAIL, f"없음: {wake}", DOWNLOAD))

    # 커스텀 .onnx 를 쓰면 openWakeWord 가 특징 모델(멜스펙트로그램·임베딩)을 알아서 받지 않는다.
    try:
        import openwakeword

        resources = Path(openwakeword.__file__).parent / "resources" / "models"
        need = ["melspectrogram.onnx", "embedding_model.onnx"]
        missing = [name for name in need if not (resources / name).is_file()]
        if missing:
            results.append(
                Check("openWakeWord 특징 모델", FAIL, f"없음: {', '.join(missing)}", DOWNLOAD)
            )
        else:
            results.append(Check("openWakeWord 특징 모델", OK))
    except ImportError:
        pass  # 패키지 점검에서 이미 실패로 나온다

    # EasyOCR 은 처음 읽을 때 받는다. 인터넷이 느린 곳이면 첫 성분 질문이 그만큼 늦어진다.
    easyocr_dir = Path(os.getenv("EASYOCR_MODULE_PATH", Path.home() / ".EasyOCR")) / "model"
    need = ["craft_mlt_25k.pth", "korean_g2.pth"]
    missing = [name for name in need if not (easyocr_dir / name).is_file()]
    if missing:
        detail = f"없음: {', '.join(missing)} — 첫 성분 질문 때 받는다"
        results.append(Check("EasyOCR 모델", WARN, detail, DOWNLOAD))
    else:
        results.append(Check("EasyOCR 모델", OK))

    # ObjectDetector 가 지금 폴더의 yolov8n.pt 를 읽는다. 없으면 처음 켤 때 받는다.
    if Path("yolov8n.pt").is_file():
        results.append(Check("yolov8n.pt", OK))
    else:
        results.append(Check("yolov8n.pt", WARN, "없음 — 처음 켤 때 인터넷으로 받는다", DOWNLOAD))

    if config.HAND_LANDMARKER_PATH.is_file():
        results.append(Check("손 관절 모델", OK))
    else:
        results.append(Check("손 관절 모델", WARN, "없으면 손 방향 단서 없이 돈다", DOWNLOAD))
    return results


def check_audio(mic: str | None, speaker: str | None) -> list[Check]:
    try:
        from jarviseo.voice.devices import device_name, find_device
    except Exception:  # noqa: BLE001
        return []  # 패키지 점검에서 이미 실패로 나온다
    results = []
    for label, query, kind, key in (
        ("마이크", mic, "input", "JARVISEO_MIC"),
        ("스피커", speaker, "output", "JARVISEO_SPEAKER"),
    ):
        try:
            results.append(Check(label, OK, device_name(find_device(query, kind), kind)))
        except Exception as e:  # noqa: BLE001
            # 못 찾으면 ValueError(고를 수 있는 이름이 들어 있다), 기본 장치가 없으면 PortAudioError
            shown = f"'{query}'" if query else "기본 장치"
            fix = f".env 의 {key} 에 번호나 위 이름 일부를 적으세요"
            results.append(Check(label, FAIL, f"{shown}: {e}", fix))
    return results


def check_camera(index: int) -> list[Check]:
    try:
        from jarviseo.capture import WebcamSource
    except Exception:  # noqa: BLE001
        return []
    fix = "--camera 번호를 바꾸거나 USB 를 다시 꽂으세요"
    try:
        with WebcamSource(index) as source:
            frame = source.read()
    except Exception as e:  # noqa: BLE001
        return [Check("카메라", FAIL, f"{index}번: {e}", fix)]
    if frame is None:
        return [Check("카메라", FAIL, f"{index}번: 열렸지만 사진이 안 나옴", fix)]
    height, width = frame.image.shape[:2]
    return [Check("카메라", OK, f"{index}번 {width}x{height}")]


def check_openai() -> list[Check]:
    from jarviseo import config

    if not config.OPENAI_API_KEY:
        return []  # .env 점검에서 이미 실패로 나온다
    import openai

    # 재시도를 끄고 짧게 끊는다. 인터넷이 끊겼을 때 점검이 수십 초 멈추면 안 된다.
    client = openai.OpenAI(api_key=config.OPENAI_API_KEY, timeout=10.0, max_retries=0)
    try:
        client.models.retrieve(config.VLM_MODEL)
    except openai.AuthenticationError:
        return [Check("OpenAI", FAIL, "키가 틀림", ".env 의 OPENAI_API_KEY 를 다시 확인하세요")]
    except openai.NotFoundError:
        detail = f"모델 {config.VLM_MODEL} 을 이 키로 쓸 수 없음"
        return [Check("OpenAI", FAIL, detail, "키의 프로젝트 권한을 확인하세요")]
    except (openai.APIConnectionError, openai.APITimeoutError):
        return [Check("OpenAI", FAIL, "연결 안 됨", "인터넷(와이파이·핫스팟)을 확인하세요")]
    except openai.APIError as e:
        return [Check("OpenAI", FAIL, str(e), "잠시 뒤 다시 돌려 보세요")]
    return [Check("OpenAI", OK, f"{config.VLM_MODEL} 사용 가능")]


def check_db(url: str) -> list[Check]:
    try:
        from sqlalchemy import make_url, text

        from jarviseo.memory import MemoryStore
    except Exception:  # noqa: BLE001
        return []
    where = make_url(url).render_as_string(hide_password=True)
    try:
        with MemoryStore(database_url=url).engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        reason = str(getattr(e, "orig", None) or e).strip().splitlines()[0]
        if url.startswith("postgresql"):
            fix = "도커를 켜고 docker compose up -d db, 아니면 --db 를 빼고 SQLite 로"
        else:
            fix = "data 폴더가 있는지, 다른 프로그램이 DB 파일을 잡고 있지 않은지 확인하세요"
        return [Check("DB", FAIL, f"{where}: {reason}", fix)]
    return [Check("DB", OK, where)]


def run_checks(args: argparse.Namespace) -> list[Check]:
    results = check_python() + check_packages()
    # jarviseo 를 못 불러오면 뒤 점검은 모두 같은 이유로 실패한다. 먼저 그것부터 고치게 한다.
    if any(r.name == "jarviseo 패키지" and r.level == FAIL for r in results):
        return results
    results += check_env() + check_models()
    results += check_audio(args.mic, args.speaker) + check_camera(args.camera)
    if not args.offline:
        results += check_openai()
    results += check_db(args.db)
    return results


def print_report(results: list[Check]) -> None:
    for r in results:
        line = f"[{r.level}] {r.name}"
        if r.detail:
            line += f" — {r.detail}"
        print(line)
        if r.fix and r.level != OK:
            print(f"        → {r.fix}")
    fails = sum(r.level == FAIL for r in results)
    warns = sum(r.level == WARN for r in results)
    print()
    if fails:
        print(f"실패 {fails}개를 고쳐야 켤 수 있습니다 (주의 {warns}개).")
    else:
        print(f"켤 준비가 됐습니다 (주의 {warns}개).")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """assistant.py 와 같은 인자를 읽고, 빠진 장치는 .env 값으로 채운다. start.py 도 쓴다."""
    parser = argparse.ArgumentParser(description="assistant.py 를 켜기 전 점검")
    parser.add_argument(
        "mic", nargs="?", help="마이크 번호나 이름 일부 (기본 .env 의 JARVISEO_MIC)"
    )
    parser.add_argument(
        "speaker", nargs="?", help="스피커 번호나 이름 일부 (기본 .env 의 JARVISEO_SPEAKER)"
    )
    parser.add_argument("--camera", type=int, default=None, help="웹캠 번호 (기본 .env 의 값)")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"턴을 남길 DB (기본 {DEFAULT_DB})")
    parser.add_argument("--offline", action="store_true", help="OpenAI 연결 점검을 건너뜀")
    args = parser.parse_args(argv)
    # config 를 맨 위에서 불러오지 않는다. jarviseo 가 설치 안 된 노트북에서도 점검은 떠야 한다.
    try:
        from jarviseo import config

        mic, speaker, camera = config.MIC_DEVICE, config.SPEAKER_DEVICE, config.CAMERA_INDEX
    except ImportError:
        mic, speaker, camera = None, None, 0
    args.mic = args.mic or mic
    args.speaker = args.speaker or speaker
    args.camera = camera if args.camera is None else args.camera
    return args


def main() -> None:
    results = run_checks(parse_args())
    print_report(results)
    sys.exit(1 if any(r.level == FAIL for r in results) else 0)


if __name__ == "__main__":
    main()
