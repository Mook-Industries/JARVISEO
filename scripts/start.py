"""노트북에서 자비서를 한 번에 켠다.  담당: 문태현 (배포)

더블클릭: scripts/start.bat(윈도우) · scripts/start.command(맥). 바탕화면에 바로가기.
명령으로 (어느 폴더에서든):
    python scripts/start.py                     # .env 의 마이크·스피커·카메라
    python scripts/start.py 1 3 --camera 0      # assistant.py 와 같은 인자
    python scripts/start.py --no-update         # 새 버전 받기 건너뜀

차례
    1. 새 버전 받기  main 브랜치이고 고친 파일이 없을 때만 git pull --ff-only. 인터넷이 없거나
                     실패하면 지금 버전으로 켠다. .env 에 JARVISEO_AUTO_UPDATE=false 면 안 한다.
                     새 버전을 받았으면 이 스크립트를 새 코드로 다시 켠다.
    2. 패키지       받은 새 버전에서 requirements.txt 가 바뀌었을 때만 pip install -r 로 깐다
    3. 모델         없는 모델만 받는다 (download_models.py)
    4. 점검         check_setup.py 와 같은 점검. 실패가 있으면 여기서 멈추고 할 일을 보여 준다
    5. 켜기         assistant.py 를 같은 프로세스에서 켠다. 점검 때 불러온 패키지를 다시 읽지 않는다
"""

import argparse
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent


def git(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def skip_reason(branch: str, dirty: bool) -> str | None:
    """git pull 을 하지 않을 이유. 없으면 None.

    개발하는 노트북(다른 브랜치, 고치던 파일)에서 켰을 때 작업이 섞이지 않게 한다.
    """
    if branch != "main":
        return f"main 이 아니라 {branch} 브랜치입니다"
    if dirty:
        return "고친 파일이 있습니다 (git status 로 확인)"
    return None


def update() -> tuple[str, str | None, str | None]:
    """git pull 을 해 보고 (한 일, 전 커밋, 뒤 커밋)을 돌려준다. 건너뛰면 커밋은 None."""
    try:
        branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        dirty = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "건너뜀 — git 을 실행하지 못했습니다", None, None
    reason = skip_reason(branch, bool(dirty))
    if reason:
        return f"건너뜀 — {reason}", None, None
    before = git("rev-parse", "HEAD").stdout.strip()
    try:
        # --ff-only: 이 노트북에서 머지 커밋이나 충돌이 생기지 않게 한다. 안 되면 그냥 실패한다.
        pulled = git("pull", "--ff-only", timeout=60.0)
    except subprocess.TimeoutExpired:
        return "건너뜀 — 60초 안에 끝나지 않았습니다 (인터넷 확인)", None, None
    if pulled.returncode != 0:
        lines = (pulled.stderr or pulled.stdout).strip().splitlines()
        return f"건너뜀 — {lines[0] if lines else 'git pull 실패'}", None, None
    after = git("rev-parse", "HEAD").stdout.strip()
    return ("새 버전을 받았습니다" if after != before else "이미 최신입니다"), before, after


def requirements_changed(before: str, after: str) -> bool:
    return git("diff", "--quiet", before, after, "--", "requirements.txt").returncode != 0


def step(title: str) -> None:
    print(f"\n== {title}", flush=True)


def assistant_argv(args: argparse.Namespace) -> list[str]:
    """점검한 장치·DB 그대로 assistant.py 에 넘긴다. 빈 장치는 ""(= 기본 장치)로 자리를 채운다."""
    return [
        str(SCRIPTS / "assistant.py"),
        args.mic or "",
        args.speaker or "",
        "--camera",
        str(args.camera),
        "--db",
        args.db,
    ]


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="새 버전 받기 → 모델 → 점검 → assistant.py",
        epilog="나머지 인자(마이크 스피커 --camera --db --offline)는 check_setup.py 와 같다.",
    )
    parser.add_argument("--no-update", action="store_true", help="새 버전 받기(git pull) 건너뜀")
    ours, rest = parser.parse_known_args()
    # data/assistant.db · yolov8n.pt 가 지금 폴더 기준 경로라 레포 맨 위에서 돌게 한다.
    os.chdir(ROOT)
    try:
        from jarviseo import config
    except ImportError:
        sys.exit("jarviseo 를 불러오지 못했습니다. conda 환경을 켜고 pip install -e . 를 하세요.")

    step("1/5 새 버전 받기")
    if ours.no_update:
        print("건너뜀 — --no-update")
    elif not config.AUTO_UPDATE:
        print("건너뜀 — .env 의 JARVISEO_AUTO_UPDATE=false (지금 버전에 고정)")
    else:
        what, before, after = update()
        print(what)
        if before and after and before != after:
            step("2/5 패키지")
            if requirements_changed(before, after):
                pip = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
                if subprocess.run(pip).returncode != 0:
                    sys.exit("패키지 설치에 실패했습니다. 위 메시지를 확인하세요.")
            else:
                print("requirements.txt 그대로")
            # 이 파일과 불러온 모듈이 옛 코드라 새 코드로 처음부터 다시 켠다.
            print("새 코드로 다시 켭니다")
            again = [sys.executable, str(Path(__file__).resolve()), "--no-update", *rest]
            sys.exit(subprocess.run(again).returncode)

    # 1 에서 새 버전을 받지 않았으면 2 는 할 일이 없다.
    sys.path.insert(0, str(SCRIPTS))
    import check_setup
    import download_models

    step("3/5 모델")
    if download_models.download_all():
        print("받지 못한 모델이 있습니다. 점검에서 꼭 필요한 것인지 봅니다.")

    step("4/5 점검")
    args = check_setup.parse_args(rest)
    results = check_setup.run_checks(args)
    check_setup.print_report(results)
    if any(r.level == check_setup.FAIL for r in results):
        sys.exit(1)

    step(f"5/5 켜기 (여기까지 {time.perf_counter() - started:.0f}초)")
    sys.argv = assistant_argv(args)
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
