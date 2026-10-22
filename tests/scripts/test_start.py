"""start.py 의 새 버전 받기 — 언제 건너뛰는지, 받으면 커밋이 바뀌는지, requirements 변경을
알아채는지. 대부분은 git 을 흉내 내고, 하나만 임시 폴더의 실제 git 저장소로 돈다.
실제 git 테스트는 윈도우에서 건너뛴다. git 을 부를 때마다 0.5초쯤 걸려 그 하나만 35초였다.
CI 의 리눅스·맥에서 돈다.
"""

import importlib.util
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "start.py"
_spec = importlib.util.spec_from_file_location("start", _SCRIPT)
start = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(start)


class FakeGit:
    """git 명령마다 미리 정한 결과를 준다. pull 하면 HEAD 가 바뀐다(pulls 가 있을 때)."""

    def __init__(self, branch="main", dirty="", pull_code=0, pulls=False, raises=None):
        self.branch, self.dirty, self.pull_code, self.pulls = branch, dirty, pull_code, pulls
        self.raises = raises
        self.head = "aaa"
        self.calls = []

    def __call__(self, *args, timeout=30.0):
        self.calls.append(args)
        if self.raises:
            raise self.raises
        out, code = "", 0
        if args[:2] == ("rev-parse", "--abbrev-ref"):
            out = self.branch
        elif args[0] == "status":
            out = self.dirty
        elif args == ("rev-parse", "HEAD"):
            out = self.head
        elif args[0] == "pull":
            code = self.pull_code
            if code == 0 and self.pulls:
                self.head = "bbb"
        return subprocess.CompletedProcess(args, code, out, "fatal: 원격에 닿지 못함\n")


def test_skip_reason_only_on_clean_main():
    """main 이고 고친 파일이 없을 때만 받는다."""
    assert start.skip_reason("main", dirty=False) is None
    assert "feat/141" in start.skip_reason("feat/141", dirty=False)
    assert "고친 파일" in start.skip_reason("main", dirty=True)


@pytest.mark.parametrize(
    ("fake", "said"),
    [
        (FakeGit(branch="feat/141"), "main 이 아니라"),
        (FakeGit(dirty=" M scripts/assistant.py"), "고친 파일"),
        (FakeGit(pull_code=1), "원격에 닿지 못함"),
        (FakeGit(raises=FileNotFoundError("git")), "git 을 실행하지 못했습니다"),
        (FakeGit(raises=subprocess.TimeoutExpired("git", 30)), "git 을 실행하지 못했습니다"),
    ],
)
def test_update_skips_and_keeps_current_version(monkeypatch, fake, said):
    """다른 브랜치·고친 파일·인터넷 없음·git 없음이면 건너뛰고 지금 버전으로 켠다."""
    monkeypatch.setattr(start, "git", fake)

    what, before, after = start.update()

    assert what.startswith("건너뜀") and said in what
    assert before is None and after is None


def test_update_reports_up_to_date_or_new(monkeypatch):
    """새 커밋이 없으면 커밋이 그대로, 있으면 바뀐 커밋을 돌려준다."""
    monkeypatch.setattr(start, "git", FakeGit())
    assert start.update() == ("이미 최신입니다", "aaa", "aaa")

    monkeypatch.setattr(start, "git", FakeGit(pulls=True))
    assert start.update() == ("새 버전을 받았습니다", "aaa", "bbb")


@pytest.mark.skipif(sys.platform == "win32", reason="윈도우는 git 이 느려 35초 걸린다")
def test_update_pulls_real_commit_and_notices_requirements(tmp_path, monkeypatch):
    """실제 git: 팀원이 main 에 올린 커밋을 받고, requirements.txt 가 바뀐 것만 알아챈다."""
    for key in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{key}_NAME", "team")
        monkeypatch.setenv(f"GIT_{key}_EMAIL", "team@example.com")

    def git(cwd, *args):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)

    def push(name, text):
        (team / name).write_text(text, encoding="utf-8")
        git(team, "add", name)
        git(team, "commit", "-q", "-m", name)
        git(team, "push", "-q", "origin", "main")

    remote, team, laptop = tmp_path / "remote.git", tmp_path / "team", tmp_path / "laptop"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    git(tmp_path, "init", "-q", "-b", "main", str(team))
    git(team, "remote", "add", "origin", str(remote))
    push("requirements.txt", "numpy\n")
    git(tmp_path, "clone", "-q", str(remote), str(laptop))
    monkeypatch.setattr(start, "ROOT", laptop)

    push("app.py", "print(1)\n")
    what, before, after = start.update()
    assert what == "새 버전을 받았습니다" and (laptop / "app.py").exists()
    assert not start.requirements_changed(before, after)

    push("requirements.txt", "numpy\nhttpx\n")
    _, before, after = start.update()
    assert start.requirements_changed(before, after)


def test_assistant_argv_keeps_device_positions():
    """장치를 비우면 "" 로 자리를 채워, 스피커가 마이크 자리로 밀리지 않게 한다."""
    args = Namespace(mic=None, speaker="Buds", camera=1, db="sqlite:///data/assistant.db")

    argv = start.assistant_argv(args)

    assert argv[1:] == ["", "Buds", "--camera", "1", "--db", "sqlite:///data/assistant.db"]
    assert argv[0].endswith("assistant.py")
