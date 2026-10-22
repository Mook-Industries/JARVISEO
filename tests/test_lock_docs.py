"""문서에 손으로 적은 패키지 버전이 requirements.lock 과 같은지 본다.

잠금 파일을 다시 만들고 문서 고치기를 잊는 것을 막는다.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_opencv_contrib_matches_lock():
    """README 의 opencv-contrib-python 다시 깔기 줄이 잠금 파일 버전을 쓴다."""
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    pinned = re.search(r"^opencv-contrib-python==(\S+)", lock, re.M).group(1)
    written = re.findall(r"opencv-contrib-python==(\S+)", readme)

    assert written and set(written) == {pinned}
