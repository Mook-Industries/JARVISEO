"""클라우드 서버가 쓸 API 키와 접속 비밀번호를 AWS Parameter Store 에 넣는다.  담당: 문태현 (배포)

서버는 배포할 때마다 Parameter Store 의 /jarviseo/ 아래 값을 받아 쓴다(deploy/update.sh).
키는 이 노트북의 .env 에서 읽어 SecureString(암호화)으로 넣는다. Git · GitHub · SSM 명령 기록에
남지 않고, 화면에도 이름만 찍는다.

접속 비밀번호(Caddy 임시 로그인, #179)는 없으면 새로 만들어 넣고 **이 화면에 한 번만** 보여 준다.
이미 있으면 그대로 둔다(--new-password 면 새로 만든다). 다시 보려면 AWS 콘솔 → Systems Manager →
Parameter Store → /jarviseo/BASIC_AUTH_PASSWORD → Show decrypted value.

    aws login --profile jarviseo --region ap-northeast-2
    python scripts/put_cloud_secrets.py
    python scripts/put_cloud_secrets.py --new-password      # 접속 비밀번호 바꾸기

넣은 뒤 다음 배포부터 서버에 반영된다(Actions → Deploy → Run workflow 로 바로 돌릴 수 있다).
"""

import argparse
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "/jarviseo/"
# 서버의 팀 웹이 쓰는 키. 이름은 .env 와 같게 둔다(update.sh 가 그대로 환경변수로 넘긴다).
KEYS = ["OPENAI_API_KEY", "GOOGLE_VISION_API_KEY", "JARVISEO_FOOD_QR_API_KEY"]
PASSWORD = "BASIC_AUTH_PASSWORD"
WINDOWS_AWS = Path(r"C:\Program Files\Amazon\AWSCLIV2\aws.exe")


def aws(*args: str, profile: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    exe = shutil.which("aws") or (str(WINDOWS_AWS) if WINDOWS_AWS.is_file() else "aws")
    cmd = [exe, *args, "--profile", profile, "--region", "ap-northeast-2"]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def put(name: str, value: str, profile: str) -> None:
    aws("ssm", "put-parameter", "--name", PREFIX + name, "--type", "SecureString",
        "--value", value, "--overwrite", "--tier", "Standard", profile=profile)  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description="서버용 키 · 접속 비밀번호를 Parameter Store 에")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--profile", default="jarviseo", help="aws login 으로 만든 프로필")
    parser.add_argument("--new-password", action="store_true", help="접속 비밀번호를 새로 만든다")
    args = parser.parse_args()

    env = dotenv_values(args.env_file)
    missing = [k for k in KEYS if not (env.get(k) or "").strip()]
    if missing:
        sys.exit(f"{args.env_file} 에 값이 없다: {', '.join(missing)}")
    if aws("sts", "get-caller-identity", profile=args.profile, check=False).returncode != 0:
        sys.exit(f"AWS 에 로그인되어 있지 않다: aws login --profile {args.profile}")

    for key in KEYS:
        put(key, env[key].strip(), args.profile)
        print(f"넣음 {PREFIX}{key}")

    exists = aws("ssm", "get-parameter", "--name", PREFIX + PASSWORD,
                 profile=args.profile, check=False).returncode == 0  # fmt: skip
    if exists and not args.new_password:
        print(f"그대로 둠 {PREFIX}{PASSWORD} (바꾸려면 --new-password)")
        return
    password = secrets.token_urlsafe(18)
    put(PASSWORD, password, args.profile)
    print(f"넣음 {PREFIX}{PASSWORD}")
    print(f"\n접속 아이디: jarviseo\n접속 비밀번호: {password}")
    print("(이 화면에서만 보여 준다. 팀에는 따로 전한다)")


if __name__ == "__main__":
    main()
