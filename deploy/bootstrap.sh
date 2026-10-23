#!/bin/bash
# 클라우드 서버에 AWS CLI v2 를 깐다. update.sh 가 처음 한 번 부른다.  담당: 문태현 (배포)
#
# 서버는 AWS CLI 로 두 가지를 한다. 둘 다 키 없이 서버의 IAM 역할(jarviseo-ec2-ssm)로 된다.
#   - Parameter Store(/jarviseo/)에서 API 키 · 접속 비밀번호 받기
#   - ECR 에 로그인해 팀 웹 이미지 받기
set -euo pipefail

if command -v aws >/dev/null 2>&1; then
  aws --version
  exit 0
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q unzip curl
tmp=$(mktemp -d)
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "$tmp/awscli.zip"
unzip -q "$tmp/awscli.zip" -d "$tmp"
"$tmp/aws/install"
rm -rf "$tmp"
aws --version
