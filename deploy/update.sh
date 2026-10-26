#!/bin/bash
# 클라우드 서버에서 새 버전을 띄운다.  담당: 문태현 (배포)
#
# GitHub Actions(.github/workflows/deploy.yml)가 main 에 머지될 때 SSM 으로 부른다. 그 전에
# /opt/jarviseo/repo 를 그 커밋으로 맞추고, 그 커밋으로 빌드해 ECR 에 올린 이미지를 WEB_IMAGE 로 넘긴다.
# 손으로 부를 때는 WEB_IMAGE 를 빼면 ECR 의 latest 를 쓴다.
#     sudo WEB_IMAGE=<계정>.dkr.ecr.ap-northeast-2.amazonaws.com/jarviseo-web:<커밋> bash /opt/jarviseo/repo/deploy/update.sh
#
# 하는 일: AWS CLI 확인 → Parameter Store 에서 키 · 접속 비밀번호 받기 → ECR 에서 이미지 받기 →
# 컨테이너 다시 띄우기 → 스키마 맞추기 → /health 확인. 어느 단계든 실패하면 멈추고 끝 코드가 0 이 아니라
# Actions 가 빨간불이 된다. 비밀값은 출력하지 않는다(이름만 찍는다).
set -euo pipefail

REGION=ap-northeast-2
ENV_FILE=/opt/jarviseo/.env
SECRETS=/opt/jarviseo/secrets.env
cd "$(dirname "$0")"

echo "== 버전 $(git -C .. log -1 --format='%h %s')"
command -v aws >/dev/null 2>&1 || bash bootstrap.sh >/dev/null

echo "== 비밀값 (Parameter Store /jarviseo/)"
# 접속 비밀번호는 Caddy 가 읽는 bcrypt 해시로 바꾼다. 해시는 할 때마다 달라져서, 비밀번호가 그대로면
# 전의 해시를 다시 쓴다(안 그러면 배포마다 Caddy 가 다시 떠서 접속이 잠깐 끊긴다).
old_sum=$(sed -n "s/^BASIC_AUTH_SHA256='\(.*\)'$/\1/p" "$SECRETS" 2>/dev/null || true)
old_hash=$(sed -n "s/^BASIC_AUTH_HASH='\(.*\)'$/\1/p" "$SECRETS" 2>/dev/null || true)
umask 077
tmp=$(mktemp)
aws ssm get-parameters-by-path --region "$REGION" --path /jarviseo/ --with-decryption \
    --query 'Parameters[].[Name,Value]' --output text |
  while IFS=$'\t' read -r name value; do
    key=${name#/jarviseo/}
    if [ "$key" = BASIC_AUTH_PASSWORD ]; then
      sum=$(printf '%s' "$value" | sha256sum | cut -d' ' -f1)
      if [ "$sum" = "$old_sum" ] && [ -n "$old_hash" ]; then
        hash=$old_hash
      else
        hash=$(docker run --rm caddy:2.10-alpine caddy hash-password --plaintext "$value")
      fi
      printf "BASIC_AUTH_SHA256='%s'\nBASIC_AUTH_HASH='%s'\n" "$sum" "$hash"
    else
      printf "%s='%s'\n" "$key" "$value"
    fi
  done > "$tmp"
if ! grep -q '^BASIC_AUTH_HASH=' "$tmp"; then
  rm -f "$tmp"
  echo "Parameter Store 에 /jarviseo/BASIC_AUTH_PASSWORD 가 없다 (scripts/put_cloud_secrets.py)" >&2
  exit 1
fi
mv "$tmp" "$SECRETS"
echo "받음: $(cut -d= -f1 "$SECRETS" | tr '\n' ' ')"

account=$(aws sts get-caller-identity --query Account --output text)
registry="$account.dkr.ecr.$REGION.amazonaws.com"
export WEB_IMAGE=${WEB_IMAGE:-$registry/jarviseo-web:latest}
# 띄운 이미지를 secrets.env 에도 적어 둔다. 그래야 배포 밖에서 손으로 ps · logs 를 칠 때도 compose 가 뜬다.
printf "WEB_IMAGE='%s'\n" "$WEB_IMAGE" >> "$SECRETS"
echo "== 이미지 $WEB_IMAGE"
aws ecr get-login-password --region "$REGION" |
  docker login --username AWS --password-stdin "$registry" >/dev/null

compose() { docker compose --env-file "$ENV_FILE" --env-file "$SECRETS" "$@"; }
compose pull --quiet api
compose up -d --remove-orphans

# Caddyfile 은 파일 하나를 컨테이너에 붙인다. git 이 그 파일을 새로 쓰면 컨테이너는 옛 파일을 계속 보고,
# compose up 도 caddy 설정(compose.yml)이 그대로라 다시 만들지 않는다. 그래서 Caddyfile 을 고쳐도
# 반영되지 않았다(10/25, #144 의 POST /api/turns 가 계속 접속 비밀번호에 막힘).
# 컨테이너 안의 파일과 내용이 다르면 caddy 만 다시 만든다(몇 초 끊긴다. 인증서는 볼륨에 남는다).
want=$(sha256sum < Caddyfile | cut -d' ' -f1)
have=$(compose exec -T caddy sha256sum /etc/caddy/Caddyfile 2>/dev/null | cut -d' ' -f1 || true)
if [ "$want" != "$have" ]; then
  echo "== Caddyfile 이 바뀌어 caddy 를 다시 띄운다"
  compose up -d --force-recreate --no-deps caddy
fi

echo "== 스키마 (없는 테이블 · 인덱스만 만든다)"
for _ in $(seq 1 30); do
  compose exec -T db pg_isready -U jarviseo -d jarviseo >/dev/null 2>&1 && break
  sleep 2
done
# 이미 있는 테이블마다 NOTICE 가 한 줄씩 나와 배포 로그를 덮어서 경고 이상만 찍는다.
compose exec -T -e PGOPTIONS='-c client_min_messages=warning' db \
  psql -v ON_ERROR_STOP=1 -q -U jarviseo -d jarviseo < schema.sql

echo "== /health"
# 막 띄운 uvicorn 이 아직 안 받으면 연결 거부가 난다. 기다리는 동안의 오류는 숨기고, 끝내 안 되면 로그를 찍는다.
for _ in $(seq 1 60); do
  if compose exec -T api python -c \
      "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())" \
      2>/dev/null; then
    # 쓰지 않는 이미지(지난 배포의 팀 웹 이미지, 약 4GB씩)를 지운다. 디스크가 20GB 라 쌓이면 찬다.
    docker image prune -af >/dev/null
    exit 0
  fi
  sleep 2
done
compose logs --tail 50 api
exit 1
