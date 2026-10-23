#!/bin/bash
# 클라우드 서버에서 새 버전을 띄운다.  담당: 문태현 (배포)
#
# GitHub Actions(.github/workflows/deploy.yml)가 main 에 머지될 때 SSM 으로 부른다.
# 그 전에 /opt/jarviseo/repo 를 origin/main 으로 맞춰 둔다. 손으로 부를 때도 같다.
#     sudo bash /opt/jarviseo/repo/deploy/update.sh
#
# 하는 일: api 이미지 다시 빌드 → 컨테이너 다시 띄우기 → 스키마 맞추기 → /health 확인.
# 어느 단계든 실패하면 멈추고 끝 코드가 0 이 아니라서 Actions 가 빨간불이 된다.
set -euo pipefail

ENV_FILE=/opt/jarviseo/.env
cd "$(dirname "$0")"
compose() { docker compose --env-file "$ENV_FILE" "$@"; }

echo "== 버전 $(git -C .. log -1 --format='%h %s')"
compose up -d --build --remove-orphans

echo "== 스키마 (없는 테이블 · 인덱스만 만든다)"
for _ in $(seq 1 30); do
  compose exec -T db pg_isready -U jarviseo -d jarviseo >/dev/null 2>&1 && break
  sleep 2
done
compose exec -T db psql -v ON_ERROR_STOP=1 -q -U jarviseo -d jarviseo < schema.sql

echo "== /health"
# 막 띄운 uvicorn 이 아직 안 받으면 연결 거부가 난다. 기다리는 동안의 오류는 숨기고, 끝내 안 되면 로그를 찍는다.
for _ in $(seq 1 30); do
  if compose exec -T api python -c \
      "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health/db').read().decode())" \
      2>/dev/null; then
    docker image prune -f >/dev/null
    exit 0
  fi
  sleep 2
done
compose logs --tail 50 api
exit 1
