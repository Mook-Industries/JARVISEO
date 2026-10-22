#!/bin/bash
# 자비서 시작 (맥). Finder 에서 더블클릭한다.  담당: 문태현 (배포)
# 하는 일은 scripts/start.py 에 있다. 여기서는 conda 환경 JARVISEO 의 파이썬만 찾는다.
# 다른 곳에 깔았으면 JARVISEO_PYTHON 에 python 경로를 적는다.

cd "$(dirname "$0")/.." || exit 1

PY="${JARVISEO_PYTHON:-}"
for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/opt/miniconda3" "$HOME/opt/anaconda3"     "$HOME/.conda" "/opt/homebrew/Caskroom/miniconda/base" "/opt/miniconda3" "/opt/anaconda3"; do
    if [ -z "$PY" ] && [ -x "$base/envs/JARVISEO/bin/python" ]; then
        PY="$base/envs/JARVISEO/bin/python"
    fi
done
if [ -z "$PY" ]; then
    echo "conda 환경 JARVISEO 를 찾지 못했습니다. docs/deploy.md 의 설치 순서를 따르세요."
    read -r -p "Enter 를 누르면 닫힙니다"
    exit 1
fi

"$PY" scripts/start.py "$@"
echo
read -r -p "Enter 를 누르면 닫힙니다"
