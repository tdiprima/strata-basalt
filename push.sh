#!/usr/bin/env bash
# Copy this folder to the GPU node, as ~/strata-basalt there.
#   ./push.sh          # to the "lala" host of ~/.ssh/config
#   ./push.sh <host>   # to another host
set -euo pipefail
cd "$(dirname "$0")"
host=${1:-lala}
rsync -av --exclude results --exclude __pycache__ --exclude .DS_Store -- ./ "$host:strata-basalt/"
echo "copied. next: ssh $host, then: cd strata-basalt && tmux new -s bench ./run_all.sh"
