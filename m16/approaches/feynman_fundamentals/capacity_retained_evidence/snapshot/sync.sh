#!/bin/bash
# pull /content/t234.tgz (results, weights, checkpoints) from a Colab session into the repo root
S=${1:-t192t4}; C=/home/user/colabcli-venv/bin/colab; ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
timeout 300 $C download -s $S /content/t234.tgz /tmp/claude-0/t234_$S.tgz >/dev/null 2>&1 && tar xzf /tmp/claude-0/t234_$S.tgz -C "$ROOT" && echo "synced $S"
