#!/bin/bash
# pull /content/res.tgz (written by runq.sh after every run) from a Colab session into this folder
S=${1:-t192gpu}; C=/home/user/colabcli-venv/bin/colab; HERE=$(cd "$(dirname "$0")/.." && pwd)
timeout 300 $C download -s $S /content/res.tgz /tmp/claude-0/res_$S.tgz >/dev/null 2>&1 && tar xzf /tmp/claude-0/res_$S.tgz -C "$HERE" && echo "synced $S"
