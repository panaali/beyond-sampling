#!/bin/bash
# keepalive.sh SESSION: loop blocking `colab exec` of keepalive.py (local background process; kill its PID to stop)
S=$1; C=/home/user/colabcli-venv/bin/colab; D=$(cd "$(dirname "$0")" && pwd)
while true; do timeout 3100 $C exec -s $S --timeout 3000 -f $D/keepalive.py >> /tmp/claude-0/keepalive_$S.log 2>&1; sleep 5; done
