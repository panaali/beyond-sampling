#!/bin/bash
# waits for the reference builder (PID $1) then runs the K0 = 14 soundness pass over all 36 baselines
cd "$(dirname "$0")"
source /home/user/clean-neurons/.venv/bin/activate
ulimit -v 2500000
while kill -0 "$1" 2>/dev/null; do sleep 30; done
python soundness.py 14 $(cd ../mini_lm && python -c "from load import names; print(' '.join(n for n in names()))") > sound_K14.log 2>&1
