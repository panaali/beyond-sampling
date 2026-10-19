#!/bin/bash
# usage: bash drive.sh W K   -- worker W of K runs every K-th job of the pre-registered queue (resumable).
cd "$(dirname "$0")"; . ../../../.venv/bin/activate; ulimit -v 2500000
python -c "
from configs import CONFIGS
for b in (256000, 1000000):
    for f, s in CONFIGS: print(f, s, b)
for f, s in CONFIGS: print(f, s, 4000000)" | awk -v w=$1 -v k=$2 'NR % k == w' | while read f s b; do
  out=runs/${f}_${s}_${b}.jsonl
  [ -f $out ] && [ $(wc -l < $out) -ge 100 ] && continue
  python run.py $f $s $b $out > /dev/null || echo "FAIL $f $s $b" >&2
done
echo "worker $1 done"
