#!/bin/bash
# T426: one grid point end to end (pre-registered): MC 2e7 (+2e8 if < 5 hits), triple choice, 3 bnb runs (2 at a time, 2 CPU-h each).
# usage: bash point.sh TAG T0   (run from m16/approaches/scale_certify with the venv active)
set -u; TAG=$1; T0=$2; ulimit -v 2500000; mkdir -p logs ../../../data/t426/mc
python mc.py $TAG $T0 2e7 0 > logs/mc_${TAG}_0.log 2>&1
H=$(python -c "import json;print(json.load(open('results/mc_${TAG}_${T0}_0.json'))['hits'])")
[ "$H" -lt 5 ] && python mc.py $TAG $T0 2e8 1 > logs/mc_${TAG}_1.log 2>&1
read A B C < <(python pick.py $TAG $T0); echo "$TAG A=$A B=$B C=$C" > logs/pick_${TAG}.log
run() { timeout 8000 python bnb_run.py $TAG $T0 $1 7200 > logs/bnb_${TAG}_${1//,/-}.log 2>&1; }
[ "$A" != "-" ] && { run $A & p1=$!; } || p1=
run $B & p2=$!
wait $p2; run $C; [ -n "$p1" ] && wait $p1
echo "point done $TAG" >> logs/pick_${TAG}.log
