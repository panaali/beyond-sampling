#!/bin/bash
# T101 driver: k = 30 (harness check vs E37) then 28, 27, ... ; stops after the first rung where a recipe fails.
cd "$(dirname "$0")"; source ../../../.venv/bin/activate; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -v 2500000
for k in ${@:-30 28 27 26 25 24 23 22 21 20}; do
  [ -f rung_k$k.json ] && python -c "import json,sys; sys.exit(0 if len(json.load(open('rung_k$k.json'))['seeds'])==5 else 1)" && { echo "k=$k cached"; } || python run_rung.py $k >> rung_k$k.log 2>&1
  python verdict.py $k >> ladder.log 2>&1 || { [ $k -ne 30 ] && { echo "STOP at k=$k" >> ladder.log; break; }; }
done
echo finished >> ladder.log
