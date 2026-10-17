#!/bin/bash
cd /home/user/clean-neurons/m16/approaches/exact_ladder
source ../../../.venv/bin/activate
ulimit -v 2500000
export OMP_NUM_THREADS=1
while kill -0 8004 2>/dev/null; do sleep 30; done
for T in 2 3 4; do python diag_outcome.py $T seed0 >> outcome.log 2>&1; done
python extrap6.py 6 seed0 200 20 > extrap6_seed0.log 2>&1
python diag_outcome.py 5 seed0 >> outcome.log 2>&1
