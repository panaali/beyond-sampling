#!/bin/bash
cd /home/user/clean-neurons/m16/approaches/exact_ladder
source ../../../.venv/bin/activate
ulimit -v 2500000
export OMP_NUM_THREADS=1
python extrap6.py 5 seed0 20 10 > extrap5_seed0_v2.log 2>&1
python extrap6.py 6 seed0 200 20 > extrap6_seed0.log 2>&1
python diag_outcome.py 5 seed0 >> outcome.log 2>&1
