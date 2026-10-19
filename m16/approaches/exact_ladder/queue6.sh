#!/bin/bash
# T46 (2026-10-19): T=5 exact on the remaining leaderboard models (seeds 2-4, then the 24 noisy copies), 1 core
cd /home/user/clean-neurons/m16/approaches/exact_ladder
source ../../../.venv/bin/activate
ulimit -v 2500000
export OMP_NUM_THREADS=1
python run_list.py 5 seed2 seed3 seed4 $(for s in 0.0025 0.005 0.01 0.02; do for k in 0 1 2 3 4 5; do echo s${s}_$k; done; done) >> run5_list.log 2>&1
