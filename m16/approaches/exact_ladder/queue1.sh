#!/bin/bash
# sequential runs, 1 core, memory-capped
cd /home/user/clean-neurons
source .venv/bin/activate
ulimit -v 2500000
export OMP_NUM_THREADS=1
for T in 2 3; do python m16/leaderboard.py truths --H 16 --T $T > m16/approaches/exact_ladder/truths_T$T.log 2>&1; done
for T in 2 3 4; do python m16/leaderboard.py run exact_ladder --H 16 --T $T > m16/approaches/exact_ladder/lb_T$T.log 2>&1; done
python m16/leaderboard.py run exact_ladder --H 16 --T 5 --groups seed > m16/approaches/exact_ladder/lb_T5.log 2>&1
