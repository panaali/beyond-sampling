#!/bin/bash
# T223: full leaderboard v3 run of tie_blind (30 models x 5 budgets), single thread, 2 GB cap.
cd "$(dirname "$0")/../../.."; source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -v 2500000
python m16/approaches/tie_blind/score.py run 1000,4000,16000,64000,256000
