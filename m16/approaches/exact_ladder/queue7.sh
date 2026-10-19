#!/bin/bash
# T46: after the T6 hybrid scoring ends, run the remaining T=5 models on 2 workers (claim files; no duplicates).
# Worker 1 (queue6.sh, no claims) is replaced at its next model boundary by a claiming worker in forward order.
cd /home/user/clean-neurons/m16/approaches/exact_ladder
source ../../../.venv/bin/activate
ulimit -v 2500000
export OMP_NUM_THREADS=1
while kill -0 2449 2>/dev/null; do sleep 60; done
python run_claim.py 5 s0.02_5 s0.02_4 s0.02_3 s0.02_2 s0.02_1 s0.02_0 s0.01_5 s0.01_4 s0.01_3 s0.01_2 s0.01_1 s0.01_0 s0.005_5 s0.005_4 s0.005_3 s0.005_2 s0.005_1 s0.005_0 s0.0025_5 s0.0025_4 s0.0025_3 s0.0025_2 s0.0025_1 s0.0025_0  >> run5_claim_rev.log 2>&1
