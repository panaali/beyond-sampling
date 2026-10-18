#!/bin/bash
# T195 phase 2 queue (runs after run5.sh; one process at a time): (8,4) full grid with oracle, then (8,5) seed 0 cap.
cd "$(dirname "$0")"; source ../../../.venv/bin/activate
ulimit -v 2500000; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while pgrep -f "attr.py 8 5 0" >/dev/null; do sleep 60; done
for s in 0 1 2 3 4; do for tau in 1e-4 1e-5 1e-6 1e-7 1e-8; do for te in cap mass; do
  [ -f prune/H8_T4_s${s}_${te}_${tau/1e-/1e-0}.json ] || [ -f prune/H8_T4_s${s}_${te}_${tau}.json ] || python prune.py 8 4 $s $tau $te oracle
done; done; done
for tau in 1e-5 1e-6 1e-7; do [ -f prune/H8_T5_s0_cap_${tau/1e-/1e-0}.json ] || python prune.py 8 5 0 $tau cap; done
