#!/bin/bash
# T195 phase 2: (8,3) pruning grid, seeds 0-4 x tau x {cap, mass}, oracle on (seconds).
cd "$(dirname "$0")"; source ../../../.venv/bin/activate
ulimit -v 2500000; export OMP_NUM_THREADS=1
python exact3.py
for s in 0 1 2 3 4; do for tau in 1e-4 1e-5 1e-6 1e-7 1e-8; do for te in cap mass; do python prune.py 8 3 $s $tau $te oracle; done; done; done
