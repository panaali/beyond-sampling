#!/bin/bash
# T217: event-word branching on exact prefix leaves.
cd "$(dirname "$0")"; source ../../../.venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -v 2500000
for m in target zoo16_4 zoo16_5; do for k in 2 3 4; do python eventword.py $m $k; done; done
