#!/bin/bash
# T206: exact-mode cost anatomy, chamber vs truth-cone start.
cd "$(dirname "$0")"; source ../../../.venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -v 2500000
for st in chamber truth; do for HT in "8 3" "16 3" "8 4" "16 4"; do python anatomy.py $HT zoo0 $st; done; done
