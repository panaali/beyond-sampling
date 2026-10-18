#!/bin/bash
# T232: k=2,3 counts + accuracy, k=4 counts + masses, for all models (sequential, 10-min timeout each).
cd "$(dirname "$0")"; source ../../../.venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -v 2500000
M="target frob0 frob1 frob2 frob3 frob4 row0 row1 row2 row3 row4 perm0 flip0"
for m in $M; do for k in 2 3; do [ -f out/${m}_k$k.json ] || timeout 600 python rc.py $m $k; done; done
for m in $M; do [ -f out/${m}_k4.json ] || timeout 600 python rc.py $m 4 mass || echo "$m k4 TIMEOUT"; done
