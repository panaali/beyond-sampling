#!/bin/bash
cd "$(dirname "$0")"; source ../../../.venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -v 2500000
for HT in "8 3" "16 3" "8 4" "16 4"; do set -- $HT; [ -f out/final_H$1_T$2.npz ] || timeout 900 python leaves.py final $1 $2; done
for m in target frob0 frob1 frob2 frob3 frob4 row0 row1 row2 row3 row4 perm0 flip0; do [ -f out/prefix_${m}_k4.npz ] || timeout 600 python leaves.py prefix $m 4; done
