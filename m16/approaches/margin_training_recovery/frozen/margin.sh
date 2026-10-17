#!/bin/bash
# T115 driver: RECIPE85B / RECIPE112 over k = 30, 28..20 (one process at a time, 1 core, 2.5 GB cap).
cd "$(dirname "$0")"; source ../../../.venv/bin/activate; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -v 2500000
for k in ${@:-30 28 27 26 25 24 23 22 21 20}; do
  python run_margin.py $k >> margin.log 2>&1
done
echo finished >> margin.log
