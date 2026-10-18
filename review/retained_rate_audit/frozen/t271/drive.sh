#!/bin/bash
cd "$(dirname "$0")/../../.."; source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -v 2500000
for f in enum sci_tao probe form; do python m16/approaches/rate_alpha/rates.py $f > m16/approaches/rate_alpha/run_$f.log 2>&1; done
