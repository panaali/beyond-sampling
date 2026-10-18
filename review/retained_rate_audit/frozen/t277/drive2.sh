#!/bin/bash
cd "$(dirname "$0")/../../.."; source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -v 2500000
while kill -0 1977
2150 2>/dev/null; do sleep 20; done
python m16/approaches/rate_T/run.py truth >> m16/approaches/rate_T/run_truth.log 2>&1
python m16/approaches/rate_T/run.py sci_tao >> m16/approaches/rate_T/run_sci.log 2>&1
