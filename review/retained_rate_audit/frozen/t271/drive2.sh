#!/bin/bash
# T271 resume after the container restart: probe (remaining model), form, then D6 region counts.
cd "$(dirname "$0")/../../.."; source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -v 2500000
python m16/approaches/rate_alpha/rates.py probe >> m16/approaches/rate_alpha/run_probe.log 2>&1
python m16/approaches/rate_alpha/rates.py form >> m16/approaches/rate_alpha/run_form.log 2>&1
python m16/approaches/rate_alpha/d6.py count > m16/approaches/rate_alpha/run_d6.log 2>&1
