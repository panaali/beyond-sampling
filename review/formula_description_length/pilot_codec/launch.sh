#!/bin/bash
set -euo pipefail
cd /workspace/work/recover-small_h4_4/review/formula_description_length/pilot_codec
exec 9>/workspace/work/codex-compute.lock
flock -x 9
test ! -e run1.log
set -o noclobber
{ time -p timeout --signal=TERM --kill-after=2s 120s bash -c 'ulimit -S -t 105; ulimit -H -t 110; ulimit -v 2097152; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; exec /workspace/clean-neurons/.venv/bin/python run.py'; } >run1.log 2>&1
