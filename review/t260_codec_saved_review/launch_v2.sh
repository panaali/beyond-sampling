#!/bin/bash
set -euo pipefail
cd /workspace/work/recover-small_h2_56/review/t260_codec_saved_review
exec 9>/workspace/work/codex-compute.lock
flock -x 9
test ! -e run2.log
set -o noclobber
{ time -p timeout --signal=TERM --kill-after=2s 120s bash -c 'ulimit -S -t 105; ulimit -H -t 110; ulimit -v 2097152; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; exec /workspace/clean-neurons/.venv/bin/python check_v2.py'; } >run2.log 2>&1
