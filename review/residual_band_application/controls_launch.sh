#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
set -o noclobber
flock /workspace/work/codex-review.lock bash -c 'set -euo pipefail; ulimit -St 19; ulimit -Ht 20; ulimit -Sv 2097152; ulimit -Hv 2097152; time -p timeout 25 /workspace/clean-neurons/.venv/bin/python controls.py' >controls_run1.log 2>&1
