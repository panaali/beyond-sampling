#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TRITON_INTERPRET=1 PYTHONPATH=/workspace/work/t261-triton-deps
set -o noclobber
flock /workspace/work/codex-compute.lock bash -c 'set -euo pipefail; ulimit -St 105; ulimit -Ht 110; ulimit -Sv 2097152; ulimit -Hv 2097152; time -p timeout 120 /workspace/clean-neurons/.venv/bin/python validate.py' >validation_run1.log 2>&1
