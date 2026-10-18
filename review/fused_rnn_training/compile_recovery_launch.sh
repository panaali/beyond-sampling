#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
set -o noclobber
flock /workspace/work/codex-compute.lock bash -c 'set -euo pipefail; ulimit -St 105; ulimit -Ht 110; ulimit -Sv 2097152; ulimit -Hv 2097152; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/workspace/work/t261-triton-deps TRITON_CACHE_DIR=/workspace/work/t261-triton-cache; unset TRITON_INTERPRET; time -p timeout 120 /workspace/clean-neurons/.venv/bin/python compile_check.py' >compile_run2.log 2>&1
