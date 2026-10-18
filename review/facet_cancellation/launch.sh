#!/usr/bin/env bash
set -euo pipefail
cd /workspace/work/recover-small_h2_56
set -o noclobber
flock /workspace/work/codex-primary.lock bash -c '
set -euo pipefail
ulimit -St 110
ulimit -Ht 110
ulimit -Sv 2097152
ulimit -Hv 2097152
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout 120 /workspace/clean-neurons/.venv/bin/python review/facet_cancellation/run.py
' > review/facet_cancellation/run1.log 2>&1
