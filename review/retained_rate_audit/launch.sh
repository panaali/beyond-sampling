#!/bin/bash
set -euo pipefail
set -C
cd /workspace/work/recover-small_h2_78/review/retained_rate_audit
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
flock /workspace/work/codex-review.lock bash -c 'set -euo pipefail; ulimit -St 19; ulimit -Ht 20; ulimit -Sv 2097152; ulimit -Hv 2097152; time -p timeout 25 /workspace/clean-neurons/.venv/bin/python check.py' >run1.log 2>&1
