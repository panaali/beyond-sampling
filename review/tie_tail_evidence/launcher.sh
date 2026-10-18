#!/usr/bin/env bash
set -euo pipefail
cd /workspace/work/recover-small_h2_78/review/tie_tail_evidence
[[ ! -e audit && ! -e run1.log ]]
ulimit -v 2097152
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
flock /workspace/work/codex-review.lock bash -c 'set -euo pipefail; ulimit -t 20; ulimit -a; time -p /workspace/clean-neurons/.venv/bin/python check_saved.py' > run1.log 2>&1
