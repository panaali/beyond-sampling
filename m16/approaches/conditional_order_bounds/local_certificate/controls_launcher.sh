#!/usr/bin/env bash
set -euo pipefail
cd /workspace/work/recover-small_h2_78/m16/approaches/conditional_order_bounds/local_certificate
[[ ! -e controls_run1.log && ! -e controls_results.json && ! -e controls_progress.jsonl ]]
ulimit -v 2097152
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
flock /workspace/work/codex-review.lock bash -c 'set -euo pipefail; ulimit -t 20; ulimit -a; time -p /workspace/clean-neurons/.venv/bin/python controls.py' > controls_run1.log 2>&1
