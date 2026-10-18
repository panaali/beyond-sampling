#!/usr/bin/env bash
set -euo pipefail
cd /workspace/work/recover-small_h2_78/m16/approaches/conditional_order_bounds/local_certificate
[[ ! -e application_run1.log && ! -e application ]]
ulimit -v 2097152
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
flock /workspace/work/codex-compute.lock bash -c 'set -euo pipefail; ulimit -t 90; ulimit -a; time -p /workspace/clean-neurons/.venv/bin/python application.py' > application_run1.log 2>&1
