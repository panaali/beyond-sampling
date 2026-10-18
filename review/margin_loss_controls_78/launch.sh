#!/bin/bash
set -euo pipefail
cd /workspace/work/recover-small_h2_78/review/margin_loss_controls_78
test ! -e run1.log
test ! -e output
set -C
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
flock /workspace/work/codex-review.lock bash -c 'set -euo pipefail; ulimit -St 16; ulimit -Ht 20; ulimit -Sv 2097152; ulimit -Hv 2097152; time -p timeout 25 /usr/bin/python3 /workspace/work/recover-small_h2_78/review/margin_loss_controls_78/run.py' >run1.log 2>&1
