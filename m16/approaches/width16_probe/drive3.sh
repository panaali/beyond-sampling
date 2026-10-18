#!/bin/bash
# T194 phase 3: (16,4) noisy copies of zoo0 (recipe108 + exact) and their MC/regions.
cd "$(dirname "$0")"; source ../../../.venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -v 2500000
python mc.py 16 4 zoo0_noisy0 zoo0_noisy1 zoo0_noisy2 zoo0_noisy3
./drive.sh 16 4 zoo0_noisy0 zoo0_noisy1 zoo0_noisy2 zoo0_noisy3
