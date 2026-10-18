#!/bin/bash
# T194 phase 2 (sequential): MC for (16,4); (8,4) exact re-runs for exact region counts; (16,4) seeds 1-4; mc2 checks.
cd "$(dirname "$0")"; source ../../../.venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -v 2500000
python mc.py 16 4 zoo0 zoo1 zoo2 zoo3 zoo4
for n in zoo0 zoo1 zoo2 zoo3 zoo4 zoo0_noisy0 zoo0_noisy1 zoo0_noisy2 zoo0_noisy3; do python run.py 8 4 $n exact; done
python mc2.py 16 3 zoo0 zoo0_noisy3 zoo2; python mc2.py 16 4 zoo0
./drive.sh 16 4 zoo1 zoo2 zoo3 zoo4
