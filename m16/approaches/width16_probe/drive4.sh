#!/bin/bash
# T205: certified exact region counts (geometry only) + sampled lower bounds for the small rungs and (8,5) zoo0.
cd "$(dirname "$0")"; source ../../../.venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -v 2500000
N="zoo0 zoo1 zoo2 zoo3 zoo4 zoo0_noisy0 zoo0_noisy1 zoo0_noisy2 zoo0_noisy3"
for n in $N; do python regions_exact.py 8 4 $n; done
for n in $N; do python regions_exact.py 16 4 $n; done
for H in 4 5 6; do python mc.py $H 3 zoo0 zoo1 zoo2 zoo3 zoo4; done
python mc.py 8 5 zoo0
python regions_exact.py 8 5 zoo0
