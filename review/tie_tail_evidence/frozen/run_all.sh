#!/bin/bash
# Stage 1: tail points + labels for all 8 models (single thread, <=2.5 GB).
source /home/user/clean-neurons/.venv/bin/activate
ulimit -v 2500000; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in z6s0 z6s1 z6s2 z6s3 z6s4 orig s005 s010; do s=$SECONDS; python tail.py $m 14; echo "$m $((SECONDS-s)) s"; done
