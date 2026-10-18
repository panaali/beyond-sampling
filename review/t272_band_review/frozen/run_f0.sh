#!/bin/bash
source /home/user/clean-neurons/.venv/bin/activate
ulimit -v 2500000; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in orig s005 s010 z10s0 z10s1 z10s2 z10s3 z10s4; do python -W ignore f0.py $m; done
