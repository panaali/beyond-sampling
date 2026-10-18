#!/bin/bash
# T195 stage 2: (8,5) seed 0, instrumented (resumable: per-half checkpoints in data/split_attribution/H8_T5_s0).
# Reference = estT_fix/run5/zoo0_recipe108.json (copied to ref/H8_T5_s0.json).
cd "$(dirname "$0")"; source ../../../.venv/bin/activate
ulimit -v 2500000; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python attr.py 8 5 0
