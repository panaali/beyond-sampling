#!/bin/bash
# T195 stage 1: reference + instrumented runs at (8,3), (8,4), seeds 0-4 (one process at a time).
cd "$(dirname "$0")"; source ../../../.venv/bin/activate
ulimit -v 2500000; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for T in 3 4; do for s in 0 1 2 3 4; do
  [ -f ref/H8_T${T}_s$s.json ] || python ref.py 8 $T $s
  [ -f out/H8_T${T}_s$s.npz ] || python attr.py 8 $T $s
done; done
