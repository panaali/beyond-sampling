#!/bin/bash
source /home/user/clean-neurons/.venv/bin/activate
ulimit -v 2500000; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C=""; for T in 3 4 5 6 8 10; do for s in 0 1 2 3 4; do C="$C 16:$T:$s"; done; done
python -W ignore kinks.py 2000 40 orig $C
python -W ignore kinks.py 8000 10 orig 16:5:0 16:10:0
