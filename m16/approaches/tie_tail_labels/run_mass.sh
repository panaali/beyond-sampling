#!/bin/bash
source /home/user/clean-neurons/.venv/bin/activate
ulimit -v 2500000; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -W ignore prefix_mass.py 2 3 4
