#!/bin/bash
source /home/user/clean-neurons/.venv/bin/activate
ulimit -v 2500000; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while pgrep -f "he14.py" >/dev/null; do sleep 20; done
python dd.py final 16 3 0
python dd.py final 16 4 0
python he15.py prefix_target_k4 final_zoo16_3_0 final_zoo16_4_0 prefix_frob0_k4 prefix_frob1_k4 prefix_frob2_k4 prefix_row0_k4 prefix_row1_k4 prefix_row2_k4
