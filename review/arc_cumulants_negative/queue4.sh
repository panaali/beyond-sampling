#!/bin/bash
# T80 relaunch after container restart (one core, <= 2.5 GB per process, sequential)
source /home/user/clean-neurons/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -v 2500000
cd /home/user/clean-neurons/review/arc_cumulants_negative
T80_LOG2_NG=20 python endpoint_ceiling.py 10 p1p2 200000 > out_ceiling_T10.txt 2>&1
T80_LOG2_NG=20 python variants_acc.py 10 joint:arc2,cell:arc2 seed1,s0.0025_0,s0.005_0,s0.01_0,s0.02_0,s0.02_1 > out_rmse_T10b.txt 2>&1
python readout_ceiling.py 4 2,3,4 cell 400000 s0.01_0 > out_readout_T4_cell_s0.01_0.txt 2>&1
python readout_ceiling.py 10 2,3 uncond 1000000 > out_readout_T10_uncond.txt 2>&1
T80_LOG2_NG=20 python variants_acc.py 10 joint:gauss,cell:gauss base > out_variants_T10.txt 2>&1
echo QUEUE_DONE > out_queue4_done.txt
