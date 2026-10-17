#!/bin/bash
source /home/user/clean-neurons/.venv/bin/activate
export OMP_NUM_THREADS=1
ulimit -v 3000000
cd /home/user/clean-neurons/review/arc_cumulants_negative
while kill -0 26668 2>/dev/null; do sleep 20; done
python readout_ceiling.py 10 2,3,4 cell 200000 > out_readout_T10_cell.txt 2>&1
python endpoint_ceiling.py 10 p1p2 200000 > out_ceiling_T10.txt 2>&1
python variants_acc.py 10 joint:arc2,joint:gauss,cell:arc2,cell:gauss base > out_variants_T10.txt 2>&1
python variants_acc.py 4 arccell:2,arccell:3,arccell:4 base > out_arccell_T4.txt 2>&1
python readout_ceiling.py 4 2,3,4 cell 400000 s0.01_0 > out_readout_T4_cell_s0.01_0.txt 2>&1
python readout_ceiling.py 10 2,3 uncond 1000000 > out_readout_T10_uncond.txt 2>&1
