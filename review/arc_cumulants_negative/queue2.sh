#!/bin/bash
# sequential job queue (one core, <=3 GB virtual per process)
source /home/user/clean-neurons/.venv/bin/activate
export OMP_NUM_THREADS=1
ulimit -v 3000000
cd /home/user/clean-neurons/review/arc_cumulants_negative
python endpoint_ceiling.py 4 perm,p1p2s 400000 seed3 > out_ceiling_T4_fine.txt 2>&1
python rootcause.py 4 64 1,2,4,16,H 1000000 3 > out_rootcause_T4_H64.txt 2>&1
python rootcause.py 10 64 2,4,16,H 1000000 3 > out_rootcause_T10_H64.txt 2>&1
python variants_acc.py 4 rim:1:-16:gauss,rim:2:-16:gauss,rim:3:-16:gauss,rim:4:-16:gauss,rim:4:-32:gauss base > out_rim_T4.txt 2>&1
