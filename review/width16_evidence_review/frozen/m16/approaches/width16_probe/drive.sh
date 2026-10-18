#!/bin/bash
# T194 driver: sequential (1 heavy process), resumable (run.py skips finished outputs). usage: setsid nohup ./drive.sh H T names... > drive_H_T.log &
cd "$(dirname "$0")"; source ../../../.venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; ulimit -v 2500000
H=$1; T=$2; shift 2
for n in "$@"; do for mode in recipe108 exact; do python run.py $H $T $n $mode; done; done
