#!/bin/bash
# runs results/jobs.txt with 2 workers, each single-threaded and memory-capped
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
xargs -P 2 -L 1 bash -c 'ulimit -v 2500000; python check.py "$@" >> results/run.log 2>&1' _ < results/jobs.txt
echo ALLDONE >> results/run.log
