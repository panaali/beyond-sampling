#!/bin/bash
# usage: queue2.sh JOBFILE   (start 2 copies for 2 workers) each line "H T seed log2n scaled [lr_mult]";
# a worker claims a job with an atomic mkdir in data/ladder_training/claims/, so two workers never run the same job.
source /home/user/clean-neurons/.venv/bin/activate
cd /home/user/clean-neurons/m16/approaches/ladder_training
ulimit -v 2500000
C=../../../data/ladder_training/claims; mkdir -p $C
while read -r line; do
  [ -z "$line" ] && continue
  id=$(echo $line | tr ' ' '_')
  mkdir $C/$id 2>/dev/null || continue
  OMP_NUM_THREADS=1 python train.py $line 2>&1 | grep -v "^Saving" >> ../../../data/ladder_training/queue.log
done < "$1"
