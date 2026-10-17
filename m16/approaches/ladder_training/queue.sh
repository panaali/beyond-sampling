#!/bin/bash
# usage: queue.sh "H T seed log2n [scaled]" ...   runs jobs one after another on one core
source /home/user/clean-neurons/.venv/bin/activate
cd /home/user/clean-neurons/m16/approaches/ladder_training
for job in "$@"; do
  set -- $job
  tag=$4; [ "$5" = scaled ] && tag=${4}s
  [ -f ../../../data/ladder_training/m_$1_$2_${tag}_$3.npz ] && continue
  OMP_NUM_THREADS=1 python train.py $1 $2 $3 $4 $5 2>&1 | grep -v "^Saving" >> ../../../data/ladder_training/queue.log
done
