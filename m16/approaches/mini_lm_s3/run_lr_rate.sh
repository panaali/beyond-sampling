#!/bin/bash
# (ii)-(iv) on the s0 seed of each architecture: exact truths if missing, then rate.py and lr.py; 2 queues.
source ../../../.venv/bin/activate; ulimit -v 2500000; export S3_THREADS=2
D=../../../data/t446
q() { for m in "$@"; do
  [ -f $D/${m}_L15_exact.json ] || python exact15.py $m
  python rate.py $m >> $D/rate_$m.log 2>&1; cp $D/rate_$m.json results/
  python lr.py $m >> $D/lr_$m.log 2>&1; cp $D/lr_$m.json results/
done; }
q dyck2_rnn_s0 dyck2_hard_s0 dyck2_linear_s0 &
q dyck2_softmax_s0 dyck2_relu_s0 dyck2_pos_s0 &
wait
