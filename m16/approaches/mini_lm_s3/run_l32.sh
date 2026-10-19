#!/bin/bash
# Addendum A, L = 32: 4e7-sample MC truths then score32; 2 queues.
source ../../../.venv/bin/activate; ulimit -v 2500000; export S3_THREADS=2
D=../../../data/t446
q() { for m in "$@"; do
  [ -f $D/${m}_L32_mc_1.npz ] || python mc32.py $m 4e7 1 >> $D/mc32_$m.log 2>&1
  python score32.py $m >> $D/score32_$m.log 2>&1
done; }
q dyck2_rnn_s110 dyck2_softmax_s110 dyck2_hard_s110 dyck2_relu_s110 dyck2_linear_s110 dyck2_pos_s110 &
q dyck2_rnn_s111 dyck2_softmax_s111 dyck2_hard_s111 dyck2_relu_s111 dyck2_linear_s111 dyck2_pos_s111 &
wait
mkdir -p results && cp $D/score32.jsonl results/ && for f in $D/*_L32_mc_1.npz; do cp $f results/; done
