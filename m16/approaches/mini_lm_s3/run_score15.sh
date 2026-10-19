#!/bin/bash
# L = 15 scoring of the frozen estimators, 2 queues (2 threads each), ulimit -v 2500000
source ../../../.venv/bin/activate; ulimit -v 2500000; export S3_THREADS=2
L=../../../data/t446
q() { for m in "$@"; do python score15.py $m >> $L/score15_$m.log 2>&1; done; }
q dyck2_rnn_s0 dyck2_softmax_s0 dyck2_hard_s0 dyck2_relu_s0 dyck2_linear_s0 dyck2_pos_s0 dyck2_rnn_s2 dyck2_softmax_s2 dyck2_hard_s2 &
q dyck2_rnn_s1 dyck2_softmax_s1 dyck2_hard_s1 dyck2_relu_s1 dyck2_linear_s1 dyck2_pos_s1 dyck2_relu_s2 dyck2_linear_s2 dyck2_pos_s2 &
wait
