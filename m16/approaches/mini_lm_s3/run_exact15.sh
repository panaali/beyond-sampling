#!/bin/bash
# exact L = 15 truths, 2 queues
source ../../../.venv/bin/activate; ulimit -v 2500000
q() { for m in "$@"; do [ -f ../../../data/t446/${m}_L15_exact.json ] || python exact15.py $m; done; }
q dyck2_rnn_s0 dyck2_rnn_s1 dyck2_rnn_s2 dyck2_softmax_s0 dyck2_softmax_s1 dyck2_softmax_s2 dyck2_hard_s0 dyck2_hard_s1 dyck2_hard_s2 > ../../../data/t446/log_q1.txt 2>&1 &
q dyck2_relu_s0 dyck2_relu_s1 dyck2_relu_s2 dyck2_linear_s0 dyck2_linear_s1 dyck2_linear_s2 dyck2_pos_s1 dyck2_pos_s2 > ../../../data/t446/log_q2.txt 2>&1 &
wait
