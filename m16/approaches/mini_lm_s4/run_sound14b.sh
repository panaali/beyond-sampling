#!/bin/bash
# remaining K0 = 14 soundness models (restarted with the one-pass modsum code), then the S1 group's L = 15 work
cd "$(dirname "$0")"
source /home/user/clean-neurons/.venv/bin/activate
ulimit -v 2500000
R="dyck2_linear_s2 dyck2_pos_s0 dyck2_pos_s1 dyck2_pos_s2 dyck2_relu_s0 dyck2_relu_s1 dyck2_relu_s2 dyck2_rnn_s0 dyck2_rnn_s1 dyck2_rnn_s2 dyck2_softmax_s0 dyck2_softmax_s1 dyck2_softmax_s2"
MS=$(cd ../mini_lm && python -c "from load import names; print(' '.join(n for n in names() if n.startswith('modsum')))")
python soundness.py 14 $R $MS > sound_K14b.log 2>&1
S1="dyck2_D6_s10 dyck2_D6_s11 dyck2_D6_s12 dyck2_R3a_s10 dyck2_R3a_s11 dyck2_R3a_s12"
python ref.py $S1 > ref_s1.log 2>&1
python soundness.py 14 $S1 > sound_K14_s1.log 2>&1
python soundness.py 13 dyck2_D6_s10 dyck2_R3a_s10 > sound_K13_s1.log 2>&1
python horizon.py 32 100 dyck2_D6_s10 dyck2_R3a_s10 > horizon_s1.log 2>&1
