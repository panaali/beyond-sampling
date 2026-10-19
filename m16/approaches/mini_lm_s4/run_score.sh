#!/bin/bash
# L = 32 scoring queue (T447), started after PID $1 ends
cd "$(dirname "$0")"
source /home/user/clean-neurons/.venv/bin/activate
ulimit -v 2500000
while kill -0 "$1" 2>/dev/null; do sleep 30; done
M=/home/user/wt-mlp1/m16/approaches/mini_lm/models_mlp1
ARCH="rnn softmax hard relu linear pos"
ALL=$(for a in $ARCH; do for s in $(seq 110 119); do echo -n "dyck2_${a}_s$s "; done; done)
S1="dyck2_D6_s10 dyck2_D6_s11 dyck2_D6_s12 dyck2_R3a_s10 dyck2_R3a_s11 dyck2_R3a_s12"
S4_PLAN="0.9:1e5,0.5:1e5,0.99:1e5" S4_TAG="_b1e5" python score32.py $M $ALL > score_b1e5.log 2>&1
S4_PLAN="0.9:1e5,0.5:1e5,0.99:1e5" S4_TAG="_b1e5" python score32.py - $S1 > score_b1e5_s1.log 2>&1
S4_PLAN="0.9:1e6" S4_TAG="_b1e6" python score32.py $M $(for a in $ARCH; do echo -n "dyck2_${a}_s110 "; done) > score_b1e6.log 2>&1
S4_PLAN="0.9:1e6" S4_TAG="_b1e6" python score32.py - dyck2_D6_s10 dyck2_R3a_s10 >> score_b1e6.log 2>&1
python ref.py $S1 > ref_s1.log 2>&1
python soundness.py 14 $S1 > sound_K14_s1.log 2>&1
python soundness.py 13 dyck2_D6_s10 dyck2_R3a_s10 > sound_K13_s1.log 2>&1
python horizon.py 32 100 dyck2_D6_s10 dyck2_R3a_s10 > horizon_s1.log 2>&1
