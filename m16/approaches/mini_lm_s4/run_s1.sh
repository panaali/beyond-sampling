#!/bin/bash
# after PID $1 ends: exact references, soundness (K0 14 and 13) and horizon for the S1 (T444) group
cd "$(dirname "$0")"
source /home/user/clean-neurons/.venv/bin/activate
ulimit -v 2500000
while kill -0 "$1" 2>/dev/null; do sleep 30; done
S1="dyck2_D6_s10 dyck2_D6_s11 dyck2_D6_s12 dyck2_R3a_s10 dyck2_R3a_s11 dyck2_R3a_s12"
python ref.py $S1 > ref_s1.log 2>&1
python soundness.py 14 $S1 > sound_K14_s1.log 2>&1
python soundness.py 13 dyck2_D6_s10 dyck2_R3a_s10 > sound_K13_s1.log 2>&1
python horizon.py 32 100 dyck2_D6_s10 dyck2_R3a_s10 > horizon_s1.log 2>&1
