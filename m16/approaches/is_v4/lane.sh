#!/bin/bash
# usage: lane.sh MODE "indices"   (MODE est|truth; skips existing outputs)
source /home/user/clean-neurons/.venv/bin/activate; cd /home/user/clean-neurons/m16/approaches/is_v4; ulimit -v 2500000
for i in $2; do [ -f ${1}_$i.json ] && continue; OMP_NUM_THREADS=1 python run4.py $1 $i >> lane_$1.log 2>&1; done
