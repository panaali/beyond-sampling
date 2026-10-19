#!/bin/bash
# usage: lane.sh "list of point indices"   (skips points whose json exists)
source /home/user/clean-neurons/.venv/bin/activate; cd /home/user/clean-neurons/m16/approaches/dominating_is2; ulimit -v 2500000
for i in $1; do [ -f t403_$i.json ] && continue; OMP_NUM_THREADS=1 python run2.py t403 $i >> lane.log 2>&1; done
