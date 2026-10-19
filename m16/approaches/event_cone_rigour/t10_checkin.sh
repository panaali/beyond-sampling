#!/bin/bash
# T416 worker check-in (run every ~2 h). Keeps at most 2 tiered10 processes alive for this worker's shard list,
# restarts dead ones (resumable), and pushes finished/partial result JSONs to claude/t400-bound-rigour.
# usage: bash t10_checkin.sh SHARDS_FILE    (each line: SURV_FILE I0 I1)
set -u
cd "$(dirname "$0")"
SHARDS=$1; LOG=t10_worker_logs; mkdir -p $LOG
source ../../../.venv/bin/activate
running=0
while read -r F I0 I1; do
  name=$(basename $F .npz); out=results_t10_${name}_2.5_${I0}.json; tag=${name}_${I0}
  if grep -q "^DONE" $LOG/$tag.log 2>/dev/null; then continue; fi
  pidf=$LOG/$tag.pid
  if [ -f $pidf ] && kill -0 $(cat $pidf) 2>/dev/null && [ "$(readlink /proc/$(cat $pidf)/cwd)" = "$PWD" ]; then running=$((running+1)); continue; fi
  if [ $running -lt 2 ]; then
    (ulimit -v 3500000; PYTHONPATH=../.. nohup python tiered10.py $F 2.5 $I0 $I1 >> $LOG/$tag.log 2>&1 & echo $! > $pidf)
    running=$((running+1)); echo "started $tag"
  fi
done < $SHARDS
git add -f results_t10_*.json 2>/dev/null
git commit -qm "T416 worker checkpoint $(hostname) $(date -u +%H:%M)" 2>/dev/null && git pull -q --no-rebase origin claude/t400-bound-rigour && git push -q origin claude/t400-bound-rigour
echo "running=$running"; grep -h "^DONE" $LOG/*.log 2>/dev/null
