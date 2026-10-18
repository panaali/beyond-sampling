#!/bin/bash
# upload a queue script and start it detached: launch.sh SESSION local_queue.sh
S=$1; Q=$2; C=/home/user/colabcli-venv/bin/colab; B=$(basename $Q)
timeout 120 $C upload -s $S $Q /content/$B 2>&1 | tail -1
for k in 1 2 3; do
  out=$(printf 'import subprocess as s\ns.Popen("cd /content && nohup bash %s > %s.out 2>&1 &",shell=True)\nprint("LAUNCHED")\n' $B $B | timeout 90 $C exec -s $S 2>&1 | tail -1)
  echo "$out" | grep -q LAUNCHED && { echo launched $B on $S; exit 0; }; sleep 10
done; echo "launch failed"; exit 1
