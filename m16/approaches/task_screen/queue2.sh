#!/bin/bash
# usage: queue2.sh jobs.txt worker_id n_workers   -- runs lines (TASK seed log2n) with index % n_workers == worker_id
# (queue.sh plus pair tasks: names starting with pair_ go to train_pair.py)
cd "$(dirname "$0")"; source ../../../.venv/bin/activate; ulimit -v 2500000
mkdir -p ../../../data/task_screen/logs
i=0
while read -r task seed log2n; do
  [ -z "$task" ] && continue
  if [ $((i % $3)) -eq "$2" ]; then
    f=../../../data/task_screen/${task}_${log2n}_${seed}.npz
    py=train.py; [[ $task == pair_* ]] && py=train_pair.py
    [ -f "$f" ] || python $py "$task" "$seed" "$log2n" > ../../../data/task_screen/logs/${task}_${log2n}_${seed}.log 2>&1
  fi
  i=$((i+1))
done < "$1"
