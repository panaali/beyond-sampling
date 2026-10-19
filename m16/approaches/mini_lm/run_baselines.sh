#!/bin/bash
# Trains and scores all 36 baselines in two queues (2 heavy processes, 2 threads each, <= 2.5 GB each).
cd "$(dirname "$0")"; source ../../../.venv/bin/activate; mkdir -p models logs
q() { for j in "$@"; do set -- $j; [ -f models/${1}_${2}_s${3}.json ] || (ulimit -v 2500000; python train.py $1 $2 $3 >> logs/train.log 2>&1); done; }
A=(); B=()
for s in 0 1 2; do for a in rnn softmax hard relu linear pos; do A+=("dyck2 $a $s"); B+=("modsum $a $s"); done; done
q "${A[@]}" & q "${B[@]}" & wait
