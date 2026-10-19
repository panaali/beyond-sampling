#!/bin/bash
# Trains and scores all 36 baselines in two queues (2 heavy processes, 2 threads each, <= 2.5 GB each).
cd "$(dirname "$0")"; source ../../../.venv/bin/activate; mkdir -p models logs
q() { for j in "$@"; do set -- $j; [ -f models/${1}_${2}_s${3}.json ] || (ulimit -v 2500000; ev=; [ -f models/${1}_${2}_s${3}.pt ] && ev=--eval_only; python train.py $1 $2 $3 $ev >> logs/train.log 2>&1); done; }
A=(); B=()
for s in 0 1 2; do for a in rnn softmax hard relu linear pos; do A+=("dyck2 $a $s"); B+=("modsum $a $s"); done; done
# queue split: the 18 dyck2 jobs alternate between the queues, modsum jobs are appended (eval-only jobs are short)
Q1=(); Q2=(); i=0; for j in "${A[@]}" "${B[@]}"; do if ((i++ % 2)); then Q2+=("$j"); else Q1+=("$j"); fi; done
q "${Q1[@]}" & q "${Q2[@]}" & wait
