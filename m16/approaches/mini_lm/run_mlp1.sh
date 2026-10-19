#!/bin/bash
# MLP1 panel (T445): 20 retrains per dyck2 architecture, seeds 100-119, pre-registered recipe unchanged.
# Usage: ./run_mlp1.sh "rnn softmax hard"   (one queue; run two queues for two heavy processes). Skips finished models.
cd "$(dirname "$0")"; source ../../../.venv/bin/activate; mkdir -p models_mlp1 logs
for a in $1; do for s in $(seq 100 119); do
  [ -f models_mlp1/dyck2_${a}_s$s.json ] || (ulimit -v 2500000; python train.py dyck2 $a $s --no_exact --eval_n 100000 --out models_mlp1 >> logs/mlp1.log 2>&1)
done; done
