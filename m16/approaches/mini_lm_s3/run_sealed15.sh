#!/bin/bash
# Addendum A, sealed L = 15 on MLP1 seeds 110-119: exact truths then ce/top at 1e5; 2 queues.
source ../../../.venv/bin/activate; ulimit -v 2500000; export S3_THREADS=2 S3_PLAN=sealed
D=../../../data/t446
q() { for m in "$@"; do
  [ -f $D/${m}_L15_exact.json ] || python exact15.py $m >> $D/sealed_exact.log 2>&1
  python score15.py $m >> $D/sealed_$m.log 2>&1
done; }
A=(); B=()
for a in rnn softmax hard relu linear pos; do for s in 110 111 112 113 114; do A+=(dyck2_${a}_s$s); done; for s in 115 116 117 118 119; do B+=(dyck2_${a}_s$s); done; done
q "${A[@]}" & q "${B[@]}" &
wait
mkdir -p results && cp $D/score15.jsonl results/score15_sealed.jsonl && for f in $D/dyck2_*_s11?_L15_exact.json; do cp $f results/; done
