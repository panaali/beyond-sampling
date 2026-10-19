#!/bin/bash
# Pre-registered pipeline for one candidate: LR screen (lr_base 0.04 / 0.012 / 0.004, seed 100, 2^24 samples),
# pick the best 2^20 validation accuracy, then the panel seeds 0, 1, 2 at 2^LOG2N. Two runs at a time (rule 8).
# usage: arch.sh TAG THREADS LOG2N
TAG=$1; TH=$2; LG=$3; DIR=$(cd "$(dirname "$0")" && pwd)
printf "$TAG 100 0.04 24 $TH\n$TAG 100 0.012 24 $TH\n$TAG 100 0.004 24 $TH\n" | xargs -P 2 -L 1 bash "$DIR/job.sh"
LR=$(cd "$DIR" && python -c "
import json,glob
r=[json.load(open(f)) for f in glob.glob('weights/${TAG}_s100_lr*_n24.json')]
assert len(r)==3, r
print(max(r,key=lambda d:d['val_acc_2^20'])['lr_base'])")
echo "$TAG screen -> lr_base $LR"
printf "$TAG 0 $LR $LG $TH\n$TAG 1 $LR $LG $TH\n$TAG 2 $LR $LG $TH\n" | xargs -P 2 -L 1 bash "$DIR/job.sh"
