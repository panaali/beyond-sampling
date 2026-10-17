#!/bin/bash
# Reproduces every number in README.md (single core, sequential). Run from the repo root.
# usage: bash review/stoch_margin/run_all.sh [fid|lev|anova|all]   (fid ~75 min, lev ~35 min, anova ~85 min)
set -e
source .venv/bin/activate
export OMP_NUM_THREADS=1
D=data/review_stoch_margin; mkdir -p $D
S=${1:-all}
if [ $S = fid ] || [ $S = all ]; then
for m in dev_s0.005_0 dev_s0.0025_1 dev_s0.005_1; do for K in 3 4; do
  python review/stoch_margin/fidelity.py $m $K 6000 256 > $D/log_fid_${m}_K$K.txt 2>&1; done; done
for m in dev_s0.005_0 dev_s0.0025_1; do python review/stoch_margin/fidelity.py $m 5 4000 256 > $D/log_fid_${m}_K5.txt 2>&1; done
fi
if [ $S = lev ] || [ $S = all ]; then
for m in dev_s0.005_0 dev_s0.0025_1 dev_s0.005_1; do
  python review/stoch_margin/leverage.py $m 400000 > $D/log_lev_$m.txt 2>&1
  python review/stoch_margin/closure.py $m 600 1024 > $D/log_clos_$m.txt 2>&1
  for K in 3 4; do python review/stoch_margin/slin_nodes.py $m $K 3000 256 > $D/log_slinn_${m}_K$K.txt 2>&1; done
done
FRESH_GZ=1 python review/stoch_margin/fidelity.py dev_s0.005_0 3 2000 256 7 > $D/log_fid_fresh.txt 2>&1
fi
if [ $S = anova ] || [ $S = all ]; then
for m in dev_s0.005_0 dev_s0.0025_1 dev_s0.005_1; do python review/stoch_margin/anova.py $m 3 800 256 256 24 > $D/log_anova_${m}_K3.txt 2>&1; done
for m in dev_s0.005_0 dev_s0.0025_1; do python review/stoch_margin/anova.py $m 4 600 256 256 24 > $D/log_anova_${m}_K4.txt 2>&1; done
fi
