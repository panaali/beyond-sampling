#!/bin/bash
# T103 diagnostics D0-D2 (single core, sequential). Run from the repo root.
source .venv/bin/activate; export OMP_NUM_THREADS=1
D=data/skeleton_quadrature; A=m16/approaches/skeleton_quadrature
for m in dev_s0.005_0 dev_s0.005_1; do python $A/d0_inner.py $m 300 12 > $D/log_d0_$m.txt 2>&1; done
for m in dev_s0.005_0 dev_s0.005_1; do python $A/d1_shape.py $m 20000 512 60 1024 > $D/log_d1_$m.txt 2>&1; done
for m in dev_s0.005_0 dev_s0.005_1; do python $A/d2_positions.py $m 24 256 > $D/log_d2_$m.txt 2>&1; done
# D3 (rules vs MC on the skeleton integral), with sampling truths
for m in dev_s0.005_0 dev_s0.005_1; do python $A/truth.py $m 20 > $D/log_truth_$m.txt 2>&1; python $A/d3_rules.py $m 8 128 2,4,8 > $D/log_d3_$m.txt 2>&1; done
# (the line above was stopped after dev_s0.005_0; for dev_s0.005_1 only the truth was kept)
# D3b deterministic product rules; D3 with 24 randomisations of the main rules at k = 4
for m in dev_s0.005_0 dev_s0.005_1; do python $A/d3b_product.py $m 128 > $D/log_d3b_$m.txt 2>&1; done
for m in dev_s0.005_0 dev_s0.005_1; do python $A/d3_rules.py $m 24 128 4 mc,strat,sob4,shared3,pertri3 _R24 > $D/log_d3R24_$m.txt 2>&1; done
