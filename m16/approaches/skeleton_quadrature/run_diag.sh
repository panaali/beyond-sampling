#!/bin/bash
# T103 diagnostics D0-D2 (single core, sequential). Run from the repo root.
source .venv/bin/activate; export OMP_NUM_THREADS=1
D=data/skeleton_quadrature; A=m16/approaches/skeleton_quadrature
for m in dev_s0.005_0 dev_s0.005_1; do python $A/d0_inner.py $m 300 12 > $D/log_d0_$m.txt 2>&1; done
for m in dev_s0.005_0 dev_s0.005_1; do python $A/d1_shape.py $m 20000 512 60 1024 > $D/log_d1_$m.txt 2>&1; done
for m in dev_s0.005_0 dev_s0.005_1; do python $A/d2_positions.py $m 24 256 > $D/log_d2_$m.txt 2>&1; done
