# L4 (12 vCPU, ~1.5 u/h): geometry-bound rungs with d <= 4 masses. Order: noisy-copy cost (DRI priority), then
# H = 16 per-seed spread, (8,3), MC truths.
source /content/cn/m16/approaches/gpu_cost_law/colab/runq.sh
for H in 8 16; do run $H 4 0 exact 11 cuda --start truth --leafsum
  for nz in rel0.005 rel0.01 rel0.02 abs0.05; do for i in 0 1; do run $H 4 0:${nz}_$i exact 11 cuda --start truth --leafsum; done; done; done
for s in 0 1 2 3 4; do run 16 4 $s exact 11 cuda --start truth; run 16 4 $s recipe108 11 cuda; done
run 16 4 0 exact 11 cuda
for s in 0 1 2 3 4; do run 16 3 $s exact 11 cuda --start truth; run 16 3 $s recipe108 11 cuda; done
run 16 3 0 exact 11 cuda
for s in 0 1 2 3 4; do run 8 3 $s exact 11 cuda --start truth; run 8 3 $s recipe108 11 cuda; done
for s in 0 1 2 3 4; do mc 16 4 $s 10 $((1640+s)); mc 16 3 $s 9 $((1630+s)); mc 8 3 $s 9 $((830+s)); done
echo QL4DONE >> /content/qgpu.done; archive
