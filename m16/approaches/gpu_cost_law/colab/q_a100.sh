# A100: T = 5, 6 rungs (float64 mass-heavy). Exact runs use the truth-cone start and sliver bisection (answer-key
# accuracy, leaf sums); recipe108 runs stay estT-identical. Cheapest/most decisive first, (8,6) last.
source /content/cn/m16/approaches/gpu_cost_law/colab/runq.sh
for s in 0 1 2 3 4; do T192_SLIVER=0.6 run 8 5 $s exact 11 cuda --start truth --leafsum; done
run 16 5 0 recipe108 11 cuda
T192_SLIVER=0.6 run 16 5 0 exact 11 cuda --start truth --leafsum
run 16 5 0 exact 11 cuda
for s in 1 2 3 4; do T192_SLIVER=0.6 run 16 5 $s exact 11 cuda --start truth; run 16 5 $s recipe108 11 cuda; done
for s in 0 1 2 3 4; do mc 16 5 $s 10 $((1650+s)); done
T192_SLIVER=0.6 run 4 6 0 exact 11 cuda --start truth --leafsum; mc 4 6 0 10 7046
for s in 0 1 2 3 4; do mc 8 6 $s 10 $((8600+s)); done
T192_SLIVER=0.6 run 8 6 0 exact 11 cuda --start truth
run 8 6 0 recipe108 11 cuda
echo QA100DONE >> /content/qgpu.done; archive
