# T234 main grid (fused kernel v3, early exit, in-kernel eval): data axis 2^30-2^33 x 100 seeds, 2^34 x 50 seeds,
# LR / warm-up / hard-case variants at 2^31 x 100 seeds. Seeds 100-199 (disjoint from earlier runs' 0-39).
C=/content/cn/m16/approaches/capacity_ceiling; cd $C; mkdir -p weights results
arch() { date > results/heartbeat_grid3; tar czf /content/t234.tmp.tgz -C /content/cn m16/approaches/capacity_ceiling/results 2>/dev/null && mv /content/t234.tmp.tgz /content/t234.tgz; }
S=""; for s in $(seq 100 199); do
  for n in 30 31 32 33; do S="$S zoo_n${n}_s$s:$s:$n:1"; done
  S="$S lrx0.5_n31_s$s:$s:31:0.5 lrx2_n31_s$s:$s:31:2 warmx4_n31_s$s:$s:31:1:65536"
  S="$S hard_bt_n31_s$s:$s:31:1::1024:0.2:0.2:0.05 hard_b_n31_s$s:$s:31:1::1024:0.2:0:0.05 hard_t_n31_s$s:$s:31:1::1024:0:0.2:0.05"
done
for s in $(seq 100 149); do S="$S zoo_n34_s$s:$s:34:1"; done
python fused_run.py grid3 --data-seed 41 --eval kernel --eval-n 1e8 --chunk 8192 --spec $S > /content/grid3.log 2>&1; echo "grid3 rc=$?" >> /content/grid3.done; arch
