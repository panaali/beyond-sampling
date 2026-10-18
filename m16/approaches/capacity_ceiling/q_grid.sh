# T234 grid with the fused kernel (16 seeds per config), shares the A100 with T270 shard 0
C=/content/cn/m16/approaches/capacity_ceiling; cd $C; mkdir -p weights results
arch() { date > results/heartbeat_grid; tar czf /content/t234.tmp.tgz -C /content/cn m16/approaches/capacity_ceiling/results m16/approaches/capacity_ceiling/weights 2>/dev/null && mv /content/t234.tmp.tgz /content/t234.tgz; }
G1=""; for s in $(seq 0 15); do
  for n in 30 31 32; do G1="$G1 zoo_n${n}_s$s:$s:$n:1"; done                       # (a) data axis, zoo settings
  G1="$G1 lrx0.5_n32_s$s:$s:32:0.5 lrx2_n32_s$s:$s:32:2 warmx4_n32_s$s:$s:32:1:65536"   # (b) LR / schedule
  G1="$G1 hard_bt_n32_s$s:$s:32:1::1024:0.2:0.2:0.05 hard_b_n32_s$s:$s:32:1::1024:0.2:0:0.05 hard_t_n32_s$s:$s:32:1::1024:0:0.2:0.05"   # (c)
done
G2=""; for s in $(seq 0 15); do G2="$G2 zoo_n33_s$s:$s:33:1"; done
( python fused_run.py grid1 --data-seed 31 --spec $G1 > /content/grid1.log 2>&1; echo "grid1 rc=$?" >> /content/grid.done; arch ) &
( python fused_run.py grid2 --data-seed 32 --spec $G2 > /content/grid2.log 2>&1; echo "grid2 rc=$?" >> /content/grid.done; arch ) &
while sleep 600; do arch; [ $(wc -l < /content/grid.done 2>/dev/null || echo 0) -ge 2 ] && break; done
echo GRIDDONE >> /content/grid.done; arch
