# T234 large-batch RECIPE VARIANTS (DRI/owner speed-up): two processes on one A100, ~2^20 steps each
C=/content/cn/m16/approaches/capacity_ceiling; cd $C; mkdir -p weights results
arch() { date > results/heartbeat_lb; tar czf /content/t234.tmp.tgz -C /content/cn m16/approaches/capacity_ceiling/results m16/approaches/capacity_ceiling/weights data/capacity_ceiling 2>/dev/null && mv /content/t234.tmp.tgz /content/t234.tgz; }
S8=""; for n in 32 33; do for s in 0 1 2; do S8="$S8 n${n}_sqrt_s$s:$s:$n:2.8284271 n${n}_lin_s$s:$s:$n:8"; done; done
S16=""; for s in 0 1 2; do S16="$S16 n34_sqrt_s$s:$s:34:4 n34_lin_s$s:$s:34:16"; done
( python run_multi.py lb8192 --batch 8192 --spec $S8 > /content/lb8192.log 2>&1; echo "lb8192 rc=$?" >> /content/lb.done; arch ) &
( python run_multi.py lb16384 --batch 16384 --spec $S16 > /content/lb16384.log 2>&1; echo "lb16384 rc=$?" >> /content/lb.done; arch ) &
while sleep 600; do arch; [ $(wc -l < /content/lb.done 2>/dev/null || echo 0) -ge 2 ] && break; done
echo LBDONE >> /content/lb.done; arch
