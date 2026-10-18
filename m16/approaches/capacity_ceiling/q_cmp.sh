# T234 kernel validation at scale: 40 seeds x 2^30 zoo config, fused kernel (Philox data) vs trainer.py (torch data)
C=/content/cn/m16/approaches/capacity_ceiling; cd $C; mkdir -p weights results
arch() { date > results/heartbeat_cmp; tar czf /content/t234.tmp.tgz -C /content/cn m16/approaches/capacity_ceiling/results m16/approaches/capacity_ceiling/weights data/capacity_ceiling 2>/dev/null && mv /content/t234.tmp.tgz /content/t234.tgz; }
SF=""; for s in $(seq 0 39); do SF="$SF s$s:$s:30:1"; done
( python fused_run.py cmp_fused --data-seed 21 --eval-n 1e7 --spec $SF > /content/cmp_fused.log 2>&1; echo "cmp_fused rc=$?" >> /content/cmp.done; arch ) &
( python run.py cmp_torch --log2n 30 --seeds $(seq 0 39) --data-seed 21 --eval-n 1e7 > /content/cmp_torch.log 2>&1; echo "cmp_torch rc=$?" >> /content/cmp.done; arch ) &
wait; echo CMPDONE >> /content/cmp.done; arch
