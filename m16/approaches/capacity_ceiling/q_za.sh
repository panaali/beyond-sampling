# T234 (a) data axis at zoo settings (batch 1024, fixed 2^14 warm-up), 2^31/2^32/2^33 x seeds 0-4 packed in one process
C=/content/cn/m16/approaches/capacity_ceiling; cd $C; mkdir -p weights results
arch() { date > results/heartbeat_za; tar czf /content/t234.tmp.tgz -C /content/cn m16/approaches/capacity_ceiling/results m16/approaches/capacity_ceiling/weights data/capacity_ceiling 2>/dev/null && mv /content/t234.tmp.tgz /content/t234.tgz; }
S=""; for n in 31 32 33; do for s in 0 1 2 3 4; do S="$S n${n}_s$s:$s:$n:1:16384"; done; done
( python run_multi.py za --batch 1024 --data-seed 7 --spec $S > /content/za.log 2>&1; echo "za rc=$?" >> /content/za.done; arch ) &
while sleep 600; do arch; [ -s /content/za.done ] && break; done
echo ZADONE >> /content/za.done; arch
