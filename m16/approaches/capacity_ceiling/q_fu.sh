# T234 fused kernel: uncontended timing, then the 2^30 zoo-config validation (5 seeds)
C=/content/cn/m16/approaches/capacity_ceiling; cd $C; mkdir -p weights results
arch() { date > results/heartbeat_fu; tar czf /content/t234.tmp.tgz -C /content/cn m16/approaches/capacity_ceiling/results m16/approaches/capacity_ceiling/weights data/capacity_ceiling 2>/dev/null && mv /content/t234.tmp.tgz /content/t234.tgz; }
python - > /content/fu_timing.log 2>&1 <<'PY'
import sys, time, torch as th
sys.path.insert(0, "/content/cn/m16/approaches/capacity_ceiling")
import fused as F
for M in (5, 40, 108, 216, 432):
    f = F.FusedTrainer([s % 5 for s in range(M)], 2 ** 30); f.step(8); th.cuda.synchronize(); t = time.time(); f.step(512); th.cuda.synchronize()
    dt = (time.time() - t) / 512; print("M %d ms/step %.3f ms/model-step %.4f" % (M, dt * 1e3, dt * 1e3 / M), flush=True)
PY
cp /content/fu_timing.log results/fu_timing.log; arch
python fused_run.py f30 --spec s0:0:30:1 s1:1:30:1 s2:2:30:1 s3:3:30:1 s4:4:30:1 > /content/f30.log 2>&1; arch
echo FUDONE >> /content/f30.log; arch
