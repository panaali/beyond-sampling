"""T234: fused kernel vs trainer.py (autograd) on identical external inputs, then speed."""
import sys, time, numpy as np, torch as th
sys.path.insert(0, "/content/cn/m16/approaches/capacity_ceiling")
import trainer as TR, fused as F
M, B, steps = 3, 1024, 6
X = th.randn(steps, M, B, 10, device="cuda")
Xp = th.full((steps, M, B, 16), float("-inf"), device="cuda"); Xp[..., :10] = X
it = iter(list(X))
TR.sample = lambda K, B_, T_, gen, dev, *a: next(it)
ref = TR.Trainer(16, 10, [0, 1, 2], 2 ** 22, warmup=2, device="cuda", graph=False)
fu = F.FusedTrainer([0, 1, 2], 2 ** 22, warmup=2)
for s in range(steps):
    ref.step(); fu.step(1, Xp[s].contiguous())
    a = [p.detach() for p in ref.params]; b = [fu.Wi, fu.Wh, fu.Wo[:, :10]]
    rel = max(float((x - y).abs().max() / x.abs().max()) for x, y in zip(a, b))
    print("step", s + 1, "max rel param diff %.2e" % rel, "loss ref", [round(v, 6) for v in ref.loss.tolist()], flush=True)
print("fused loss sums", fu.lossacc[:, 0].tolist())
for M2 in (5, 40):
    f2 = F.FusedTrainer(list(range(M2)), 2 ** 30)
    f2.step(4); th.cuda.synchronize(); t = time.time(); f2.step(256); th.cuda.synchronize()
    dt = (time.time() - t) / 256
    print("M", M2, "fused ms/step %.3f (%.4f per model-step)" % (dt * 1e3, dt * 1e3 / M2), "acc", (f2.lossacc[:, 1] / 260).tolist()[:3], flush=True)
