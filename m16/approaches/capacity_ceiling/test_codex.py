"""T234/T261 cross-check: torch autograd (trainer.py), fused.py (this lane) and Codex's tiled_step.py (T261,
copied from ccr-6a61f25c-wn80ii:review/fused_rnn_training/tiled_step.py) on identical inputs, then timing."""
import sys, time, torch as th
sys.path.insert(0, "/content/cn/m16/approaches/capacity_ceiling")
import trainer as TR, fused as F, codex_tiled_step as CX
K, B, steps = 3, 1024, 3
X = th.randn(steps, K, B, 10, device="cuda")
Xp = th.full((steps, K, B, 16), float("-inf"), device="cuda"); Xp[..., :10] = X
orig = TR.sample
def feed():
    it = iter(list(X)); TR.sample = lambda K_, B_, T_, gen, dev, *a: next(it)
feed(); ref = TR.Trainer(16, 10, [0, 1, 2], 2 ** 22, warmup=2, device="cuda", graph=False)
for _ in range(steps): ref.step()
feed(); cx = CX.trainer_class(TR)(16, 10, [0, 1, 2], 2 ** 22, warmup=2, device="cuda", graph=False)
for _ in range(steps): cx.step()
fu = F.FusedTrainer([0, 1, 2], 2 ** 22, warmup=2)
for s in range(steps): fu.step(1, Xp[s].contiguous())
P = {"torch": [p.detach() for p in ref.params], "codex": [p.detach() for p in cx.params], "fused": [fu.Wi, fu.Wh, fu.Wo[:, :10]]}
for a, b in (("torch", "fused"), ("torch", "codex"), ("fused", "codex")):
    print(a, "vs", b, "max rel param diff after %d steps: %.2e" % (steps, max(float((x - y).abs().max() / x.abs().max()) for x, y in zip(P[a], P[b]))), flush=True)
TR.sample = orig
for M in (108,):
    c2 = CX.trainer_class(TR)(16, 10, list(range(M)), 2 ** 30, device="cuda")
    for _ in range(5): c2.step()
    th.cuda.synchronize(); t = time.time()
    for _ in range(200): c2.step()
    th.cuda.synchronize(); dt = (time.time() - t) / 200
    print("codex tiled_step M=%d: %.3f ms/step (%.4f ms/model-step)" % (M, dt * 1e3, dt * 1e3 / M), flush=True)
