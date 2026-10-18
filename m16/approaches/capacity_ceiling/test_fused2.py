"""T234: fused kernel v2 tests: (1) external-input equivalence at batch 1024 and 2048; (2) hard-case data dump stats."""
import sys, numpy as np, torch as th
sys.path.insert(0, "/content/cn/m16/approaches/capacity_ceiling")
import trainer as TR, fused as F
for B in (1024, 2048):
    M, steps = 2, 5
    X = th.randn(steps, M, B, 10, device="cuda")
    Xp = th.full((steps, M, B, 16), float("-inf"), device="cuda"); Xp[..., :10] = X
    it = iter(list(X)); TR.sample = lambda K, B_, T_, gen, dev, *a: next(it)
    ref = TR.Trainer(16, 10, [0, 1], 2 ** 22, warmup=2, batch=B, device="cuda", graph=False)
    fu = F.FusedTrainer([0, 1], 2 ** 22, warmup=2, batch=B)
    for s in range(steps):
        ref.step(); fu.step(1, Xp[s].contiguous())
    rel = max(float((x.detach() - y).abs().max() / x.abs().max()) for x, y in zip(ref.params, [fu.Wi, fu.Wh, fu.Wo[:, :10]]))
    print("B", B, "max rel param diff after", steps, "steps: %.2e" % rel, flush=True)
fu = F.FusedTrainer([0, 1], 2 ** 30, batch=[1024, 2048], f_blind=[0.25, 0.0], f_tie=[0.25, 0.2], gap=[0.05, 0.05])
D = th.zeros(2 * fu.B * 16, device="cuda"); fu.step(1, dump=D); D = D.view(2, fu.B, 16).cpu().numpy()
for m, (Bm, fb, ft) in enumerate([(1024, .25, .25), (2048, 0, .2)]):
    x = D[m, :Bm, :10]; lab = D[m, :Bm, 15].astype(int)
    nb, nt = int(round(fb * Bm)), int(round(ft * Bm))
    pos = (x > 0).sum(1); srt = np.sort(x, 1)
    print("model", m, "labels==argsort:", (lab == np.argsort(x, 1)[:, -2]).mean(),
          "| blind part <=1 positive:", (pos[:nb] <= 1).mean() if nb else "-", "| plain part <=1 positive: %.4f" % (pos[nb + nt:] <= 1).mean(),
          "| tie part max gap23: %.4f" % (srt[nb:nb + nt, -2] - srt[nb:nb + nt, -3]).max(),
          "| plain median gap23: %.3f" % np.median(srt[nb + nt:, -2] - srt[nb + nt:, -3]), "| plain mean/std %.3f %.3f" % (x[nb + nt:].mean(), x[nb + nt:].std()), flush=True)
