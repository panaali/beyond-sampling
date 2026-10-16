"""Cross-model diagnostics: (a) flip rate when inputs of rank>3 are resampled below m3,
(b) cut-HDMR additivity R2 of the margin change in those inputs near 2-3 ties,
(c) per-neuron gate unpredictability given (previous gate pattern, rank symbol, sign), top 5 neurons."""
import sys; sys.path.insert(0, "/home/user/clean-neurons/m16")
import numpy as np
from arc_eval import forward, Wi0, Wh0, Wo0, noisy_models
from scipy.special import ndtr, ndtri
def zoo(sd):
    from alg_zoo import zoo_2nd_argmax
    m = zoo_2nd_argmax(16, 10, seed=sd); f = lambda t: t.detach().numpy().astype(np.float64)
    return f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight)
models = {"orig": (Wi0, Wh0, Wo0)}
for sd in range(5): models[f"seed{sd}"] = zoo(sd)
for name, M in models.items():
    rng = np.random.default_rng(8); N = 200_000
    x = rng.standard_normal((N, 10)); o = np.argsort(-x, 1); tgt = o[:, 1]
    m3 = np.take_along_axis(x, o[:, 2:3], 1)
    def redraw(x):
        y = x.copy(); u = rng.random((N, 7)) * ndtr(m3); np.put_along_axis(y, o[:, 3:], ndtri(u), 1); return y
    def margin(z):
        L = forward(M, z); lt = L[np.arange(N), tgt].copy(); L[np.arange(N), tgt] = -np.inf; return lt - L.max(1)
    ok = margin(x) > 0; a = redraw(x); b = redraw(x)
    flip = np.mean(ok != (margin(a) > 0))
    ma, mb = margin(a), margin(b); madd = ma.copy()
    for r in range(3, 10):
        y = a.copy(); p = o[:, r]; y[np.arange(N), p] = b[np.arange(N), p]; madd += margin(y) - ma
    g = (np.take_along_axis(x, o[:, 1:2], 1) - m3)[:, 0] < 0.15
    R2 = 1 - (mb - madd)[g].var() / (mb - ma)[g].var()
    Wi, Wh, _ = M; h = np.zeros((N, 16)); B = []
    for t in range(10):
        pre = h @ Wh.T + x[:, t:t+1] * Wi[None]; B.append(pre > 0); h = np.maximum(pre, 0)
    B = np.stack(B, 1); G = B @ (1 << np.arange(16))
    rel = np.array([(x[:, :t+1] > x[:, t:t+1]).sum(1) + 1 for t in range(10)]).T; r = np.minimum(rel, 4) * 2 + (x > 0)
    unc = np.zeros(16)
    for t in range(1, 10):
        _, inv = np.unique(G[:, t-1] * 16 + r[:, t], return_inverse=True); cnt = np.bincount(inv)
        on = np.stack([np.bincount(inv, B[:, t, j]) for j in range(16)], 1); unc += np.minimum(on, cnt[:, None] - on).sum(0) / N
    top = np.argsort(-unc)[:5]
    print(f"{name}: err {1-ok.mean():.4f} flip(rank>3 resampled) {flip:.4f} additive R2 {R2:.2f} gate-unpred total {unc.sum():.2f}/seq, top {[(int(i), round(float(unc[i]),2)) for i in top]}", flush=True)
