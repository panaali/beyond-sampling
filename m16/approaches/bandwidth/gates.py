"""H2/H5: which tie-point gate bits does the band offset depend on?
(1) Single-bit gate-flip intervention on the exact linear map at the tie: m, s, delta_lin = -m/s recomputed with
    one (step, neuron) gate flipped. A bit is 'relevant' if |delta_flip - delta| > max(0.5|delta|, 0.005).
(2) Canonical-gate predictor (zero fitted params, deductive): gates taken from the expected-order-statistic input
    with the same full ordering (tie kept), applied to the actual input.
(3) Natural gate diversity: number of distinct relevant-gate patterns per (a,b,p1) class.
usage: python gates.py <model> [N]"""
import sys
import numpy as np
import wtool as W

nm = sys.argv[1] if len(sys.argv) > 1 else "orig"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
M = W.get_model(nm); Wi, Wh, Wo = M
R = dict(np.load(f"data/{nm}.npz"))
keep = np.nonzero(R["v"] > 0)[0][:N]
y, a, b, p1, delta = R["y"][keep], R["a"][keep], R["b"][keep], R["p1"][keep], R["delta"][keep]
n = len(y); rows = np.arange(n)
d = np.zeros_like(y); d[rows, a] = 1 / W.SQ2; d[rows, b] = -1 / W.SQ2
G = np.unpackbits(R["G"][keep], 1)[:, :160].reshape(n, 10, 16).astype(bool)


def lin(G):
    h = np.zeros((n, 16)); dh = np.zeros((n, 16))
    for t in range(10):
        h = np.where(G[:, t], h @ Wh.T + y[:, t:t + 1] * Wi[None], 0)
        dh = np.where(G[:, t], dh @ Wh.T + d[:, t:t + 1] * Wi[None], 0)
    L = h @ Wo.T; dL = dh @ Wo.T
    m = L[rows, a] - L[rows, b]; s = dL[rows, a] - dL[rows, b]
    return -m / s, L


dl0, L0 = lin(G)
print(nm, "n", n, "delta_lin == delta frac", np.mean(np.abs(dl0 - delta) < 1e-6))
rel = np.zeros((n, 10, 16), bool); dch = np.zeros((n, 10, 16))
for t in range(10):
    for i in range(16):
        G2 = G.copy(); G2[:, t, i] ^= True
        dl, _ = lin(G2)
        dch[:, t, i] = dl - dl0
        rel[:, t, i] = np.abs(dl - dl0) > np.maximum(0.5 * np.abs(dl0), 0.005)
nrel = rel.reshape(n, -1).sum(1)
print("relevant bits per line: mean %.1f median %d p10 %d p90 %d" % (nrel.mean(), np.median(nrel),
      np.percentile(nrel, 10), np.percentile(nrel, 90)))
# relevance by step relative to b (and to a)
tt = np.arange(10)[None, :, None]
for nmr, ref in [("b", b), ("a", a), ("p1", p1)]:
    off = (tt - ref[:, None, None])
    out = []
    for o in range(-9, 10):
        msk = np.broadcast_to(off == o, rel.shape)
        if msk.sum():
            out.append(f"{o:+d}:{rel[msk].mean():.2f}")
    print(f"P(bit relevant) by step - {nmr}:", " ".join(out))
print("P(relevant) by neuron:", np.round(rel.mean((0, 1)), 2))
# bits whose natural state actually varies across lines in the same (a,b,p1) class AND are relevant
cls = (a * 10 + b) * 10 + p1
nvary = []; npat = []
for c in np.unique(cls):
    i = cls == c
    if i.sum() < 15:
        continue
    Gi = G[i].reshape(i.sum(), -1); Ri = rel[i].reshape(i.sum(), -1)
    varying = (Gi.mean(0) > 0.02) & (Gi.mean(0) < 0.98)
    nvary.append((varying & (Ri.mean(0) > 0.1)).sum())
    npat.append(len(np.unique(Gi[:, varying], axis=0)) / i.sum())
print("per (a,b,p1) class: #bits that vary (2-98%%) and are relevant in >10%% of lines: mean %.1f" % np.mean(nvary),
      " distinct varying-gate patterns per line: %.2f" % np.mean(npat))
# canonical-gate predictor
ref = np.sort(np.random.default_rng(0).standard_normal((400000, 10)), 1)[:, ::-1].mean(0)
rk = np.argsort(np.argsort(-y + 1e-9 * (np.arange(10) == b[:, None]), 1), 1)  # ranks, tie broken a before b
yc = ref[rk]
mid = (yc[rows, a] + yc[rows, b]) / 2; yc[rows, a] = mid; yc[rows, b] = mid   # keep the tie
Gc = W.jac_at_tie(M, yc, a, b, +1)["G"]
dlc, _ = lin(Gc)
agree = (Gc == G).reshape(n, -1).mean(1)
def r2(p, t):
    return 1 - np.mean((t - p) ** 2) / np.var(t)
cl = lambda z: np.clip(z, -0.3, 0.3)
print("canonical gates: bit agreement %.3f, all-160 agree %.3f" % (agree.mean(), (agree == 1).mean()))
print("R2 on delta (clipped +-0.3): delta_lin(true gates) %.3f, canonical-gate delta %.3f" % (r2(cl(dl0), cl(delta)), r2(cl(dlc), cl(delta))))
print("E|delta| true %.4f  canonical-gate %.4f" % (np.abs(delta).mean(), np.abs(cl(dlc)).mean()))
np.savez_compressed(f"data/gates_{nm}.npz", rel=rel, dch=dch.astype(np.float32), keep=keep, dlc=dlc)
