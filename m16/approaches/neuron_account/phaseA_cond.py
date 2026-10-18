"""Phase A (on-manifold check, DRI request): conditional resampling.

Replace h_t[j] (or a mode's coordinates of h_10) by its value on ANOTHER input from the same stratum, then run
the rest exactly. Strata (donor = random other member of the stratum; singleton strata keep their own value):
  S0 unconditional (as phaseA.py)
  S1 same gate pattern of h_t (16 bits) + same x9 case + same age of the running max at t
  S2 same cell (age1, age2) at t + same quintile bins of m1 and m2 at t
  S3 S1 and S2 together
Steps t = 5 and 10. Modes: final pattern 64975, donors restricted to that pattern and matched by S1 / S3.
N = 400k (seed 241061).
"""
import json
import numpy as np
from nacommon import sample, run, run_from, target, cases, Wo, Wh, T, H
from encode import prefix_stats

N = 400_000
x = sample(N, 241061)
hs, pre = run(x)
ref = (hs[:, -1] @ Wo.T).argmax(1)
y = target(x)
rng = np.random.default_rng(241062)
r9 = np.argsort(np.argsort(-x, 1), 1)[:, 9]
case = np.minimum(r9, 2)


def donors(key):
    """Index of a random other member of the same stratum (self for singletons)."""
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.lexsort((rng.random(len(inv)), inv))
    g = inv[order]
    start = np.r_[0, np.where(np.diff(g))[0] + 1]
    size = np.diff(np.r_[start, len(g)])
    pos = np.arange(len(g)) - np.repeat(start, size)
    sz = np.repeat(size, size)
    nxt = np.repeat(start, size) + (pos + 1) % sz          # cyclic shift inside each group
    d = np.empty(len(g), int)
    d[order] = order[nxt]
    return d, float((sz == 1).mean())


def strata(t):
    m, a, K = prefix_stats(x, t)
    gate = (hs[:, t] > 0) @ (1 << np.arange(H))
    q = lambda v: np.searchsorted(np.quantile(v, [.2, .4, .6, .8]), v)
    s1 = np.stack([gate, case, a[:, 0]], 1)
    s2 = np.stack([a[:, 0], a[:, 1] if t > 1 else a[:, 0], q(m[:, 0]), q(m[:, 1] if t > 1 else m[:, 0])], 1)
    return {"S0": np.zeros((N, 1), int), "S1": s1, "S2": s2, "S3": np.concatenate([s1, s2], 1)}


def met(lg, idx=slice(None)):
    p = lg.argmax(1)
    return {"acc": float((p == y[idx]).mean()), "agree": float((p == ref[idx]).mean())}


out = {"N": N, "base_acc": float((ref == y).mean()), "neurons": {}, "modes64975": {}}
for t in (5, 10):
    S = strata(t)
    D = {k: donors(v) for k, v in S.items()}
    out["neurons"][t] = {k: {"singleton_frac": D[k][1]} for k in S}
    for k, (d, _) in D.items():
        for j in range(H):
            h = hs[:, t].copy()
            h[:, j] = hs[d, t, j]
            out["neurons"][t][k][f"n{j}"] = met(run_from(h, x, t))
        print(t, k, "singletons %.3f" % D[k][1],
              " ".join(f"n{j}:{out['base_acc'] - out['neurons'][t][k][f'n{j}']['acc']:.3f}" for j in range(H)), flush=True)

# modes of final pattern 64975 with matched donors
g = 64975
gate10 = (pre[:, T] > 0) @ (1 << np.arange(H))
idx = np.where(gate10 == g)[0]
act = np.array([j for j in range(H) if g >> j & 1])
lam, R = np.linalg.eig(Wh[np.ix_(act, act)])
L = np.linalg.inv(R)
used, modes = set(), []
for k in np.argsort(-np.abs(lam)):
    if k in used:
        continue
    ks = [k] + ([m for m in range(len(lam)) if m not in used and m != k and abs(lam[m] - lam[k].conj()) < 1e-8][:1]
                if abs(lam[k].imag) > 1e-9 else [])
    used.update(ks)
    modes.append(ks)
S = strata(T)
for k in ("S0", "S1", "S3"):
    d, sing = donors(S[k][idx])
    out["modes64975"][k] = {"singleton_frac": sing}
    for ks in modes:
        P = (R[:, ks] @ L[ks, :]).real
        l = lam[ks[0]]
        name = f"|l|={abs(l):.3f}" + (f",period={2 * np.pi / abs(np.angle(l)):.2f}" if abs(l.imag) > 1e-9 else ",real")
        h = hs[idx, T].copy()
        hd = hs[idx[d], T]
        h[:, act] += (hd[:, act] - h[:, act]) @ P.T
        out["modes64975"][k][name] = met(h @ Wo.T, idx)
    print("modes", k, "singletons %.3f" % sing, {n: round(v["acc"], 3) for n, v in out["modes64975"][k].items() if n != "singleton_frac"}, flush=True)
json.dump(out, open("results/phaseA_cond.json", "w"), indent=1)
