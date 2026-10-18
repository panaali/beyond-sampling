"""T367 (sealing third party for the T286 child; DRI): t364 sampler plus the share of hits with g12 = v1 - v2 < .1.
Original: T364 deep-tail cross-check (request of session_01HPuYWfHz2C9jVqL15Q2rZ1, pre-reg claude/t286-cheap-frontier b36a278f
rare_events/README.md "T364"): exact conditional sampler of x ~ N(0, I_10) given g23 = v2 - v3 > t0, GPU float64.
v2 ~ density 90 phi(v2) Phi(-v2) Phi(v2 - t0)^8 (inverse CDF on a 2e6-point grid of [-8, 8], linear interpolation);
v1 = -ndtri(Phi(-v2) U) (N truncated above v2); v3 = ndtri(Phi(v2 - t0) U^(1/8)); 7 lows = ndtri(Phi(v3) U); the 10
values go to a uniform random permutation of the positions. Estimate p = P(g23 > t0) * mean(pick == pos(v3)), with
P(g23 > t0) = the grid integral of the density (trapezoid). One CUDA Philox generator per pair, seed 3640000 + index.
usage: python tail_mc.py weights.npz NAME:T0:N ... -> tail_mc.json"""
import json, math, os, sys, time
import numpy as np, torch as th
DEV = os.environ.get("TAIL_DEV", "cuda"); B = 1 << 22 if DEV == "cuda" else 1 << 16
z = np.load(sys.argv[1]); out = {}
Phi = lambda v: 0.5 * th.erfc(-v / math.sqrt(2))
for k, arg in enumerate(sys.argv[2:]):
    nm, t0, n = arg.split(":"); t0 = float(t0); n = int(float(n)); i = list(z["names"]).index(nm)
    Wi, Wh, Wo = (th.as_tensor(z[a][i], dtype=th.float64, device=DEV) for a in ("Wi", "Wh", "Wo"))
    v = th.linspace(-8, 8, 2_000_001, dtype=th.float64, device=DEV)
    logf = math.log(90) - v ** 2 / 2 - 0.5 * math.log(2 * math.pi) + th.log(Phi(-v)) + 8 * th.log(Phi(v - t0))
    f = th.exp(logf); cdf = th.cat([th.zeros(1, dtype=th.float64, device=DEV), th.cumsum((f[1:] + f[:-1]) / 2 * (v[1] - v[0]), 0)])
    Pg = float(cdf[-1]); cdf = cdf / cdf[-1]
    seed = int(os.environ.get("SEED0", "3670000")) + k; g = th.Generator(device=DEV); g.manual_seed(seed); hit = 0; hit12 = 0; done = 0; t1 = time.time()
    while done < n:
        m = min(B, n - done); U = th.rand(m, 11, generator=g, device=DEV, dtype=th.float64)
        j = th.searchsorted(cdf, U[:, 0].contiguous()).clamp(1, len(v) - 1); w = (U[:, 0] - cdf[j - 1]) / (cdf[j] - cdf[j - 1]).clamp_min(1e-300)
        v2 = v[j - 1] + w * (v[j] - v[j - 1])
        v1 = -th.special.ndtri(Phi(-v2) * U[:, 1]); v3 = th.special.ndtri(Phi(v2 - t0) * U[:, 2] ** 0.125)
        lows = th.special.ndtri(Phi(v3)[:, None] * U[:, 3:10])
        vals = th.cat([v1[:, None], v2[:, None], v3[:, None], lows], 1)          # slot 2 = v3
        perm = th.rand(m, 10, generator=g, device=DEV, dtype=th.float64).argsort(1)  # position of slot s = perm[:, s]
        x = th.empty_like(vals); x.scatter_(1, perm, vals)
        h = th.zeros(m, 16, dtype=th.float64, device=DEV)
        for t in range(10):
            h = th.relu(h @ Wh.T + x[:, t:t + 1] * Wi[None])
        ok = (h @ Wo.T).argmax(1) == perm[:, 2]; hit += int(ok.sum()); hit12 += int((ok & (v1 - v2 < 0.1)).sum()); done += m
    q = hit / n
    out[f"{nm}@{t0:g}"] = dict(model=nm, t0=t0, n=n, hits=hit, cond_rate=q, hits_g12_lt_0p1=hit12, share_g12_lt_0p1=hit12 / max(hit, 1), P_g23=Pg, p=Pg * q,
                               se=Pg * math.sqrt(max(q * (1 - q), 1e-300) / n), seed=seed, precision="float64", sec=time.time() - t1)
    print(json.dumps(out[f"{nm}@{t0:g}"]), flush=True)
    json.dump(out, open(os.environ.get("OUT", "tail_mc.json"), "w"), indent=1)
