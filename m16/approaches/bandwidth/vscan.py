"""The dominant sensitivity direction is the tie value v. Scan delta(v) exactly for fixed other inputs, v from
o2 to o1 (the tie stays at ranks 2-3), 240 grid points. Count: sign changes of delta, local extrema of delta,
and distinct tie-point gate patterns along the scan (all per unit of v). Also the lag-correlation of delta(v).
usage: python vscan.py <model> [lines]"""
import sys
import numpy as np
import wtool as W
nm = sys.argv[1] if len(sys.argv) > 1 else "orig"; L = int(sys.argv[2]) if len(sys.argv) > 2 else 150
M = W.get_model(nm)
R = dict(np.load(f"data/{nm}.npz"))
idx = np.nonzero((R["v"] > 0) & (R["o1"] - np.maximum(R["o2"], 0) > 0.6))[0][:L]
K = 240
ext, sgn, pats, lens, ac = [], [], [], [], {0.02: [], 0.05: [], 0.1: [], 0.2: []}
allD = []
for i in idx:
    y, a, b, o1, o2 = R["y"][i], R["a"][i], R["b"][i], R["o1"][i], R["o2"][i]
    lo = max(o2, 0.0) + 0.02; hi = o1 - 0.02
    vs = np.linspace(lo, hi, K)
    Y = np.repeat(y[None], K, 0); Y[:, a] = vs; Y[:, b] = vs
    tau = W.SQ2 * np.minimum(o1 - vs, vs - o2)
    out = W.walk(M, Y, np.full(K, a), np.full(K, b), tau)
    d = np.clip(out["delta"], -0.3, 0.3)
    G = W.jac_at_tie(M, Y, np.full(K, a), np.full(K, b))["G"].reshape(K, -1)
    nchg = (G[1:] != G[:-1]).any(1).sum()
    dd = np.diff(d)
    e = ((dd[1:] * dd[:-1]) < 0).sum()
    ext.append(e); sgn.append((np.sign(d[1:]) != np.sign(d[:-1])).sum()); pats.append(nchg); lens.append(hi - lo)
    dz = (d - d.mean()) / (d.std() + 1e-12); step = (hi - lo) / (K - 1)
    for lag in ac:
        j = int(round(lag / step))
        if 0 < j < K - 10:
            ac[lag].append(np.mean(dz[j:] * dz[:-j]))
lens = np.array(lens)
print(f"{nm}: {len(idx)} scans, mean v-range {lens.mean():.2f}")
print("  per unit v: local extrema of delta %.1f, sign changes %.1f, tie-gate pattern changes %.1f" %
      (np.sum(ext) / lens.sum(), np.sum(sgn) / lens.sum(), np.sum(pats) / lens.sum()))
print("  autocorrelation of delta(v) at lag:", {k: round(float(np.mean(v)), 2) for k, v in ac.items()})
