"""T417: stratified Q-level estimate. The simplex is cut into 2^K pieces by longest-edge bisection (exact halves of the
volume), NS uniform points per piece; Q = c0 vol mean_pieces mean_points f. Much lower variance than qcheck.py's plain
MC when the integrand varies over many orders of magnitude (far / adversarial cases).
usage: python qstrat.py flagged K NS        re-estimate the flagged cases of results/qcheck_{2,3}.json
       python qstrat.py run N SEED K NS     fresh cases as qcheck.py, every case recorded -> results/qstrat_<seed>.json"""
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from flint import fmpq  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "event_cone_rigour"))
sys.path.insert(0, HERE)
import m5t_fast as F  # noqa: E402

import gen  # noqa: E402
from qcheck import log_zint  # noqa: E402


def pieces(M, K):
    P = [M]
    for _ in range(K):
        Q = []
        for S in P:
            V = S.shape[0]
            best = (-1, 0, 1)
            for i in range(V):
                for j in range(i + 1, V):
                    e = float(((S[i] - S[j]) ** 2).sum())
                    if e > best[0]:
                        best = (e, i, j)
            _, i, j = best
            mid = 0.5 * (S[i] + S[j])
            A = S.copy(); A[i] = mid; B = S.copy(); B[j] = mid
            Q += [A, B]
        P = Q
    return np.array(P)


def log_q(M, d, mlow, jp3, t0, vol, K, NS, rng):
    """log Q estimate and its relative standard error (stratified: equal-volume pieces)."""
    Ps = pieces(M, K)
    n, V, _ = Ps.shape
    lam = rng.dirichlet(np.ones(V), (n, NS))
    m = np.einsum("pkv,pvd->pkd", lam, Ps).reshape(-1, M.shape[1])
    lz = np.concatenate([log_zint((c * c).sum(1), c[:, jp3], d, mlow, t0) for c in np.array_split(m, max(1, len(m) // 4096))])
    lz = lz.reshape(n, NS)
    mx = lz.max(); w = np.exp(lz - mx)
    mp_ = w.mean(1); vp = w.var(1, ddof=1) / NS
    mean = mp_.mean(); se = math.sqrt(vp.sum()) / n
    return -0.5 * d * math.log(2 * math.pi) + math.log(float(vol)) + mx + math.log(mean), se / mean


def flagged(K, NS):
    rng = np.random.default_rng(1)
    out = []
    for s in (2, 3):
        for w in json.load(open(os.path.join(HERE, "results", f"qcheck_{s}.json")))["worst"]:
            M = np.array(w["M"]); d, mlow = w["d"], w["mlow"]
            jp3 = None
            # recover jp3 / jp2 / volume from the case: the slice pair is the column pair with m_p2 = 1 + m_p3
            for a in range(d):
                for b in range(d):
                    if a != b and np.all(M[:, a] == 1 + M[:, b]):
                        jp2, jp3 = a, b
            vol = gen.volume(M, jp2)
            zr = fmpq(*[int(v) for v in w["zratio"].split("/")]) if "/" in w["zratio"] else fmpq(int(w["zratio"]))
            P = F.Params(d, mlow, jp3, w["t0"], w["zbins"], zr)
            vlo, vhi = F.q2iv(vol)
            lo, hi = F.bounds_batch(M[None], M[None], np.array([vlo]), np.array([vhi]), P, mc=np.array(w["mc"])[None])
            lq, rse = log_q(M, d, mlow, jp3, w["t0"], vol, K, NS, rng)
            r = dict(family=w["family"], hi_over_q=math.exp(min(700, math.log(hi[0]) - lq)) if hi[0] > 0 else 0.0,
                     lo_over_q=math.exp(min(700, math.log(lo[0]) - lq)) if lo[0] > 0 else 0.0, rse=rse,
                     old_lo_over_q=w["lo_over_q"], old_rse=w["rse"])
            out.append(r); print(r, flush=True)
    json.dump(out, open(os.path.join(HERE, "results", f"qstrat_flagged_{K}_{NS}.json"), "w"), indent=0)


def run(N, seed, K, NS):
    rng = np.random.default_rng(417_950 + seed)
    seeds = gen.Seeds(); fams = [f for f in gen.FAMILIES if f != "under"]
    rows = []; i = 0; t0_ = time.time()
    while len(rows) < N:
        f = fams[i % len(fams)]; i += 1
        c = gen.make(f, rng, seeds)
        if c is None or c["vol"] == 0 or not np.all(np.isfinite(c["mc"])):
            continue
        P = F.Params(c["d"], c["mlow"], c["jp3"], c["t0"], c["zbins"], c["zratio"])
        try:
            lo, hi = F.bounds_batch(c["Mlo"][None], c["Mhi"][None], np.array([c["vlo"]]), np.array([c["vhi"]]), P, mc=c["mc"][None])
        except Exception:  # noqa: BLE001
            continue
        lq, rse = log_q(c["M"], c["d"], c["mlow"], c["jp3"], c["t0"], c["vol"], K, NS, rng)
        lh = math.log(hi[0]) - lq if hi[0] > 0 else -math.inf
        ll = math.log(lo[0]) - lq if lo[0] > 0 else -math.inf
        rows.append((f, lh, ll, rse))
    fl = [r for r in rows if (r[1] < math.log(max(1e-300, 1 - 6 * r[3] - 1e-6))) or (r[2] > math.log(1 + 6 * r[3] + 1e-6))]
    good = [r for r in rows if r[3] < 0.01]
    res = dict(cases=len(rows), flags=len(fl), flagged=fl[:50], rse_lt_1pct=len(good),
               min_hi_over_q_rse_lt_1pct=math.exp(min(r[1] for r in good)) if good else None,
               max_lo_over_q_rse_lt_1pct=math.exp(max(r[2] for r in good)) if good else None,
               rse_quantiles=[float(v) for v in np.quantile([r[3] for r in rows], [0.5, 0.9, 0.99, 1])],
               elapsed=time.time() - t0_)
    json.dump(res, open(os.path.join(HERE, "results", f"qstrat_{seed}.json"), "w"), indent=0, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "flagged"}))


if __name__ == "__main__":
    if sys.argv[1] == "flagged":
        flagged(int(sys.argv[2]), int(sys.argv[3]))
    else:
        run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
