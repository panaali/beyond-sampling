"""T417: Q-level check, independent of the M5t formula: Monte Carlo over the simplex of the exact z-integral
Q = c0 vol E_m[ int_{z>=t0} z^(d-1) exp(-z^2 |m|^2 / 2) Phi(z m_p3)^mlow dz ]
(z-integral by composite Gauss-Legendre in log space, m uniform in the simplex). Not rigorous (statistical, float);
it checks that fast's [lo, hi] brackets Q, i.e. also the formula's own derivation (T400), on the adversarial cases.
usage: python qcheck.py N SEED [NM]  -> results/qcheck_<seed>.json"""
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from scipy.special import log_ndtr, logsumexp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "event_cone_rigour"))
sys.path.insert(0, HERE)
import m5t_fast as F  # noqa: E402

import gen  # noqa: E402

GX, GW = np.polynomial.legendre.leggauss(16)


def log_zint(s, y, d, mlow, t0):
    """log int_t0^inf z^(d-1) e^(-z^2 s/2) Phi(z y)^mlow dz for arrays s, y (composite Gauss-Legendre)."""
    zpk = np.sqrt((d - 1) / s)                                      # peak of z^(d-1) e^(-z^2 s/2)
    zmax = np.maximum(t0, zpk) + 12.0 / np.sqrt(s) + 1e-12
    zmax = np.maximum(zmax, np.sqrt(t0 ** 2 + 2 * 60.0 / s))
    P = 96
    edges = t0 + (zmax - t0)[:, None] * (np.arange(P + 1) / P) ** 2   # refined near t0
    a, b = edges[:, :-1], edges[:, 1:]
    z = (0.5 * (b - a))[..., None] * GX + (0.5 * (a + b))[..., None]
    lw = np.log(0.5 * (b - a))[..., None] + np.log(GW)
    lf = (d - 1) * np.log(z) - 0.5 * z * z * s[:, None, None]
    if mlow:
        lf = lf + mlow * log_ndtr(z * y[:, None, None])
    return logsumexp((lf + lw).reshape(len(s), -1), axis=1)


def main():
    N, seed = int(sys.argv[1]), int(sys.argv[2]); NM = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    rng = np.random.default_rng(417_900 + seed)
    seeds = gen.Seeds()
    fams = [f for f in gen.FAMILIES if f != "under"]
    st = dict(cases=0, viol_hi=0, viol_lo=0, skipped=0, min_hi_over_q=math.inf, max_lo_over_q=0.0, worst=[], per_family={})
    t0_ = time.time(); i = 0
    while st["cases"] < N:
        f = fams[i % len(fams)]; i += 1
        c = gen.make(f, rng, seeds)
        if c is None or c["vol"] == 0 or not np.all(np.isfinite(c["mc"])):
            continue
        P = F.Params(c["d"], c["mlow"], c["jp3"], c["t0"], c["zbins"], c["zratio"])
        lo, hi = F.bounds_batch(c["Mlo"][None], c["Mhi"][None], np.array([c["vlo"]]), np.array([c["vhi"]]), P, mc=c["mc"][None])
        lo, hi = float(lo[0]), float(hi[0])
        M = c["M"]; V = M.shape[0]
        lam = rng.dirichlet(np.ones(V), NM)
        m = lam @ M
        s = (m * m).sum(1); y = m[:, c["jp3"]]
        lz = log_zint(s, y, c["d"], c["mlow"], c["t0"])
        lc = -0.5 * c["d"] * math.log(2 * math.pi) + math.log(float(c["vol"]))
        mx = lz.max(); wv = np.exp(lz - mx)
        mean = wv.mean(); se = wv.std(ddof=1) / math.sqrt(NM)
        lq = lc + mx + math.log(mean)
        if not np.isfinite(lq) or mean <= 0:
            st["skipped"] += 1; continue
        st["cases"] += 1
        pf = st["per_family"].setdefault(f, dict(cases=0, viol=0, min_hi_over_q=math.inf, max_lo_over_q=0.0))
        pf["cases"] += 1
        rse = se / mean
        lhi = math.log(hi) if hi > 0 else -math.inf
        llo = math.log(lo) if lo > 0 else -math.inf
        r_hi = math.exp(min(700, lhi - lq)) if hi > 0 else 0.0
        r_lo = math.exp(min(700, llo - lq)) if lo > 0 else 0.0
        st["min_hi_over_q"] = min(st["min_hi_over_q"], r_hi); pf["min_hi_over_q"] = min(pf["min_hi_over_q"], r_hi)
        st["max_lo_over_q"] = max(st["max_lo_over_q"], r_lo); pf["max_lo_over_q"] = max(pf["max_lo_over_q"], r_lo)
        vh = r_hi < 1 - 6 * rse - 1e-6; vl = r_lo > 1 + 6 * rse + 1e-6
        if vh or vl:
            st["viol_hi"] += vh; st["viol_lo"] += vl; pf["viol"] += 1
            if len(st["worst"]) < 50:
                st["worst"].append(dict(family=f, hi_over_q=r_hi, lo_over_q=r_lo, rse=rse, d=c["d"], mlow=c["mlow"], t0=c["t0"],
                                        M=M.tolist(), mc=c["mc"].tolist(), zbins=c["zbins"], zratio=str(c["zratio"])))
    st["elapsed"] = time.time() - t0_
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(st, open(os.path.join(HERE, "results", f"qcheck_{seed}.json"), "w"), indent=0)
    print(json.dumps({k: v for k, v in st.items() if k != "worst"}))


if __name__ == "__main__":
    main()
