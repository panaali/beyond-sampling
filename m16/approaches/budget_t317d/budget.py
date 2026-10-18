"""T324 per-approximation error budget of T317d (DIAGNOSTIC SAMPLING, paired). Pointwise form of the estimator on
exact-law inputs, with each declared approximation switchable (README). Own code; the candidate is imported only for
its transverse points (cont_d.TABLES) and for est (consistency check).  usage: python budget.py [N=200000]"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.environ.get("CAND", "/tmp/claude-0/-home-user-clean-neurons/d43de5b0-d4b8-5960-94ba-74926c630209/scratchpad/t317d/m16/approaches/r1_cont")
sys.path.insert(0, CAND)
import cont_d  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "../oracle_floor"))
from panel import ROOT, dev2, forward  # noqa: E402

T, TMAX = 10, 0.5
N = int(float(sys.argv[1])) if len(sys.argv) > 1 else 200_000
PTS = {t["name"]: t["point"] for t in cont_d.TABLES}
V4T = {t["name"]: t["v4"] for t in cont_d.TABLES}

# node(v3) = the candidate's K=1 gauss_v4 node (weighted mean of phi Phi^6 on its own 40,000-point grid [v3-8, v3)),
# tabulated on v3 in [-9, 9] (step 1e-3) and interpolated, so that V1 is reproduced exactly.
_vt = np.arange(-9.0, 9.0005, 1e-3); _u = np.linspace(-8, 0, 40001)[:-1]
def _tab():
    out = np.empty(len(_vt))
    for i in range(0, len(_vt), 500):
        v = _vt[i:i + 500, None] + _u[None]; w = norm.pdf(v) * ndtr(v) ** 6
        out[i:i + 500] = (w * v).sum(1) / w.sum(1)
    return out
_nt = _tab()
def node(v3):
    return np.interp(v3, _vt, _nt)
mu = lambda v: -norm.pdf(v) / np.maximum(ndtr(v), 1e-300)
dmu = lambda v: -v * mu(v) + mu(v) ** 2


class Point:
    """Sampled inputs and their skeleton / closure positions."""
    def __init__(self, x):
        o = np.argsort(-x, 1); self.tri = o[:, :3]; n = len(x); self.n = n
        free = np.sort(o[:, 3:], 1); self.p4 = free[:, -1]; self.lows = free[:, :-1]
        v = np.take_along_axis(x, self.tri, 1); self.v2 = v[:, 1]; self.g12 = v[:, 0] - v[:, 1]; self.g23 = v[:, 1] - v[:, 2]
        self.b = self.v2 <= 0


def verdict(model, P, v1, v2, v3, v4, low):
    x = np.empty((P.n, T)); r = np.arange(P.n)
    x[r, P.tri[:, 0]] = v1; x[r, P.tri[:, 1]] = v2; x[r, P.tri[:, 2]] = v3; x[r, P.p4] = v4
    for j in range(6):
        x[r, P.lows[:, j]] = low
    return (forward(model, x).argmax(1) == P.tri[:, 1]).astype(float)


def anchor(P, transverse_on):
    g12s = np.where(P.b, PTS["b"][1], PTS["nb"][1]); g23s = np.where(P.b, PTS["b"][2], PTS["nb"][2])
    v2a = np.where(P.b, PTS["b"][0], PTS["nb"][0]) if transverse_on else P.v2
    v3a = v2a - g23s
    v4a = np.where(P.b, V4T["b"], V4T["nb"]) if transverse_on else node(v3a)    # candidate's own v4* at its anchors
    return v2a, g12s, g23s, v3a, v4a


def closure_low(v3, v3a, v4a, mulin_on):
    if mulin_on:                                  # declared: v4 = v3 - delta, lows linear in v4 around v4*
        v4 = v3 - (v3a - v4a); return v4, mu(v4a) + dmu(v4a) * (v4 - v4a)
    v4 = node(v3); return v4, mu(v4)              # exact closure c along the line


def variant(model, P, tmax_on=True, blind_on=True, mulin_on=True, product_on=True, transverse_on=True):
    v2a, g12s, g23s, v3a, v4a = anchor(P, transverse_on)
    v3 = v2a - P.g23; v4, low = closure_low(v3, v3a, v4a, mulin_on)
    s23 = verdict(model, P, v2a + g12s, v2a, v3, v4, low)                       # line 23 at t = g23
    s12 = verdict(model, P, v2a + P.g12, v2a, v3a, v4a, mu(v4a))                # line 12 at t = g12 (exact at v3*)
    cut23 = tmax_on & (P.g23 >= TMAX); cut12 = (tmax_on & (P.g12 >= TMAX)) | (blind_on & P.b)
    if product_on:
        return np.where(cut23, 1.0, s23) * np.where(cut12, 1.0, s12)
    J = verdict(model, P, v2a + P.g12, v2a, v3, v4, low)                        # joint at (v2a, g12, g23)
    return np.where(cut23 & cut12, 1.0, np.where(cut23, s12, np.where(cut12, s23, J)))


def closure_c(model, P):
    v3 = P.v2 - P.g23; v4 = node(v3)
    return verdict(model, P, P.v2 + P.g12, P.v2, v3, v4, mu(v4))


ONE = {"tmax": "tmax_on", "blind": "blind_on", "mulin": "mulin_on", "product": "product_on", "transverse": "transverse_on"}
ORDER = ["tmax", "blind", "mulin", "product", "transverse"]

if __name__ == "__main__":
    tr = json.load(open(os.path.join(ROOT, "data/oracle_floor/dev2_truths.json")))["truths"]
    path = os.path.join(HERE, "results.json"); res = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    for idx, (name, model) in enumerate(dev2()):
        if name in res:
            continue
        rng = np.random.default_rng(324000 + idx); x = rng.standard_normal((N, T)); P = Point(x)
        ok = (forward(model, x).argmax(1) == P.tri[:, 1]).astype(float)
        V = {"V1": variant(model, P)}
        for a in ORDER:
            V["off_" + a] = variant(model, P, **{ONE[a]: False})
        flags = {}
        for i, a in enumerate(ORDER):
            flags[ONE[a]] = False; V["path%d_%s" % (i + 1, a)] = variant(model, P, **flags)
        V["E1_direct"] = closure_c(model, P)
        e, _ = cont_d.est(model)
        r = dict(truth=tr[name]["acc"], est_d=e, mc_ok=float(ok.mean()))
        for k, v in V.items():
            d = v - ok
            r[k] = dict(mean=float(v.mean()), se=float(v.std() / np.sqrt(N)), paired_bias_pp=float(100 * d.mean()), paired_se_pp=float(100 * d.std() / np.sqrt(N)))
        r["identity_E1"] = float(np.abs(V["path5_transverse"] - V["E1_direct"]).max())
        res[name] = r; json.dump(res, open(path, "w"), indent=1)
        print(idx, name, "est %.4f V1mc %.4f (z %+.1f)  E1 %.4f  truth %.4f  id %.0f  %.0fs" % (
            e, r["V1"]["mean"], (r["V1"]["mean"] - e) / r["V1"]["se"], r["E1_direct"]["mean"], r["truth"], r["identity_E1"], time.time() - t0), flush=True)
