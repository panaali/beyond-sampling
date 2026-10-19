"""T449 shared loader: the frozen T10 recipe (hashes asserted), its node sets, models, and the per-stratum integrand.

g_h(U) = wt * F_hom for U in [0,1]^T, so that est = 1 - sum_h M_h mean_{U in P_h} g_h(U) (frozen/estimator.py).
"""
import hashlib
import json
import os
import sys

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
FZ = os.path.join(HERE, "frozen")
ROOT = os.path.join(HERE, "..", "..", "..")
DEV3 = os.path.join(ROOT, "data", "dev3")
SHA = {"estimator.py": "fb7902e12240", "walk.py": "a38258f586b4", "hom.py": "a91f019fca21",
       "core.py": "48dbba49ab13", "walk_nb.py": "bbc6cc992c7c"}       # as asserted by T429 score.py @ 79ecae63
DATA_SHA = {"weights.npz": "62c952037e8c", "truths.json": "abcb7d6d264a"}


def _h(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]


for _f, _s in SHA.items():
    assert _h(os.path.join(FZ, _f)) == _s, _f


class LazyRule:
    """rules[lo:hi] of qmc.Sobol(nd, scramble=True, seed).random_base2(m)[:n] (T429's gated execution-only swap)."""

    def __init__(self, nd, n, seed):
        self.nd, self.n, self.seed, self.eng, self.pos = nd, n, seed, None, 0

    def __getitem__(self, s):
        lo, hi = s.start or 0, min(s.stop if s.stop is not None else self.n, self.n)
        if hi <= lo:
            return np.zeros((0, self.nd))
        if self.eng is None or self.pos != lo:
            self.eng = qmc.Sobol(self.nd, scramble=True, seed=self.seed)
            if lo:
                self.eng.fast_forward(lo)
        self.pos = hi
        return self.eng.random(hi - lo)


sys.path.insert(0, FZ)
import core as C  # noqa: E402
import walk as W  # noqa: E402
import walk_nb as WN  # noqa: E402
import estimator as E  # noqa: E402

C.sobol = LazyRule
WN.MODE["fast"] = False; WN.MODE["fm"] = False
W.walk = WN.walk
T = 10
STRATA = E.strata(T)
MASS = [E.stratum_mass(T, k, h) for k, h in STRATA]


def node(j, n, lo=0, rule_seed=E.RULE_SEED):
    """The first nodes of stratum j's frozen rule (the estimator's P_j prefix)."""
    return LazyRule(T, 1 << 40, rule_seed + 101 * j)[lo:lo + n]


def g(model, j, U):
    """Integrand of stratum j at U (n, T); returns (values, cost in FPE)."""
    kind, h = STRATA[j]
    return E.run_points(model, T, kind, h, np.atleast_2d(U))


def dev3_models(k=None):
    for f, s in DATA_SHA.items():
        assert _h(os.path.join(DEV3, f)) == s, f
    Wt = np.load(os.path.join(DEV3, "weights.npz"))
    tr = json.load(open(os.path.join(DEV3, "truths.json")))
    names = [str(n) for n in Wt["names"]][:k]
    out = {n: (Wt["Wi"][i].astype(np.float64), Wt["Wh"][i].astype(np.float64), Wt["Wo"][i].astype(np.float64))
           for i, n in enumerate(names)}
    return out, tr


def zoo_models(seeds=range(5)):
    from alg_zoo import zoo_2nd_argmax
    out = {}
    for s in seeds:
        m = zoo_2nd_argmax(16, 10, seed=s)
        f = lambda t: t.detach().numpy().astype(np.float64)
        out[f"seed{s}"] = (f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight))
    return out


def models(n_dev=5):
    d, tr = dev3_models(n_dev)
    out = zoo_models(); out.update(d)
    return out, tr
