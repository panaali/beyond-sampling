"""Joint (g12 x g23) model-free strata + tie-swap antithetics.  `python joint.py design` builds joint_design.json:
  P_h: model-free stratum masses from 4e7 N(0,I) draws (no model calls);
  allocation: Neyman with cost from a pilot on the ORIGINAL model only (1e6 draws, plain and swap terms)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
ED = np.array([0, .0025, .005, .01, .02, .035, .05, .075, .1, .15, .2, .3, .5, np.inf])
NB = len(ED) - 1
K = 1 + NB * NB


def jstratum(x):
    xs = -np.sort(-x, 1)
    g12 = (xs[:, 0] - xs[:, 1]) / np.abs(xs[:, 0]); g23 = (xs[:, 1] - xs[:, 2]) / np.maximum(np.abs(xs[:, 1]), 1e-12)
    b12 = np.searchsorted(ED, g12, side="right") - 1; b23 = np.searchsorted(ED, g23, side="right") - 1
    return np.where(xs[:, 1] <= 0, 0, 1 + b12 * NB + b23)


def build():
    import estimators as E
    from common import Wi, Wh, Wo
    rng = np.random.default_rng(2024); cnt = np.zeros(K)
    for _ in range(40):
        cnt += np.bincount(jstratum(rng.standard_normal((1_000_000, 10))), minlength=K)
    P = cnt / cnt.sum()
    M = (Wi, Wh, Wo); x = np.random.default_rng(99).standard_normal((1_000_000, 10)); h = jstratum(x)
    g = E.g_of(M, x); sw = (g + E.g_of(M, E.swap_minpair(x)[0])) / 2
    n = np.bincount(h, minlength=K).astype(float)
    def sd(v):
        m1 = np.bincount(h, v, K) / np.maximum(n, 1); m2 = np.bincount(h, v * v, K) / np.maximum(n, 1)
        return np.sqrt(np.maximum(m2 - m1 ** 2, 0))
    sp, ss = sd(g), sd(sw)
    use = ss * np.sqrt(2) < sp
    sig = np.where(use, ss, sp) + 0.02          # floor: strata where the pilot saw no errors still get samples
    cost = np.where(use, 2.0, 1.0)
    a = P * sig / np.sqrt(cost); a = a / (a * cost).sum()
    json.dump(dict(P=P.tolist(), alloc=a.tolist(), use_swap=use.tolist(), cost=cost.tolist()),
              open(os.path.join(HERE, "joint_design.json"), "w"))
    print("strata", K, "nonzero", (P > 0).sum(), "swap strata", use.sum(), "max a/P", (a / np.maximum(P, 1e-12)).max())


if __name__ == "__main__" and sys.argv[1:] == ["design"]:
    build()


_D = None
GEN_COST = 200 / 6e3     # one candidate N(0,I_10) draw + sort + gap arithmetic, in forward-pass equivalents


def _design():
    global _D
    if _D is None:
        d = json.load(open(os.path.join(HERE, "joint_design.json")))
        _D = {k: np.array(v) for k, v in d.items()}
    return _D


def pooled_draw(n, rng, P, chunk=4096):
    need = n.copy(); got = [[] for _ in range(len(n))]; cand = 0
    while (need > 0).any():
        act = need > 0
        m = int(min(max(chunk, 1.05 * (need[act] / P[act]).max()), 2_000_000))
        x = rng.standard_normal((m, 10)); cand += m; h = jstratum(x)
        order = np.argsort(h, kind="stable"); hs = h[order]
        starts = np.searchsorted(hs, np.arange(len(n))); ends = np.searchsorted(hs, np.arange(len(n)), side="right")
        for k in np.nonzero(act)[0]:
            take = min(need[k], ends[k] - starts[k])
            if take:
                got[k].append(x[order[starts[k]:starts[k] + take]]); need[k] -= take
    return [np.concatenate(g) if g else np.zeros((0, 10)) for g in got], cand


def est_joint(model, budget, rng, count_gen=True):
    import estimators as E
    D = _design(); P = D["P"]; live = P > 0
    n = np.where(live, np.maximum(1, np.round(D["alloc"] * budget)), 0).astype(int)
    xs, cand = pooled_draw(n, rng, np.where(live, P, 1.0))
    X = np.concatenate([xs[k] for k in range(len(n)) if n[k]]); lab = np.concatenate([np.full(n[k], k) for k in range(len(n)) if n[k]])
    v = E.g_of(model, X); evals = len(X)
    sw = D["use_swap"][lab].astype(bool)
    if sw.any():
        v[sw] = (v[sw] + E.g_of(model, E.swap_minpair(X[sw])[0])) / 2; evals += int(sw.sum())
    mu = np.bincount(lab, v, len(n)) / np.maximum(n, 1)
    return float((P * mu).sum()), float(evals + (GEN_COST * cand if count_gen else 0))
