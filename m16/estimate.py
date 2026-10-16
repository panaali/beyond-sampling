"""Attempt at a deductive accuracy estimate for M_16,10 (ARC challenge), from designed probes, not random inputs.

Error model: (a) blind region (global 2nd largest <= 0), probability 11/1024 analytic;
(b) 2nd-vs-3rd near-tie flips: for each configuration (p1,p2,p3) with x9 below both, probe the network at a
canonical input and bisect the relative gap g=(m2-m3)/m2 at which it flips to p3. Predicted flip probability =
P(relative gap < g*), using the model-free distribution of that gap.
"""
import numpy as np
from common import Wo, run

T = 10
_ref = np.sort(np.random.default_rng(0).standard_normal((400000, T)), 1)[:, ::-1]   # model-free order stats
EOS = _ref.mean(0)                                    # E[v1] >= E[v2] >= ...
REL_GAP = (_ref[:, 1] - _ref[:, 2]) / np.abs(_ref[:, 1])    # model-free distribution of (v2-v3)/|v2| (global ranks)


def canonical_input(p1, p2, p3, arrangement_seed):
    """x9 = 4th largest (below both), top 3 at p1,p2,p3, remaining ranks in a fixed shuffled order."""
    x = np.empty(T)
    x[p1], x[p2], x[p3], x[9] = EOS[0], EOS[1], EOS[2], EOS[3]
    rest = [i for i in range(9) if i not in (p1, p2, p3)]
    vals = EOS[4:].copy(); np.random.default_rng(arrangement_seed).shuffle(vals)
    x[rest] = vals
    return x


def flip_threshold(p1, p2, p3, seed, iters=14):
    """Largest relative gap g at which the network picks p3 when m3 = m2*(1-g)."""
    base = canonical_input(p1, p2, p3, seed)
    m2 = base[p2]
    def picks3(g):
        x = base.copy(); x[p3] = m2 * (1 - g)
        return (run(x[None])[:, -1] @ Wo.T).argmax(1)[0] == p3
    if not picks3(1e-6):
        return 0.0, 1
    lo, hi, n = 1e-6, 1.0, 1
    if picks3(hi):
        return hi, 2
    for _ in range(iters):
        mid = (lo + hi) / 2; n += 1
        lo, hi = (mid, hi) if picks3(mid) else (lo, mid)
    return lo, n + 1


def threshold_general(order, seed, iters=12, eos=None):
    """order: positions holding global ranks 1,2,3 (len 3). Answer = order[1], competitor = order[2].
    Remaining ranks get shuffled canonical values. Bisect relative gap (v2-v3)/v2 below which the net picks order[2]."""
    eos = EOS if eos is None else eos
    x = np.empty(T)
    for r, p in enumerate(order):
        x[p] = eos[r]
    rest = [i for i in range(T) if i not in order]
    ranks = np.arange(3, T); np.random.default_rng(seed).shuffle(ranks); x[rest] = eos[ranks]
    a, b = order[1], order[2]; v2 = x[a]
    def picks_b(g):
        y = x.copy(); y[b] = v2 * (1 - g)
        return (run(y[None])[:, -1] @ Wo.T).argmax(1)[0] == b
    n = 1
    if not picks_b(1e-6):
        return 0.0, n
    lo, hi = 1e-6, 1.0
    n += 1
    if picks_b(hi):
        return hi, n
    for _ in range(iters):
        mid = (lo + hi) / 2; n += 1
        lo, hi = (mid, hi) if picks_b(mid) else (lo, mid)
    return lo, n


def estimate_accuracy(seeds=(0,), rng_order=None):
    """Deductive-style estimate: enumerates every (rank1, rank2, rank3) position triple over all 10 positions."""
    evals, flip_p = 0, []
    for p1 in range(T):
        for p2 in range(T):
            for p3 in range(T):
                if len({p1, p2, p3}) < 3:
                    continue
                g = []
                for s in seeds:
                    th, n = threshold_general((p1, p2, p3), s); g.append(th); evals += n
                flip_p.append((REL_GAP < np.mean(g)).mean())
    p_blind = 11 / 1024
    e_blind = 8 / 9                       # argued: with <=1 positive input the net cannot rank the rest
    p_flip = float(np.mean(flip_p))       # all 720 triples equally likely under i.i.d. inputs
    err = p_blind * e_blind + (1 - p_blind) * p_flip
    return 1 - err, evals, dict(p_flip=p_flip, blind=p_blind * e_blind)


REL_GAP12 = (_ref[:, 0] - _ref[:, 1]) / np.abs(_ref[:, 0])   # model-free distribution of (v1-v2)/|v1|


def threshold_top(order, seed, iters=12, eos=None):
    """Bisect the relative gap (v1-v2)/v1 below which the net picks the max (order[0]) instead of order[1]."""
    eos = EOS if eos is None else eos
    x = np.empty(T)
    for r, p in enumerate(order):
        x[p] = eos[r]
    rest = [i for i in range(T) if i not in order]
    ranks = np.arange(3, T); np.random.default_rng(seed).shuffle(ranks); x[rest] = eos[ranks]
    a, b = order[1], order[0]; v2 = x[a]
    def picks_max(g):
        y = x.copy(); y[b] = v2 / (1 - g)      # (v1-v2)/v1 = g
        return (run(y[None])[:, -1] @ Wo.T).argmax(1)[0] == b
    n = 1
    if not picks_max(1e-6):
        return 0.0, n
    lo, hi = 1e-6, 0.95
    n += 1
    if picks_max(hi):
        return hi, n
    for _ in range(iters):
        mid = (lo + hi) / 2; n += 1
        lo, hi = (mid, hi) if picks_max(mid) else (lo, mid)
    return lo, n


def estimate_accuracy_v2(seeds=(0,)):
    """Adds 1st-vs-2nd near-tie flips to estimate_accuracy (treated as disjoint from 2nd-vs-3rd flips)."""
    acc1, evals, parts = estimate_accuracy(seeds)
    fp = []
    for p1 in range(T):
        for p2 in range(T):
            for p3 in range(T):
                if len({p1, p2, p3}) < 3:
                    continue
                g = []
                for s in seeds:
                    th, n = threshold_top((p1, p2, p3), s); g.append(th); evals += n
                fp.append((REL_GAP12 < np.mean(g)).mean())
    p_top = float(np.mean(fp))
    err = parts["blind"] + (1 - 11 / 1024) * (parts["p_flip"] + p_top)
    return 1 - err, evals, dict(parts, p_top=p_top)


_SD = _ref.std(0)    # model-free s.d. of each order statistic


def threshold_with_spread(order, seed, kind, eps=0.05):
    """Threshold at the canonical point plus its s.d. from first-order sensitivity to the other inputs."""
    fn = threshold_general if kind == "23" else threshold_top
    g0, n = fn(order, seed)
    if g0 <= 0 or g0 >= 0.95:
        return g0, 0.0, n
    var = 0.0
    for rk in range(3, T):                    # perturb each non-top-3 order statistic by eps s.d.
        eos = EOS.copy(); eos[rk] += eps * _SD[rk]
        gp, n1 = fn(order, seed, eos=eos); n += n1
        var += ((gp - g0) / eps) ** 2         # (d g / d sd-unit)^2, unit variance per input
    return g0, np.sqrt(var), n


def estimate_accuracy_v3(seed=0, z=np.linspace(-3, 3, 13)):
    w = np.exp(-z ** 2 / 2); w /= w.sum()
    out, evals = {}, 0
    for kind, gaps in (("23", REL_GAP), ("12", REL_GAP12)):
        ps = []
        for p1 in range(T):
            for p2 in range(T):
                for p3 in range(T):
                    if len({p1, p2, p3}) < 3:
                        continue
                    g0, sd, n = threshold_with_spread((p1, p2, p3), seed, kind); evals += n
                    sg = np.sort(gaps)
                    cdf = np.searchsorted(sg, np.clip(g0 + sd * z, 0, None)) / len(sg)
                    ps.append(float((w * cdf).sum()))
        out[kind] = float(np.mean(ps))
    err = (11 / 1024) * (8 / 9) + (1 - 11 / 1024) * (out["23"] + out["12"])
    return 1 - err, evals, out
