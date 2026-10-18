"""ARC's evaluation protocol for accuracy estimators on M_16,10 (AlgZoo blog, 'Challenge' + footnote 4).

Make noisy copies of the model (noise on all weights, enough to change accuracy substantially), get ground
truth for each copy from many samples, and compare estimators by mean squared error vs compute
(compute = model forward evaluations of one length-10 sequence).

Estimators are functions est(model, budget, rng) -> (estimate, evals_used). Any estimator, including one
supplied by someone else, can be plugged in here and compared on equal terms.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "m16")
from common import Wi as Wi0, Wh as Wh0, Wo as Wo0  # noqa: E402

T = 10


def forward(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), Wh.shape[0]))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return h @ Wo.T


def true_acc(model, n=2_000_000, seed=12345, chunk=500_000):
    rng = np.random.default_rng(seed); ok = 0
    chunk = min(chunk, n); assert n % chunk == 0
    for _ in range(n // chunk):
        x = rng.standard_normal((chunk, T))
        ok += (forward(model, x).argmax(1) == np.argsort(x, 1)[:, -2]).sum()
    return ok / n


def noisy_models(k, sigma, seed=0):
    """Gaussian noise with s.d. sigma * RMS of each weight matrix."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(k):
        out.append(tuple(W + sigma * np.sqrt((W ** 2).mean()) * rng.standard_normal(W.shape)
                         for W in (Wi0, Wh0, Wo0)))
    return out


# ---------------------------------------------------------------- estimators
def est_sampling(model, budget, rng):
    x = rng.standard_normal((budget, T))
    return float((forward(model, x).argmax(1) == np.argsort(x, 1)[:, -2]).mean()), budget


_ref = np.sort(np.random.default_rng(0).standard_normal((400_000, T)), 1)[:, ::-1]
EOS = _ref.mean(0)
REL_GAP23 = np.sort((_ref[:, 1] - _ref[:, 2]) / np.abs(_ref[:, 1]))
REL_GAP12 = np.sort((_ref[:, 0] - _ref[:, 1]) / np.abs(_ref[:, 0]))


def _canon(order, seed):
    x = np.empty(T)
    for r, p in enumerate(order):
        x[p] = EOS[r]
    rest = [i for i in range(T) if i not in order]
    ranks = np.arange(3, T); np.random.default_rng(seed).shuffle(ranks); x[rest] = EOS[ranks]
    return x


def _bisect(model, base, move, wrong, iters=12):
    """Largest g in [0, .95] at which move(base, g) makes the model output `wrong`."""
    def bad(g):
        return forward(model, move(base, g)[None]).argmax(1)[0] == wrong
    if not bad(1e-6):
        return 0.0, 1
    if bad(0.95):
        return 0.95, 2
    lo, hi, n = 1e-6, 0.95, 2
    for _ in range(iters):
        mid = (lo + hi) / 2; n += 1
        lo, hi = (mid, hi) if bad(mid) else (lo, mid)
    return lo, n


def est_probe(model, budget=None, rng=None, seed=0):
    """Deductive probe estimator (FINDINGS section 5, v2): blind region + 2nd/3rd and 1st/2nd tie thresholds.
    Uses a fixed set of designed probes; `budget` is ignored."""
    evals, p23, p12 = 0, [], []
    for p1 in range(T):
        for p2 in range(T):
            for p3 in range(T):
                if len({p1, p2, p3}) < 3:
                    continue
                base = _canon((p1, p2, p3), seed)
                v2 = base[p2]
                g, n = _bisect(model, base, lambda b, g: _set(b, p3, v2 * (1 - g)), p3); evals += n
                p23.append(np.searchsorted(REL_GAP23, g) / len(REL_GAP23))
                g, n = _bisect(model, base, lambda b, g: _set(b, p1, v2 / (1 - g)), p1); evals += n
                p12.append(np.searchsorted(REL_GAP12, g) / len(REL_GAP12))
    # blind region: probe the model on canonical blind inputs instead of assuming 8/9
    blind_err, nb = _blind_probe(model); evals += nb
    err = (11 / 1024) * blind_err + (1 - 11 / 1024) * (np.mean(p23) + np.mean(p12))
    return 1 - err, evals


def _set(b, i, v):
    y = b.copy(); y[i] = v; return y


def _blind_probe(model):
    """Canonical inputs for 'at most one positive': every assignment of order statistics conditioned on <=1
    positive, one per position of the largest (deterministic)."""
    neg = -np.abs(_ref[(_ref > 0).sum(1) <= 1])        # negatives-only part
    xs = []
    for p in range(T):
        x = np.sort(_ref[(_ref > 0).sum(1) == 1], 1)[:, ::-1].mean(0).copy()   # mean order stats given 1 positive
        order = [p] + [i for i in range(T) if i != p]
        y = np.empty(T); y[order] = x; xs.append(y)
    xs = np.array(xs)
    ok = forward(model, xs).argmax(1) == np.argsort(xs, 1)[:, -2]
    return 1 - ok.mean(), len(xs)


def evaluate(estimators, k=12, sigma=0.05, budgets=(1000, 4000, 16000, 64000), reps=4, seed=0):
    models = noisy_models(k, sigma, seed)
    truth = [true_acc(m) for m in models]
    res = {"sigma": sigma, "truth": truth}
    for name, est in estimators.items():
        rows = []
        for b in budgets:
            se, used = [], []
            for mi, m in enumerate(models):
                for r in range(reps):
                    e, u = est(m, b, np.random.default_rng(1000 * mi + r))
                    se.append((e - truth[mi]) ** 2); used.append(u)
            rows.append({"budget": b, "evals": float(np.mean(used)), "mse": float(np.mean(se))})
            if est is est_probe:      # deterministic, budget-independent
                break
        res[name] = rows
    return res


if __name__ == "__main__":
    sigma = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    out = evaluate({"sampling": est_sampling, "probe": est_probe}, sigma=sigma)
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"results/arc_eval_sigma{sigma}.json", "w"), indent=1)
