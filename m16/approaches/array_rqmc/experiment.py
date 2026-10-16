"""RMSE vs n for Array-RQMC and baselines, over many independent randomisations.

usage: python experiment.py <set> <n_list> <reps> <methods>
  set: orig | s0025 | s005   (orig: truth 0.95317945 from 20M; noisy: noisy_models(6, sigma), truth from 4M)
Writes results/<set>_<methods>_<nlist>.json with every estimate (so variance and MSE can be recomputed).
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../..")); sys.path.insert(0, HERE)
from common import Wi, Wh, Wo                                     # noqa: E402
from arc_eval import noisy_models, forward                        # noqa: E402
import stratified as st                                           # noqa: E402
from core import (est_mc, est_rqmc, est_array, sort_m1, sort_gap, sort_tie, ValueFn, init, step)  # noqa: E402

ORIG = (Wi, Wh, Wo)
_V = {}


def V():
    if "V" not in _V:
        cache = os.path.join(HERE, "results", "V.npz")
        if os.path.exists(cache):
            z = np.load(cache); _V["V"] = ValueFn({int(k[1:]): z[k] for k in z.files})
        else:
            _V["V"] = ValueFn.fit(ORIG)       # pilot on the ORIGINAL model only
            np.savez(cache, **{f"t{k}": v for k, v in _V["V"].coefs.items()})
    return _V["V"]


def sort_m2(S):
    return np.where(np.isfinite(S["m2"]), S["m2"], S["m1"] - 3)


DESIGN = json.load(open(os.path.join(HERE, "../../stratified_design.json")))


def est_strat(m, n, rng):
    return st.estimate(lambda x: forward(m, x), n, rng, np.array(DESIGN["P"]), np.array(DESIGN["alloc"]))[0]


METHODS = {
    "mc": lambda m, n, r: est_mc(m, n, r, cmc=False),
    "strat": est_strat,
    "mc+cmc": lambda m, n, r: est_mc(m, n, r),
    "rqmc+cmc": est_rqmc,
    "arr_m1": lambda m, n, r: est_array(m, n, r, sort_m1),
    "arr_gap": lambda m, n, r: est_array(m, n, r, sort_gap),
    "arr_V": lambda m, n, r: est_array(m, n, r, V()),
    "arr2_m1m2": lambda m, n, r: est_array(m, n, r, sort_m2, sort2=sort_m1),
    "arr_tie": lambda m, n, r: est_array(m, n, r, sort_tie),
    "arr2_m2tie": lambda m, n, r: est_array(m, n, r, sort_tie, sort2=sort_m2),
    "arr2_m1V": lambda m, n, r: est_array(m, n, r, V(), sort2=sort_m1),
}


def models(which):
    if which == "orig":
        return [ORIG], [0.95317945]
    sigma = {"s0025": 0.0025, "s005": 0.005}[which]
    ms = noisy_models(6, sigma)
    cache = os.path.join(HERE, "results", f"truth_{which}.json")
    if os.path.exists(cache):
        tr = json.load(open(cache))
    else:
        tr = [true_acc4m(m) for m in ms]; json.dump(tr, open(cache, "w"))
    return ms, tr


def true_acc4m(m, n=4_000_000, seed=12345, chunk=250_000):
    rng = np.random.default_rng(seed); ok = 0
    for _ in range(n // chunk):
        x = rng.standard_normal((chunk, 10))
        ok += (forward(m, x).argmax(1) == np.argsort(x, 1)[:, -2]).sum()
    return ok / n


if __name__ == "__main__":
    which, ns, reps, meth = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], int(sys.argv[3]), sys.argv[4].split(",")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    ms, tr = models(which)
    out = {"set": which, "truth": tr, "reps": reps, "res": {}}
    for name in meth:
        for n in ns:
            t0 = time.time(); E = []
            for mi, m in enumerate(ms):
                E.append([float(METHODS[name](m, n, np.random.default_rng([mi, r, n, 77]))) for r in range(reps)])
            E = np.array(E); tr_ = np.array(tr)[:, None]
            var = E.var(1, ddof=1).mean(); mse = ((E - tr_) ** 2).mean()
            out["res"][f"{name}|{n}"] = E.tolist()
            print(f"{which} {name:10s} n={n:6d} n*var={n*var:.5f} n*mse={n*mse:.5f} bias={(E-tr_).mean():+.1e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    fn = f"{which}_{'-'.join(meth)}_{'-'.join(map(str, ns))}.json"
    json.dump(out, open(os.path.join(HERE, "results", fn), "w"))
