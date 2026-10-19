"""T428: is estT's 'exact' mode exact? Compare it with >= 1e8-sample MC on 20 (4,4) models, and locate any gap.
estT.py is imported UNCHANGED; the diagnostic variant only swaps its quadrature table (estT._RULES) at run time.

usage: OMP_NUM_THREADS=1 python t428.py scan            -> t428_scan.json  (exact, default rules, + 1e8 MC per model)
       OMP_NUM_THREADS=1 python t428.py rules NAME SEED  -> t428_rules.json (exact with default vs 512-node rules)
Models: zoo (4,4) seeds 0-4, ours 30sL2 0-4, 31s 0-4, 30s 42/20/40/10/44 (= every (4,4) model of T87/T111)."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "arc_m43_repro"))
sys.path.insert(0, os.path.join(HERE, "..", "understand_m43"))
import numpy as np
import estT
from m43 import Ops
from zoo44 import zoo, mc_acc
from models import load

MODELS = [("zoo", s) for s in range(5)] + [("30sL2", s) for s in range(5)] + [("31s", s) for s in range(5)] + \
         [("30s", s) for s in (42, 20, 40, 10, 44)]


def get(name, s):
    return zoo(4, 4, s) if name == "zoo" else load(4, 4, s, n_tag=name)


def exact(m):
    o = Ops(); t0 = time.time()
    p = estT.estT(m, o, mode="exact")
    return p, o.n, time.time() - t0


def scan(n_mc=100_000_000):
    fn = os.path.join(HERE, "t428_scan.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for name, s in MODELS:
        k = f"{name}_{s}"
        if k in res:
            continue
        m = get(name, s); p, ops, sec = exact(m)
        mc = mc_acc(m, n_mc, np.random.default_rng(428_000 + 100 * len(res) + s))
        z = (mc - p) / np.sqrt(p * (1 - p) / n_mc)
        res[k] = dict(exact=p, mc=mc, n_mc=n_mc, z=float(z), ops=ops, sec=sec)
        print(f"{k}: exact {p:.6f} MC1e8 {mc:.6f} diff {mc - p:+.2e} z {z:+.2f}", flush=True)
        json.dump(res, open(fn, "w"), indent=1)
    zs = np.array([r["z"] for r in res.values()])
    print(f"n={len(zs)} mean z {zs.mean():+.2f} rms z {np.sqrt((zs**2).mean()):.2f} max |z| {abs(zs).max():.2f}")


def rules(name, s, n=512):
    fn = os.path.join(HERE, "t428_rules.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    m = get(name, s)
    p0, ops0, _ = exact(m)
    saved = dict(estT._RULES)
    estT._RULES.update(wide=estT._rule(n, 3), thin=estT._rule(n, 3))
    try:
        p1, ops1, sec = exact(m)
    finally:
        estT._RULES.update(saved)
    res[f"{name}_{s}"] = dict(default=p0, fine=p1, nodes=n, shift=p1 - p0, sec_fine=sec)
    print(f"{name}_{s}: default {p0:.7f} {n}-node {p1:.7f} shift {p1 - p0:+.2e}", flush=True)
    json.dump(res, open(fn, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "scan":
        scan()
    else:
        rules(sys.argv[2], int(sys.argv[3]))
