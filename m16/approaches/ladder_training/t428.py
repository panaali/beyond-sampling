"""T428: is estT's 'exact' mode exact? Compare it with >= 1e8-sample MC on 20 (4,4) models, and locate any gap.
estT.py is imported UNCHANGED; the diagnostic variant only swaps its quadrature table (estT._RULES) at run time.

usage: OMP_NUM_THREADS=1 python t428.py scan            -> t428_scan.json  (exact, default rules, + 1e8 MC per model)
       OMP_NUM_THREADS=1 python t428.py rules NAME SEED [N=512] -> t428_rules.json (exact, default vs N-node rules)
       OMP_NUM_THREADS=1 python t428.py fine             -> t428_fine.json (all scanned models with 128-node rules)
       OMP_NUM_THREADS=1 python t428.py which NAME SEED  -> t428_which.json (refine only 'wide' or only 'thin')
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
    res[f"{name}_{s}_{n}"] = dict(default=p0, fine=p1, nodes=n, shift=p1 - p0, sec_fine=sec)
    print(f"{name}_{s}: default {p0:.7f} {n}-node {p1:.7f} shift {p1 - p0:+.2e}", flush=True)
    json.dump(res, open(fn, "w"), indent=1)


def which(name, s, n=128):
    """Refine only one of the two default rules: which one carries the error?"""
    fn = os.path.join(HERE, "t428_which.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    m = get(name, s); saved = dict(estT._RULES); out = {}
    for part in ("wide", "thin"):
        estT._RULES[part] = estT._rule(n, 3)
        try:
            out[part] = exact(m)[0]
        finally:
            estT._RULES.update(saved)
    res[f"{name}_{s}"] = out
    print(f"{name}_{s}: refine wide only {out['wide']:.7f}, refine thin only {out['thin']:.7f}", flush=True)
    json.dump(res, open(fn, "w"), indent=1)


def fine(n=128):
    """Re-run every scanned model with an n-node rule for both tables; z against the SAME 1e8 MC values."""
    scan_res = json.load(open(os.path.join(HERE, "t428_scan.json")))
    fn = os.path.join(HERE, "t428_fine.json")
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    saved = dict(estT._RULES)
    for name, s in MODELS:
        k = f"{name}_{s}"
        if k in res or k not in scan_res:
            continue
        estT._RULES.update(wide=estT._rule(n, 3), thin=estT._rule(n, 3))
        try:
            p, ops, sec = exact(get(name, s))
        finally:
            estT._RULES.update(saved)
        r = scan_res[k]; z = (r["mc"] - p) / np.sqrt(p * (1 - p) / r["n_mc"])
        res[k] = dict(fine=p, default=r["exact"], shift=p - r["exact"], z_fine=float(z), z_default=r["z"], ops=ops, sec=sec)
        print(f"{k}: fine {p:.6f} shift {p - r['exact']:+.2e} z default {r['z']:+.2f} -> fine {z:+.2f}", flush=True)
        json.dump(res, open(fn, "w"), indent=1)
    zf = np.array([r["z_fine"] for r in res.values()]); zd = np.array([r["z_default"] for r in res.values()])
    print(f"n={len(zf)} rms z default {np.sqrt((zd**2).mean()):.2f} -> fine {np.sqrt((zf**2).mean()):.2f}; "
          f"mean z fine {zf.mean():+.2f}; max |z| fine {abs(zf).max():.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "scan":
        scan()
    elif sys.argv[1] == "fine":
        fine()
    elif sys.argv[1] == "which":
        which(sys.argv[2], int(sys.argv[3]))
    else:
        rules(sys.argv[2], int(sys.argv[3]), *map(int, sys.argv[4:5]))
