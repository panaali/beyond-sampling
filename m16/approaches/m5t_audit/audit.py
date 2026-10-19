"""T417: end-to-end audit of m5t_fast.bounds_batch against the independent arb reference (ref.py).
usage: python audit.py FAMILY N SEED [TAG]   -> results/audit_<family>_<seed><tag>.json (+ worst cases)
FAMILY is one of gen.FAMILIES or 'mix' (round robin). Pre-registered criteria: README.md."""
import json
import os
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "event_cone_rigour"))
sys.path.insert(0, HERE)
import m5t_fast as F  # noqa: E402
from flint import arb, fmpq  # noqa: E402

import gen  # noqa: E402
import ref  # noqa: E402

TOL = fmpq(1, 10 ** 10)          # shortfall tolerance (relative), pre-registered


def fq(x):
    return fmpq(*float(x).as_integer_ratio())


def run_case(c):
    P = F.Params(c["d"], c["mlow"], c["jp3"], c["t0"], c["zbins"], c["zratio"])
    lo, hi = F.bounds_batch(c["Mlo"][None], c["Mhi"][None], np.array([c["vlo"]]), np.array([c["vhi"]]), P, mc=c["mc"][None])
    lo, hi = float(lo[0]), float(hi[0])
    if not np.all(np.isfinite(c["mc"])):                 # degenerate tangent centre: only check the output is sane
        return dict(lo=lo, hi=hi, nan=bool(np.isnan(lo) or np.isnan(hi)), bad_mc=True, hi_inf=bool(np.isinf(hi)))
    Mq = [[fq(x) for x in row] for row in c["M"]]
    HIl, HIh, LOl, LOh, nex = ref.ref_bounds(Mq, c["vol"], c["mc"], c["d"], c["mlow"], c["jp3"], P.edges)
    r = dict(lo=lo, hi=hi, HIl=float(HIl.mid()) if HIl.is_finite() else float("inf"), LOh=float(LOh.mid()) if LOh.is_finite() else float("inf"),
             LOl=float(LOl.mid()), nex=nex)
    fin_hi = not np.isnan(hi); fin_lo = np.isfinite(lo)
    r["nan"] = (not fin_hi) or (not fin_lo)
    if r["nan"]:
        return r
    ahi = arb(fq(hi)) if np.isfinite(hi) else None
    alo = arb(fq(lo))
    r["viol_hi"] = bool(ahi is not None and LOl.is_finite() and ahi < LOl)          # fast hi < reference lo
    r["viol_lo"] = bool(HIh.is_finite() and alo > HIh)                            # fast lo > reference hi
    r["short_hi"] = bool(ahi is not None and HIl.is_finite() and ahi < HIl * (1 - TOL))
    r["short_lo"] = bool(LOh.is_finite() and alo > LOh * (1 + TOL))
    r["hi_inf"] = not np.isfinite(hi)
    # tightness: fast hi / exact formula hi, fast lo / exact formula lo
    r["rh"] = float((ahi / HIl).mid()) if (ahi is not None and HIl.is_finite() and HIl > 0) else None
    r["rl"] = float((alo / LOh).mid()) if (LOh.is_finite() and LOh > 0) else None
    return r


def main():
    fam, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); tag = sys.argv[4] if len(sys.argv) > 4 else ""
    rng = np.random.default_rng(417_000 + seed)
    seeds = gen.Seeds()
    fams = gen.FAMILIES if fam == "mix" else [fam]
    stats = {f: dict(cases=0, failed_construct=0, exceptions=0, nan=0, viol_hi=0, viol_lo=0, short_hi=0, short_lo=0,
                     hi_inf=0, bad_mc=0, min_rh=None, max_rl=None, rh=[], t_ref=0.0, t_fast=0.0) for f in fams}
    worst = []
    t_start = time.time(); last = t_start
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"audit_{fam}_{seed}{tag}.json")

    def dump():
        js = {}
        for f, s in stats.items():
            s2 = {k: v for k, v in s.items() if k != "rh"}
            if s["rh"]:
                q = np.quantile(np.array(s["rh"]), [0, 0.001, 0.5, 0.999, 1])
                s2["rh_quantiles_0_.001_.5_.999_1"] = [float(v) for v in q]
            js[f] = s2
        json.dump(dict(family=fam, n=n, seed=seed, elapsed=time.time() - t_start, stats=js, worst=worst[:200]),
                  open(out, "w"), indent=0)
    i = 0
    while sum(s["cases"] for s in stats.values()) < n:
        f = fams[i % len(fams)]; i += 1
        s = stats[f]
        try:
            c = gen.make(f, rng, seeds)
        except Exception:  # noqa: BLE001
            c = None
        if c is None:
            s["failed_construct"] += 1
            continue
        s["cases"] += 1
        try:
            r = run_case(c)
        except Exception as e:  # noqa: BLE001
            s["exceptions"] += 1
            if s["exceptions"] <= 5:
                worst.append(dict(kind="exception", family=f, err=repr(e), tb=traceback.format_exc()[-800:],
                                  case={k: (v.tolist() if isinstance(v, np.ndarray) else str(v)) for k, v in c.items()}))
            continue
        bad = False
        for k in ("nan", "viol_hi", "viol_lo", "short_hi", "short_lo", "hi_inf", "bad_mc"):
            if r.get(k):
                s[k] += 1
                bad = bad or k not in ("hi_inf", "bad_mc")
        if r.get("rh") is not None:
            s["rh"].append(r["rh"])
            s["min_rh"] = r["rh"] if s["min_rh"] is None else min(s["min_rh"], r["rh"])
        if r.get("rl") is not None:
            s["max_rl"] = r["rl"] if s["max_rl"] is None else max(s["max_rl"], r["rl"])
        if bad and len(worst) < 200:
            worst.append(dict(kind="flag", family=f, res=r,
                              case={k: (v.tolist() if isinstance(v, np.ndarray) else str(v)) for k, v in c.items()}))
        if time.time() - last > 300:
            dump(); last = time.time()
    dump()
    print(json.dumps({f: {k: v for k, v in s.items() if k != "rh"} for f, s in stats.items()}))


if __name__ == "__main__":
    main()
