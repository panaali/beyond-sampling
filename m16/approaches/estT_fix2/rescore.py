"""T443: re-score results that used the OLD estT d=4 orthant (understand_m43/estT.py, 12/48-node rules) with the
fixed engine estT_fix/estT.py (T131/E46). Recipe logic is unchanged; only the cone masses are corrected.

Post-hoc variant B (not pre-registered): append "r24" -> old engine with the wide rule graded p=3, 24 nodes
(rulecost.py: within 3e-6 of estT_fix at +3-10% ops); writes rescore_<part>_r24.json. Truth stays estT_fix exact
(read from rescore_<part>.json).
usage: OMP_NUM_THREADS=1 python rescore.py e41      # zoo (4,4) seeds 0-4, clean + 8 noisy each (T87 / E41)
       OMP_NUM_THREADS=1 python rescore.py t111     # 15 retrained (4,4) models of T111, clean + 8 noisy each
       OMP_NUM_THREADS=1 python rescore.py f12      # E41 exhaustion checklist F1 (eps .01) and F2 (inter exact), seed 0
Writes rescore_<part>.json. Noisy truths are the SAVED 1e7-MC truths of the original runs (same copies: noise seed 50).
Ratios as run87: clean err^2 / (p(1-p) / (ops/96)); noisy (mean err^2 - pq/1e7) / (pq / (mean ops / 96))."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "..")
sys.path[:0] = [os.path.join(A, "estT_fix"), os.path.join(A, "arc_m43_repro"), os.path.join(A, "understand_m43")]
import numpy as np
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "fix"
if VARIANT == "r24":   # post-hoc variant B: OLD engine, wide rule graded p = 3 with 24 nodes (thin unchanged 48/p3)
    sys.path.insert(0, os.path.join(A, "understand_m43"))
import estT
if VARIANT == "r24":
    assert os.path.samefile(os.path.dirname(estT.__file__), os.path.join(A, "understand_m43")), estT.__file__
    estT._RULES["wide"] = estT._rule(24, 3)
else:
    assert os.path.samefile(os.path.dirname(estT.__file__), os.path.join(A, "estT_fix")), estT.__file__
from m43 import Ops
from zoo44 import zoo
from zoo43 import noisy_copies

OPS_S = 96
INP = os.path.join(HERE, "inputs")


def run(m, mode="recipe", **kw):
    estT.STATS.update(cells=0, cut=0, resolved=0, simplices=0); o = Ops(); t0 = time.time()
    p = estT.estT(m, o, mode=mode, **kw)
    return dict(p=p, ops=o.n, cells=estT.STATS["cells"], sec=time.time() - t0)


TRUTH = [None]


def score(m, old_clean, old_noisy, key=None):
    ex = run(m, "exact") if TRUTH[0] is None else dict(p=TRUTH[0][key]["exact_fix"], ops=TRUTH[0][key]["exact_ops_fix"])
    rc = run(m)
    p = ex["p"]; e = rc["p"] - p
    out = dict(exact_fix=p, exact_old=old_clean["exact"], recipe_fix=rc["p"], recipe_old=old_clean["est"],
               ops_fix=rc["ops"], ops_old=old_clean["ops"], exact_ops_fix=ex["ops"],
               err_fix=e, ratio_fix=e ** 2 / (p * (1 - p) / (rc["ops"] / OPS_S)),
               ratio_truthonly=(old_clean["est"] - p) ** 2 / (p * (1 - p) / (old_clean["ops"] / OPS_S)),
               ratio_old=old_clean["ratio"])
    rows = old_noisy["rows"]; ests, opss = [], []
    for m2, r in zip(noisy_copies(m, len(rows), 0.05, 50), rows):
        q = run(m2); ests.append(q["p"]); opss.append(q["ops"])
    tr = np.array([r["truth"] for r in rows]); e2 = np.array(ests) - tr; pq = np.mean(tr * (1 - tr)); om = np.mean(opss)
    out.update(noisy_ratio_fix=float(max(np.mean(e2 ** 2) - pq / 1e7, 0) / (pq / (om / OPS_S))),
               noisy_rms_fix=float(np.sqrt(np.mean(e2 ** 2))), noisy_ratio_old=old_noisy["ratio"], noisy_ops_fix=float(om))
    return out


def main(part):
    fn = os.path.join(HERE, f"rescore_{part}.json" if VARIANT == "fix" else f"rescore_{part}_{VARIANT}.json")
    TRUTH[0] = None if VARIANT == "fix" else json.load(open(os.path.join(HERE, f"rescore_{part}.json")))
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    if part == "e41":
        old = json.load(open(os.path.join(A, "understand_m43", "run87.json")))
        jobs = [(f"zoo_{s}", lambda s=s: zoo(4, 4, s), old["clean"][str(s)], old["noisy"][str(s)]) for s in range(5)]
    elif part == "t111":
        jobs = []
        for tag in ("30sL2", "31s", "30s"):
            old = json.load(open(os.path.join(INP, f"run111_{tag}.json")))
            for s in old["clean"]:
                def ld(tag=tag, s=s):
                    d = np.load(os.path.join(INP, f"m_4_4_{tag}_{s}.npz")); return d["Wi"], d["Wh"], d["Wo"]
                jobs.append((f"{tag}_{s}", ld, old["clean"][s], old["noisy"][s]))
    elif part == "f12":
        m = zoo(4, 4, 0); ex = run(m, "exact")["p"]; orig = estT._split_impact
        res["exact_fix"] = ex
        for label, kw, inter in (("F1 eps=0.01", dict(eps=0.01), False), ("F2 intermediate exact, last margin eps=0.1", dict(eps=0.1), True)):
            if inter:
                estT._split_impact = lambda cells, A_, w, eps, ops, exact: orig(cells, A_, w, eps, ops, True)
            try:
                r = run(m, **kw)
            finally:
                estT._split_impact = orig
            e = r["p"] - ex; r.update(err=e, ratio=e ** 2 / (ex * (1 - ex) / (r["ops"] / OPS_S)))
            res[label] = r; print(label, f"err {100*e:+.4f}pp ops {r['ops']/1e6:.1f}M ratio {r['ratio']:.2f}", flush=True)
        json.dump(res, open(fn, "w"), indent=1); return
    for k, ld, oc, on in jobs:
        if k in res:
            continue
        r = score(ld(), oc, on, k); res[k] = r
        print(f"{k}: exact old {r['exact_old']:.6f} fix {r['exact_fix']:.6f} | recipe err fix {100*r['err_fix']:+.3f}pp "
              f"ratio old {r['ratio_old']:.2f} -> truth-only {r['ratio_truthonly']:.2f} -> full fix {r['ratio_fix']:.2f} | "
              f"noisy old {r['noisy_ratio_old']:.2f} -> fix {r['noisy_ratio_fix']:.2f}", flush=True)
        json.dump(res, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
