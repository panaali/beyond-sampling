"""T281 cost re-audit under the written convention (TASKS Directive 8, DRI 2026-10-18, from T312):
charge every weight-dependent operation at T196 rates (multiply-add = 2; Phi = 20), the estimator's own forwards at
their instrumented FLOPs; do not charge precomputed functions of the input law alone.

usage: PANEL=hp1|hp2 python cost_audit.py t306|t300
Runs the frozen estimator unchanged (estimates are checked against the scorer's raw file), recounts its FLOPs, and
writes data/<panel>/raw_<name>_conv2.jsonl for aggregate_F. Counting rules (per model):

t306_r1w (r1_walk/r1w2.est, A=B=False):
  * walk forwards: fpe_walk (the code's own count of value+tangent forward equivalents, reproduced to 0.3% by T312)
    x 5760 FLOPs (instrumented forward: T(2H^2+2H) + 2HT);
  * W._profile pieces from the model's walks: per (interval, GL node) and per finite endpoint: shift 1 + Phi 20
    (+ 3 for ^8 in '23') ; difference 1 ; accumulate 1;
  * model-free pieces (tail [TMAX, inf) and the full line [0, inf)): Phi tables precomputed once (not charged);
    accumulate 1 per (row, node);
  * combination tot += (I12 I23 pdf) @ w: 4 FLOPs per (triple, node) per region (pdf at nodes: model-free).
t300_r0 (r0_integral/tb.est, n23=1, newton=1, nblind=0, use12=False):
  * 2 forwards (value + tangent, same instrumented cost) per triple x 5760;
  * per (triple, node): V2N - T23 (1), Phi (20), ^8 (3), times the model-free table 90 pdf (1 - Phi(V2N)) (1),
    multiply-add with the GL weight (2) = 27; the t12 = 0 factor is model-free.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score  # noqa: E402

SUB = os.path.join(score.ROOT, "data", "subs")
FWD = 5760.0
PHI = 20.0


def t306():
    f = score.load_sub(os.path.join(SUB, "t306_684a22b9/m16/approaches/r1_walk/estimator.py"), "est")
    import r1w2, walk as W
    orig = W._profile
    acc = {"phi": 0.0}

    def prof(pieces, n, nodes, kind, tmax=W.TMAX):
        G = len(nodes)
        for rows, lo, hi in pieces:
            m = hi > lo
            if not m.any():
                continue
            a, b = lo[m], hi[m]
            free = (np.all(a == 0.0) or np.all(np.abs(a - r1w2.TMAX) < 1e-12)) and np.all(b >= W.INF - 1e-12)
            if free:                                         # tail / full line: model-free table, accumulate only
                acc["phi"] += m.sum() * G
                continue
            nend = (b < W.INF - 1e-12).sum() + len(a)        # finite endpoints
            per = 1 + PHI + (3 if kind == "23" else 0)
            acc["phi"] += nend * G * per + 2 * m.sum() * G   # difference + accumulate
        return orig(pieces, n, nodes, kind, tmax)
    W._profile = prof

    def est(model):
        acc["phi"] = 0.0
        e, c, parts = r1w2.est(model, parts=True)
        new = parts["fpe_walk"] * FWD + acc["phi"] + 4 * r1w2.NT * W.G * len(r1w2.REG[False])
        return e, c, new
    return est


def t300():
    score.load_sub(os.path.join(SUB, "t300_8df9793b/m16/approaches/r0_shared/estimator.py"), "est")
    import tb, r0
    n = len(r0.TRIPLES)

    def est(model):
        m = tuple(np.asarray(w, np.float64) for w in model)
        e, c = tb.est(m, n23=1, newton=1, nblind=0, use12=False)
        return e, c, 2 * n * FWD + n * tb.G_NODES * 27
    return est


if __name__ == "__main__":
    name = dict(t306="t306_r1w", t300="t300_r0")[sys.argv[1]]
    est = dict(t306=t306, t300=t300)[sys.argv[1]]()
    names, man, models, tr = score.load_panel()
    old = {r["k"]: r for r in map(json.loads, open(os.path.join(score.D, f"raw_{name}.jsonl")))}
    with open(os.path.join(score.D, f"raw_{name}_conv2.jsonl"), "w") as f:
        for k, nm in enumerate(names):
            e, c, new = est(models[k])
            assert abs(e - old[k]["est"]) < 1e-12 and abs(c - old[k]["cost"]) < 1e-3 * c, (k, e, old[k])
            f.write(json.dumps(dict(k=k, name=nm, B=0, est=e, cost=float(new), cost_old=float(c))) + "\n")
    rows = [json.loads(l) for l in open(os.path.join(score.D, f"raw_{name}_conv2.jsonl"))]
    co, cn = np.array([r["cost_old"] for r in rows]), np.array([r["cost"] for r in rows])
    print(f"{name} {score.PANEL}: old {co.mean():.4g} FLOPs ({co.mean() / score.FPE:.0f} FPE), "
          f"new {cn.mean():.4g} ({cn.mean() / score.FPE:.0f} FPE), ratio {np.mean(cn / co):.3f}")
    score.aggregate_F(name + "_conv2", "conv2", (0,))
