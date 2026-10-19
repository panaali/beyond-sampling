"""T452 tests (pre-registered in README.md).
usage: python t452_test.py gate SHARDS STEP     identity of cell_geometry_faster vs geom_fast.cell_geometry_fast on every
                                                STEP-th cell of the (9,3,2) shards -> results_gate_<shards>.jsonl
       python t452_test.py speed                the 87 tier-2 sizing cells: geometry + mass_cell_fast (budget 4000),
                                                reference vs new, timed, [lb, ub] compared -> results_speed.json
       python t452_test.py ddcheck N            double-double volume error vs exact fmpq on N simplices"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "event_cone_rigour")); sys.path.insert(0, HERE)
import geom_fast as GF  # noqa: E402
import geom_faster as G2  # noqa: E402
import m5t_fast as F  # noqa: E402
import rigour as R  # noqa: E402
from prof import tier2_sample  # noqa: E402


def shard(s):
    c = np.load(os.path.join(HERE, "data", f"surv_dfs_d0.333_9-3-2_k4to10_s{s}of6.npz"), allow_pickle=True)
    return c, tuple(int(v) for v in c["tri"]), 1 / R.q(float(c["delta"]))


def same_geom(a, b):
    if a is None or b is None:
        return a is None and b is None
    (ga, (Va, ia, Sa, Ma)), (gb, (Vb, ib, Sb, Mb)) = a, b
    if not (Va == Vb and ia == ib and Ma == Mb):
        return False
    return all((np.array_equal(x, y) and getattr(x, "dtype", None) == getattr(y, "dtype", None)) if isinstance(x, np.ndarray)
               else x == y for x, y in zip(ga, gb))


def run(fn, *a, **k):
    t = time.perf_counter()
    try:
        return fn(*a, **k), time.perf_counter() - t, None
    except MemoryError:
        return None, time.perf_counter() - t, "MemoryError"
    except Exception as e:  # noqa: BLE001
        return None, time.perf_counter() - t, repr(e)[:200]


def gate(shards, step, maxcells=25):
    """Resumable; at most maxcells new cells per process (a fresh process per batch keeps the address space small).
    A cell recorded with an error is retried once in a fresh process (attempt 2) before it counts."""
    path = os.path.join(HERE, f"results_gate_{''.join(map(str, shards))}.jsonl")
    rec_by = {}
    if os.path.exists(path):
        for r in map(json.loads, open(path)):
            rec_by[(r["s"], r["i"])] = r
    todo = []
    for s in shards:
        N = len(np.load(os.path.join(HERE, "data", f"surv_dfs_d0.333_9-3-2_k4to10_s{s}of6.npz"), allow_pickle=True)["gates"])
        for i in range(0, N, step):
            r = rec_by.get((s, i))
            if r is None or (r["ok"] is not True and r.get("attempt", 1) < 2):
                todo.append((s, i, 1 if r is None else 2))
    if not todo:
        print("GATE DONE"); return
    with open(path, "a") as fo:
        cur = None
        for s, i, att in todo[:maxcells]:
            if cur != s:
                c, tri, norm = shard(s); gates = c["gates"]; cur = s
                g = None
            g = gates[i].astype(int)
            b, tb, eb = run(G2.cell_geometry_faster, tri, g, 10, norm=norm, event=True, return_exact=True)
            a, ta, ea = run(GF.cell_geometry_fast, tri, g, 10, norm=norm, event=True, return_exact=True)
            rec = dict(s=s, i=i, t_ref=ta, t_new=tb, err_ref=ea, err_new=eb, attempt=att)
            if ea is None and eb is None:
                rec["ok"] = same_geom(a, b)
                rec["vertices"] = a[0][-1] if a else 0; rec["simplices"] = len(a[0][5]) if a else 0
            elif ea is None:
                rec["ok"] = False                  # the new code failed where the reference did not
            else:
                rec["ok"] = None                   # reference failed: not comparable
            del a, b
            fo.write(json.dumps(rec) + "\n"); fo.flush()


def speed():
    c, tri, norm = shard(0)
    G2.cell_geometry_faster(tri, c["gates"][284].astype(int), 10, norm=norm, event=True)    # numba compile
    rows = []
    for i in tier2_sample():
        g = c["gates"][i].astype(int)
        gr, tgr, er = run(GF.cell_geometry_fast, tri, g, 10, norm=norm, event=True)
        mr, tmr, emr = run(F.mass_cell_fast, tri, g, 10, 2.5, budget=4000, norm=norm, event=True, geom=gr) if er is None else (None, 0.0, er)
        del gr
        gn, tgn, en = run(G2.cell_geometry_faster, tri, g, 10, norm=norm, event=True)
        mn, tmn, emn = run(F.mass_cell_fast, tri, g, 10, 2.5, budget=4000, norm=norm, event=True, geom=gn) if en is None else (None, 0.0, en)
        del gn
        rows.append(dict(i=i, t_geom_ref=tgr, t_mass_ref=tmr, t_geom_new=tgn, t_mass_new=tmn, err_ref=er or emr, err_new=en or emn,
                         same_mass=(mr == mn) if (er is None and en is None and emr is None and emn is None) else None, ub=mr[1] if mr else None))
        print(rows[-1], flush=True)
    ok = [r for r in rows if r["err_ref"] is None]
    s = dict(cells=len(rows), comparable=len(ok), mass_mismatch=sum(r["same_mass"] is False for r in ok),
             new_errors_where_ref_ok=sum(r["err_new"] is not None for r in ok),
             geom_ref=sum(r["t_geom_ref"] for r in ok), geom_new=sum(r["t_geom_new"] for r in ok),
             mass_ref=sum(r["t_mass_ref"] for r in ok), mass_new=sum(r["t_mass_new"] for r in ok))
    s["speedup_geom"] = s["geom_ref"] / s["geom_new"]
    s["speedup_tier2"] = (s["geom_ref"] + s["mass_ref"]) / (s["geom_new"] + s["mass_new"])
    print(json.dumps(s, indent=1))
    json.dump(dict(summary=s, cells=rows), open(os.path.join(HERE, "results_speed.json"), "w"), indent=0)


def ddcheck(nsimp):
    """Empirical check of the double-double volume error bound: on simplices of sampled (9,3,2) cells, the exact fmpq
    volume against xh + xl, relative error / (claimed bound err / |x|)."""
    import math
    from flint import fmpq_mat, fmpq
    c, tri, norm = shard(0)
    rng = np.random.default_rng(452)
    worst = 0.0; worst_abs = 0.0; cnt = 0; viol = 0
    for i in rng.permutation(len(c["gates"])):
        if cnt >= nsimp:
            break
        rs = G2.cell_rows(tri, c["gates"][i].astype(int), 10, norm, True)
        if rs is None:
            continue
        cap = {}
        ov = G2.volumes_staircase

        def vs(V, inc, S, n, A, b):
            cap.update(V=V, inc=inc, S=S, n=n, A=A, b=b); return ov(V, inc, S, n, A, b)
        G2.volumes_staircase = vs
        try:
            out = G2.cell_geometry_faster(tri, c["gates"][i].astype(int), 10, norm=norm, event=True)
        except MemoryError:
            out = None
        finally:
            G2.volumes_staircase = ov
        if out is None or "S" not in cap or len(cap["S"]) > 20000:
            continue
        V, inc, S, n, A, b = (cap[k] for k in ("V", "inc", "S", "n", "A", "b"))
        Sa = np.asarray(S, np.int64); m = len(A)
        incm = np.array([sum(1 << j for j in f) for f in inc], dtype=np.uint64)
        rows, mask, ok = G2._chains(Sa, incm, m)
        for t in rng.choice(len(S), min(200, len(S)), replace=False):
            if not ok[t]:
                continue
            ph, pl = 1.0, 0.0; rel = 0.0
            for l in range(n):
                h, lw, e = G2.q2dd(sum(A[rows[t, l]][k] * V[Sa[t, l]][k] for k in range(n)) - b[rows[t, l]])
                ph, pl = G2._dd_mul(ph, pl, h, lw); rel += e / h
            Mx = fmpq_mat(n, n, [x for r in rows[t] for x in A[r]])
            h, lw, e = G2.q2dd(abs(Mx.det())); rel += e / h
            qh, ql = G2._dd_mul(h, lw, float(math.factorial(n)), 0.0)
            xh, xl = G2._dd_div(ph, pl, qh, ql)
            exact = abs(fmpq_mat(n, n, [V[Sa[t, j]][k] - V[Sa[t, 0]][k] for j in range(1, n + 1) for k in range(n)]).det()) / math.factorial(n)
            err_claim = abs(xh) * (1.000001 * (rel + (n + 3) * G2.EPS_OP) * 2.0)
            got = abs((fmpq(*xh.as_integer_ratio()) + fmpq(*xl.as_integer_ratio())) - exact)
            r_ = float(got) / err_claim
            worst = max(worst, r_); worst_abs = max(worst_abs, float(got / exact)); cnt += 1; viol += r_ > 1
    res = dict(simplices=cnt, violations=viol, worst_err_over_claimed=worst, worst_rel_err=worst_abs, claimed_rel_order=(9 + 3) * G2.EPS_OP * 2)
    print(json.dumps(res)); json.dump(res, open(os.path.join(HERE, "results_ddcheck.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "gate":
        gate([int(x) for x in sys.argv[2].split(",")], int(sys.argv[3]))
    elif sys.argv[1] == "speed":
        speed()
    elif sys.argv[1] == "ddcheck":
        ddcheck(int(sys.argv[2]))
