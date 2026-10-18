"""T326 (DRI follow-up): ORACLE floors for the two big cost items, per DEV2 model (lower bounds on what any deductive
screen could achieve; the selections below use the walk's own results, so they are not estimators):
(a) prefix tree restricted to the triples that carry the error mass: for each line, rows sorted by their contribution
    to the estimate's deficit (estimate with this line assumed correct minus actual), the smallest set carrying
    q in {100, 99, 90, 50}% of it; prefix nodes counted as distinct (input, tangent) prefixes over that set only.
(b) later cells with an ideal per-row cap: each row followed only up to its LAST cell that holds a wrong piece
    (cells after it are all-correct, so a perfect margin screen could skip them). Tangent / root / value / readout
    cost of the kept cells, plus the same restricted to (a)'s 99% rows.
-> results_floors.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont_e as E  # noqa: E402
from cont_e import H, T, NT, TMAX  # noqa: E402

FPE = 5948
QS = (1.0, 0.99, 0.9, 0.5)


def prefix_nodes(x0, d, rows):
    if len(rows) == 0:
        return 0
    x0, d = x0[rows], d[rows]; inv_prev = np.zeros(len(rows), int); n = 0
    for s in range(T):
        key = np.round(np.column_stack([inv_prev, x0[:, s], d[:, s]]), 12)
        _, inv = np.unique(key, axis=0, return_inverse=True); inv_prev = inv.ravel(); n += inv_prev.max() + 1
    return int(n)


def walk_rec(model, x0, d, k, tmax):
    """cont_e.walk_cont, recording per later cell: row, its cost parts, and whether each cell holds a wrong piece."""
    Wi, Wh, Wo = model; n = len(x0); pieces = []
    Z, DZ, G, R, J, nodes = E.first_cell(model, x0, d)
    tref = np.full(n, E.EPS)
    hT = np.where(G[:, -1], Z[:, -1], 0.0); dhT = np.where(G[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0
    cells = []                              # (row, cell index, wrong?, cost) ; cell 0 = first cell (cost 0 here)
    cidx = np.zeros(n, int); pending = {}   # cost of the restart that opened the current cell of each row
    while len(act) and it < 400:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
        lo, hi = E.correct_interval(L[act], dL[act], tref[act], k[act], t[act], nxt)
        wrong = (hi - lo) < (nxt - t[act]) * (1 - 1e-12) - 1e-15
        for r_, w_ in zip(act, wrong):
            cells.append((int(r_), int(cidx[r_]), bool(w_), pending.pop(int(r_), 0.0)))
        pieces.append((act.copy(), lo, hi))
        t[act] = nxt
        go = act[nxt < tmax]; ss = sstar[nxt < tmax]
        if not len(go):
            break
        tn = t[go]
        Z[go] += (tn - tref[go])[:, None, None] * DZ[go]; L[go] += (tn - tref[go])[:, None] * dL[go]; tref[go] = tn
        jj = J[go, ss]; G[go, ss, jj] = ~G[go, ss, jj]
        for idx in range(len(go)):
            row, s0 = go[idx], ss[idx]
            dh = np.where(G[row, s0], DZ[row, s0], 0.0)
            for s in range(s0 + 1, T):
                DZ[row, s] = Wh @ dh + d[row, s] * Wi
                dh = np.where(G[row, s], DZ[row, s], 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tn[idx] - Z[row, s0:] / DZ[row, s0:]
            r = np.where((r > tn[idx] + 1e-12 * (1 + abs(tn[idx]))) & np.isfinite(r), r, np.inf)
            r[0, jj[idx]] = np.inf
            R[row, s0:] = r.min(1); J[row, s0:] = r.argmin(1)
            dL[row] = Wo @ dh
            pending[int(row)] = dict(tan=(T - 1 - s0) * E.F_TAN, root=(T - s0) * E.F_ROOT, val=2 * T * H + 1 + H,
                                     read=E.F_LATER_READ)
            cidx[row] += 1
        act = go
    return pieces, nodes, cells


def floors(model):
    model = tuple(np.asarray(w, np.float64) for w in model); k = E.TRI[:, 1]
    walked = {}
    for tb in E.TABLES:
        for kind in ("23", "12"):
            if f"line{kind}" in tb:
                x, dx = tb[f"line{kind}"]; p, nodes, cells = walk_rec(model, x, dx, k, TMAX)
                R, A, B, _ = E.merge(p, NT, TMAX); I, _ = E.profile(R, A, B, NT, tb["nodes"], kind, TMAX, tb["phi0"])
                walked[f"{tb['name']}-{kind}"] = dict(I=I, nodes=nodes, cells=cells, x=x, dx=dx)
    out = {}
    for tb in E.TABLES:
        nm = tb["name"]
        I23 = walked[f"{nm}-23"]["I"]; I12 = walked[f"{nm}-12"]["I"] if f"{nm}-12" in walked else tb["blind12"][None]
        full23 = tb["phi0"]["m8"][None]; full12 = tb["blind12"][None]
        for kind, I, other, full in (("23", I23, I12, full23), ("12", I12 if f"{nm}-12" in walked else None, I23, full12)):
            line = f"{nm}-{kind}"
            if line not in walked:
                continue
            w = walked[line]
            deficit = (((full - I) * other * tb["dens"][None]) @ tb["wts"]) / NT        # per row, in accuracy units
            dpos = np.where(deficit > 1e-13, deficit, 0.0); order = np.argsort(-dpos); cum = np.cumsum(dpos[order]); order = np.argsort(-dpos); cum = np.cumsum(dpos[order])
            tot = cum[-1] if len(cum) else 0.0
            rec = dict(nodes_all=w["nodes"], deficit_total=float(deficit.sum()), rows_nonzero=int((dpos > 0).sum()))
            # (b): ideal per-row cap = last wrong cell
            last_wrong = {}
            for r_, c_, wr, _ in w["cells"]:
                if wr:
                    last_wrong[r_] = max(last_wrong.get(r_, -1), c_)
            def later_cost(rows_ok):
                tot_c = dict(tan=0.0, root=0.0, val=0.0, read=0.0); full_c = dict(tan=0.0, root=0.0, val=0.0, read=0.0)
                for r_, c_, wr, cost in w["cells"]:
                    if not cost:
                        continue
                    for kk in full_c:
                        full_c[kk] += cost[kk]
                    if r_ in rows_ok and c_ <= last_wrong.get(r_, -1):
                        for kk in tot_c:
                            tot_c[kk] += cost[kk]
                return full_c, tot_c
            for q in QS:
                m = int(np.searchsorted(cum, q * tot * (1 - 1e-12)) + 1) if tot > 0 else 0
                rows = np.sort(order[:m]); rows = rows[dpos[rows] > 0] if q == 1.0 else rows
                full_c, kept = later_cost(set(rows.tolist()))
                rec[f"q{q}"] = dict(rows=int(len(rows)), nodes=prefix_nodes(w["x"], w["dx"], rows),
                                    later_full=full_c, later_kept_idealcap=kept)
            full_c, kept_all = later_cost(set(range(NT)))
            rec["later_idealcap_allrows"] = kept_all
            out[line] = rec
    return out


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(HERE, "..", "r0_integral"))
    import r0  # noqa: E402
    res = {}
    for nm, m in r0.dev2():
        res[nm] = floors(m); print(nm, {l: (v["rows_nonzero"], v["q0.99"]["rows"]) for l, v in res[nm].items()}, flush=True)
    json.dump(res, open(os.path.join(HERE, "results_floors.json"), "w"))
    lines = list(next(iter(res.values())))
    print("\nmean over 80 DEV2, FPE (per line; prefix tree = nodes x F_STEP):")
    for l in lines:
        g = lambda f: float(np.mean([f(res[n][l]) for n in res]))
        print(f"{l}: rows with error {g(lambda v: v['rows_nonzero']):.0f}/720; prefix tree all {g(lambda v: v['nodes_all']) * E.F_STEP / FPE:.0f}")
        for q in QS:
            print(f"   q{q:.2f}: rows {g(lambda v: v[f'q{q}']['rows']):6.1f}  prefix tree {g(lambda v: v[f'q{q}']['nodes']) * E.F_STEP / FPE:6.1f}"
                  f"  later cells (ideal cap, these rows) {g(lambda v: sum(v[f'q{q}']['later_kept_idealcap'].values())) / FPE:6.1f}"
                  f"  [tangents {g(lambda v: v[f'q{q}']['later_kept_idealcap']['tan']) / FPE:6.1f}]")
        print(f"   ideal cap, all rows: later cells {g(lambda v: sum(v['later_idealcap_allrows'].values())) / FPE:.0f}"
              f" of {g(lambda v: sum(v['q1.0']['later_full'].values())) / FPE:.0f} FPE (tangents {g(lambda v: v['later_idealcap_allrows']['tan']) / FPE:.0f} of {g(lambda v: v['q1.0']['later_full']['tan']) / FPE:.0f})")
