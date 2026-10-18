"""T326 step 1: T317e's executed cost by component, on all 80 DEV2 (instrumented copy of cont_e.walk_cont; the total
must equal cont_e.est's charged FLOPs exactly). -> results_cost_table.json"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont_e as E  # noqa: E402
from cont_e import H, T, C, NT, TMAX  # noqa: E402

FPE = 5948


def walk_tally(model, x0, d, k, tmax, tag, acc):
    Wi, Wh, Wo = model; n = len(x0)
    Z, DZ, G, R, J, nodes = E.first_cell(model, x0, d)
    acc[f"{tag}: B2 prefix tree (first-cell steps)"] += nodes * E.F_STEP
    acc[f"{tag}: first-cell readout + correct set"] += n * E.F_CELL
    acc[f"{tag}: #rows"] += n; acc[f"{tag}: #prefix nodes"] += nodes
    tref = np.full(n, E.EPS)
    hT = np.where(G[:, -1], Z[:, -1], 0.0); dhT = np.where(G[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0; pieces = []
    while len(act) and it < 400:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
        lo, hi = E.correct_interval(L[act], dL[act], tref[act], k[act], t[act], nxt)
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
            acc[f"{tag}: restart values + toggle"] += 2 * T * H + 1 + H
            acc[f"{tag}: restart tangents"] += (T - 1 - s0) * E.F_TAN
            acc[f"{tag}: restart roots"] += (T - s0) * E.F_ROOT
            acc[f"{tag}: later-cell readout + correct set"] += E.F_LATER_READ
            acc[f"{tag}: #later cells"] += 1
        act = go
    return pieces


def tally(model):
    model = tuple(np.asarray(w, np.float64) for w in model)
    k = E.TRI[:, 1]; acc = defaultdict(float)
    for tb in E.TABLES:
        lines = [("23", tb["line23"])] + ([("12", tb["line12"])] if "line12" in tb else [])
        for kind, (x, dx) in lines:
            tag = f"{tb['name']}-{kind}"
            p = walk_tally(model, x, dx, k, TMAX, tag, acc)
            R, A, B, ops = E.merge(p, NT, TMAX); acc[f"{tag}: merge"] += ops
            _, f = E.profile(R, A, B, NT, tb["nodes"], kind, TMAX, tb["phi0"]); acc[f"{tag}: Phi profile"] += f
        acc["combination (I12 I23 dens . wts)"] += NT * len(tb["nodes"]) * 4
    acc["combination (I12 I23 dens . wts)"] += NT
    return acc


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(HERE, "..", "r0_integral"))
    import r0  # noqa: E402
    per = []; check = 0.0
    for nm, m in r0.dev2():
        a = tally(m); tot = sum(v for kk, v in a.items() if "#" not in kk)
        e, f = E.est(m); check = max(check, abs(tot - f)); per.append(a)
    keys = sorted({kk for a in per for kk in a})
    mean = {kk: float(np.mean([a.get(kk, 0.0) for a in per])) for kk in keys}
    tot = sum(v for kk, v in mean.items() if "#" not in kk)
    print(f"max |tally - est flops| = {check}; mean C = {tot / FPE:.1f} FPE")
    # grouped
    groups = defaultdict(float)
    for kk, v in mean.items():
        if "#" in kk:
            continue
        comp = kk.split(": ")[-1]
        comp = {"restart values + toggle": "later cells: values + toggle", "restart tangents": "later cells: tangents",
                "restart roots": "later cells: roots", "later-cell readout + correct set": "later cells: readout + correct set"}.get(comp, comp)
        groups[comp] += v
    print("\nby component (all lines):")
    for kk, v in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {kk:45s} {v / FPE:8.1f} FPE  {100 * v / tot:5.1f}%")
    lines = defaultdict(float)
    for kk, v in mean.items():
        if "#" not in kk and ": " in kk:
            lines[kk.split(": ")[0]] += v
    print("\nby line:")
    for kk, v in lines.items():
        print(f"  {kk:10s} {v / FPE:8.1f} FPE  {100 * v / tot:5.1f}%   rows {mean[kk + ': #rows']:.0f}, prefix nodes "
              f"{mean[kk + ': #prefix nodes']:.0f}, later cells {mean.get(kk + ': #later cells', 0):.0f}")
    print("\nfull:"); [print(f"  {kk:55s} {v:12.0f}" + ("" if "#" in kk else f"  {100 * v / tot:5.1f}%")) for kk, v in mean.items()]
    json.dump(dict(check=check, mean=mean, groups=dict(groups), lines=dict(lines), C=tot / FPE),
              open(os.path.join(HERE, "results_cost_table.json"), "w"), indent=1)
