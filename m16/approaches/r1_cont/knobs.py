"""T326 step 2: the pre-registered 80-configuration knob grid on T317e (README, T326 step 2 pre-registration).
Knobs: tmax (walk window; gap beyond assumed correct), cap (later cells followed per row; at the cap the current cell's
affine logits are frozen out to tmax: its correct set is solved on [t, tmax], at no extra charge beyond the cell's
own readout), line subset. Everything else is cont_e (T317e) unchanged; (tmax .5, cap inf, all lines) == cont_e.est.
usage: python knobs.py -> results_knobs.json"""
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont_e as E  # noqa: E402
from cont_e import H, T, NT  # noqa: E402

TMAXS = (0.5, 0.3, 0.2, 0.1)
CAPS = (np.inf, 4, 2, 1, 0)
SUBSETS = {"all": ("nb-23", "nb-12", "b-23"), "drop_nb12": ("nb-23", "b-23"), "drop_b23": ("nb-23", "nb-12"),
           "nb23_only": ("nb-23",)}
FPE = 5948


def walk_cap(model, x0, d, k, tmax, cap, max_cells=400):
    """cont_e.walk_cont with a per-row cap on later cells (cap = inf reproduces it exactly)."""
    Wi, Wh, Wo = model; n = len(x0); pieces = []
    Z, DZ, G, R, J, nodes = E.first_cell(model, x0, d)
    flops = nodes * E.F_STEP + n * E.F_CELL_T317
    tref = np.full(n, E.EPS)
    hT = np.where(G[:, -1], Z[:, -1], 0.0); dhT = np.where(G[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0; cnt = np.zeros(n)
    while len(act) and it < max_cells:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
        nxt = np.where(cnt[act] >= cap, tmax, nxt)                         # capped rows: freeze this cell to tmax
        lo, hi = E.correct_interval(L[act], dL[act], tref[act], k[act], t[act], nxt)
        pieces.append((act.copy(), lo, hi))
        t[act] = nxt
        go = act[nxt < tmax]; ss = sstar[nxt < tmax]
        if not len(go):
            break
        cnt[go] += 1
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
            flops += 2 * T * H + 1 + H + (T - 1 - s0) * E.F_TAN + (T - s0) * E.F_ROOT + E.F_LATER_READ_T317
        act = go
    n_later = sum(len(p[0]) for p in pieces) - n
    flops += n * (E.F_CELL - E.F_CELL_T317) + n_later * (E.F_LATER_READ - E.F_LATER_READ_T317)   # as walk_counted
    return pieces, flops


def line_profiles(model):
    """{(line, tmax, cap): (I (NT, G), FLOPs)} for the walked lines; model-free replacements for dropped lines."""
    model = tuple(np.asarray(w, np.float64) for w in model); k = E.TRI[:, 1]; out = {}
    for tb in E.TABLES:
        for kind in ("23", "12"):
            key = f"line{kind}"
            if key not in tb:
                continue
            x, dx = tb[key]; line = f"{tb['name']}-{kind}"
            for tmax, cap in itertools.product(TMAXS, CAPS):
                p, f = walk_cap(model, x, dx, k, tmax, cap)
                R, A, B, ops = E.merge(p, NT, tmax)
                I, fp = E.profile(R, A, B, NT, tb["nodes"], kind, tmax, tb["phi0"])
                out[(line, tmax, cap)] = (I, f + ops + fp)
    return out


def combine(prof, tmax, cap, subset):
    lines = SUBSETS[subset]; tot = np.zeros(NT); flops = 0.0
    for tb in E.TABLES:
        nm = tb["name"]; G_ = len(tb["nodes"])
        if f"{nm}-23" in lines:
            I23, f = prof[(f"{nm}-23", tmax, cap)]; flops += f
        else:
            I23 = np.repeat(tb["phi0"]["m8"][None], NT, 0)                  # g23 assumed correct (model-free)
        if nm.startswith("nb") and f"{nm}-12" in lines:
            I12, f = prof[(f"{nm}-12", tmax, cap)]; flops += f
        else:
            I12 = np.repeat(tb["blind12"][None], NT, 0)                     # g12 assumed correct (model-free)
        tot += (I12 * I23 * tb["dens"][None]) @ tb["wts"]
        flops += NT * G_ * 4
    flops += NT
    return float(tot.mean()), float(flops)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(HERE, "..", "r0_integral"))
    import r0  # noqa: E402
    truth = r0.truths()["truths"]; rows = {}; t0 = time.time(); check = []
    cfgs = list(itertools.product(TMAXS, CAPS, SUBSETS))
    for nm, m in r0.dev2():
        prof = line_profiles(m)
        rows[nm] = {f"{a}|{b}|{c}": combine(prof, a, b, c) for a, b, c in cfgs}
        e0, f0 = E.est(m); e1, f1 = rows[nm][f"0.5|{np.inf}|all"]; check.append(e0 == e1 and f0 == f1)
        print(nm, "T317e reproduced:", check[-1], round(time.time() - t0), flush=True)
    res = []
    for a, b, c in cfgs:
        key = f"{a}|{b}|{c}"
        err = np.array([rows[n][key][0] - truth[n]["acc"] for n in rows]); p = np.array([truth[n]["acc"] for n in rows])
        Cm = float(np.mean([rows[n][key][1] for n in rows])) / FPE; v = float(np.mean(p * (1 - p)))
        rms = float(np.sqrt((err ** 2).mean()))
        res.append(dict(tmax=a, cap=None if b == np.inf else int(b), subset=c, C=Cm, rms_pp=100 * rms,
                        bias_pp=100 * float(err.mean()), F_lenient=rms ** 2 * Cm / v, F_strict=rms ** 2 * max(Cm, 1000) / .074))
    json.dump(dict(check_all=bool(all(check)), configs=res, rows=rows), open(os.path.join(HERE, "results_knobs.json"), "w"))
    print("T317e reproduced on all 80:", all(check))
    res.sort(key=lambda r: r["C"]); best = None; front = []
    for r in res:
        if not front or r["F_strict"] < front[-1]["F_strict"]:
            front.append(r)
    print("\nPareto frontier (C, strict F):")
    for r in front:
        print(f"  C {r['C']:7.0f}  RMS {r['rms_pp']:.3f}  bias {r['bias_pp']:+.3f}  F_strict {r['F_strict']:.3f}  F_len {r['F_lenient']:.2f}  "
              f"tmax {r['tmax']} cap {r['cap']} {r['subset']}")
    ok = [r for r in res if r["C"] <= 2000]
    sel = min(ok, key=lambda r: r["F_strict"]) if ok else None
    print("\nselected (min strict F at C <= 2000):", sel, "\nGO:", bool(sel and sel["F_strict"] <= 0.8))
