"""T331 (Fable r5 D1): T317e with tmax upward, and with a zero-cost affine extension beyond tmax (README, T331
pre-registration). ext: the last walked cell's correct set is solved on [t, inf) instead of [t, tmax] in the same single
correct_interval call, so the beyond-tmax gap is wrong past the frozen cell's root instead of assumed correct.
est(model, tmax, ext) -> (estimate, FLOPs as executed); (.5, False) == cont_e.est. usage: python t331.py"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cont_e as E  # noqa: E402
from cont_e import H, T, NT, INF  # noqa: E402

VARIANTS = {"T317e": (0.5, False), "i_tmax1": (1.0, False), "ii_tmax2": (2.0, False), "iii_ext": (0.5, True),
            "iv_tmax1_ext": (1.0, True), "v_tmax2_ext": (2.0, True)}


def walk(model, x0, d, k, tmax, ext, max_cells=400):
    Wi, Wh, Wo = model; n = len(x0); pieces = []
    Z, DZ, G, R, J, nodes = E.first_cell(model, x0, d)
    flops = nodes * E.F_STEP + n * E.F_CELL_T317
    tref = np.full(n, E.EPS)
    hT = np.where(G[:, -1], Z[:, -1], 0.0); dhT = np.where(G[:, -1], DZ[:, -1], 0.0)
    L = hT @ Wo.T; dL = dhT @ Wo.T
    t = np.zeros(n); act = np.arange(n); it = 0
    while len(act) and it < max_cells:
        it += 1
        sstar = R[act].argmin(1); nxt = np.minimum(R[act, sstar], tmax)
        hi_arg = np.where(nxt >= tmax, INF, nxt) if ext else nxt
        lo, hi = E.correct_interval(L[act], dL[act], tref[act], k[act], t[act], hi_arg)
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
            flops += 2 * T * H + 1 + H + (T - 1 - s0) * E.F_TAN + (T - s0) * E.F_ROOT + E.F_LATER_READ_T317
        act = go
    n_later = sum(len(p[0]) for p in pieces) - n
    flops += n * (E.F_CELL - E.F_CELL_T317) + n_later * (E.F_LATER_READ - E.F_LATER_READ_T317)
    return pieces, flops


def merge_ext(pieces, n):
    """E.merge without the model-free tail: pieces already extend past tmax (to the frozen root, or INF)."""
    cur_a = np.full(n, np.nan); cur_b = np.full(n, np.nan); out_r, out_a, out_b = [], [], []; ops = 0
    for rows, lo, hi in pieces:
        m = hi > lo; r, a, b = rows[m], lo[m], hi[m]; ops += 2 * len(rows)
        touch = cur_b[r] == a
        cur_b[r[touch]] = b[touch]
        new = ~touch
        old = r[new][~np.isnan(cur_a[r[new]])]
        out_r.append(old); out_a.append(cur_a[old]); out_b.append(cur_b[old])
        cur_a[r[new]] = a[new]; cur_b[r[new]] = b[new]
    open_ = np.flatnonzero(~np.isnan(cur_a))
    out_r.append(open_); out_a.append(cur_a[open_]); out_b.append(cur_b[open_])
    return np.concatenate(out_r), np.concatenate(out_a), np.concatenate(out_b), ops


def est(model, tmax=0.5, ext=False):
    model = tuple(np.asarray(w, np.float64) for w in model)
    k = E.TRI[:, 1]; flops = 0.0; tot = np.zeros(NT)
    for tb in E.TABLES:
        nodes = tb["nodes"]; I = {}
        for kind in ("23", "12"):
            if f"line{kind}" not in tb:
                continue
            x, dx = tb[f"line{kind}"]
            p, f = walk(model, x, dx, k, tmax, ext); flops += f
            R, A, B, ops = merge_ext(p, NT) if ext else E.merge(p, NT, tmax); flops += ops
            I[kind], f = E.profile(R, A, B, NT, nodes, kind, tmax, tb["phi0"]); flops += f
        I12 = I["12"] if "12" in I else np.repeat(tb["blind12"][None], NT, 0)
        tot += (I12 * I["23"] * tb["dens"][None]) @ tb["wts"]
        flops += NT * len(nodes) * 4
    flops += NT
    return float(tot.mean()), float(flops)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(HERE, "..", "r0_integral"))
    import r0  # noqa: E402
    truth = r0.truths()["truths"]; rows = {}; chk = []
    for nm, m in r0.dev2():
        rows[nm] = {v: est(m, *a) for v, a in VARIANTS.items()}
        chk.append(rows[nm]["T317e"] == E.est(m)); print(nm, {v: (round(e, 5), round(f / 5948)) for v, (e, f) in rows[nm].items()}, flush=True)
    p = np.array([truth[n]["acc"] for n in rows]); var = float(np.mean(p * (1 - p))); s = {"T317e_reproduced": bool(all(chk))}
    for v in VARIANTS:
        e = np.array([rows[n][v][0] - truth[n]["acc"] for n in rows]); C = float(np.mean([rows[n][v][1] for n in rows])) / 5948
        mse = float((e ** 2).mean())
        s[v] = dict(C=C, rms_pp=100 * np.sqrt(mse), bias_pp=100 * float(e.mean()), mseC=mse * C, F_len=mse * C / var,
                    F_strict=mse * max(C, 1000) / .074)
    b = s["T317e"]
    for v in VARIANTS:
        s[v]["mseC_ratio"] = s[v]["mseC"] / b["mseC"]; s[v]["C_growth"] = s[v]["C"] / b["C"] - 1
        s[v]["bias_move_pp"] = s[v]["bias_pp"] - b["bias_pp"]
        s[v]["go"] = bool(s[v]["mseC_ratio"] <= .80)
        s[v]["stop"] = bool(s[v]["C_growth"] > .15 or abs(s[v]["bias_move_pp"]) < .08)
    print(json.dumps(s, indent=1))
    json.dump(dict(summary=s, rows=rows), open(os.path.join(HERE, "results_t331.json"), "w"), indent=1)
