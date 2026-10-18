"""T218: label the heavy tail of sci_tao's tie-face integrand by mechanism.

usage: python tail.py MODEL [logn]   MODEL in {z6s0..z6s4, orig, s005, s010}
-> results/MODEL.json: per rule seed (A=700, B=701) the tail points (g^2, labels) plus the global moments.
"""
import json
import os
import sys

import numpy as np
from scipy.special import ndtr

import core as C

HERE = os.path.dirname(os.path.abspath(__file__))
SQ2 = np.sqrt(2.0)


def target_model():
    from alg_zoo import example_2nd_argmax
    m = example_2nd_argmax().eval()
    f = lambda w: w.detach().numpy().astype(np.float64)
    return f(m.rnn.weight_ih_l0)[:, 0], f(m.rnn.weight_hh_l0), f(m.linear.weight)


def noisy(model, sigma, seed=0):
    """Same convention as m16/arc_eval.noisy_models (first copy of k): s.d. sigma * RMS of each matrix."""
    rng = np.random.default_rng(seed)
    return tuple(W + sigma * np.sqrt((W ** 2).mean()) * rng.standard_normal(W.shape) for W in model)


def get_model(name):
    if name.startswith("z6s"):
        return C.zoo(16, 6, seed=int(name[3:]))
    m = target_model()
    return {"orig": m, "s005": noisy(m, 0.005), "s010": noisy(m, 0.01)}[name]


def walk_one(model, y, a, b, tau, kind, eps=1e-10, max_cells=4000):
    """Scalar exact walk of one segment y + t d, |t| < tau (same geometry as core.walk_mass).
    Returns the cells: list of (lo, hi, (step, neuron) of the gate flip ending the cell or None,
    [(wlo, whi, wrong_class, mass)])."""
    Wi, Wh, Wo = model
    T = len(y); H = Wh.shape[0]
    d = np.zeros(T); d[a] = 1 / SQ2; d[b] = -1 / SQ2
    tpos, tneg = (a, b) if kind == "23" else (b, a)
    t = -tau; cells = []
    while t < tau and len(cells) < max_cells:
        tc = t + eps * (1 + abs(t))
        h = np.zeros(H); dh = np.zeros(H); nxt = tau; brk = None
        for s in range(T):
            z = Wh @ h + (y[s] + tc * d[s]) * Wi
            dz = Wh @ dh + d[s] * Wi
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tc - z / dz
            r = np.where((r > tc) & np.isfinite(r), r, np.inf)
            j = int(np.argmin(r))
            if r[j] < nxt:
                nxt = r[j]; brk = (s, j)
            g = z > 0
            h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
        L = Wo @ h; dL = Wo @ dh
        wrong = []
        for p_lo, p_hi, k in ((t, min(nxt, 0.0), tneg), (max(t, 0.0), nxt, tpos)):
            if p_hi <= p_lo:
                continue
            # piecewise: argmax of L + (s - tc) dL over [p_lo, p_hi]; split at all pairwise crossings
            pts = [p_lo, p_hi]
            for i in range(len(L)):
                for j2 in range(i + 1, len(L)):
                    e = dL[i] - dL[j2]
                    if e != 0:
                        r0 = tc - (L[i] - L[j2]) / e
                        if p_lo < r0 < p_hi:
                            pts.append(r0)
            pts = np.sort(pts)
            for lo, hi in zip(pts[:-1], pts[1:]):
                if hi <= lo:
                    continue
                mid = 0.5 * (lo + hi)
                cls = int(np.argmax(L + (mid - tc) * dL))
                if cls != k:
                    m = ndtr(hi) - ndtr(lo)
                    if wrong and wrong[-1][2] == cls and abs(wrong[-1][1] - lo) < 1e-12:
                        w = wrong[-1]; wrong[-1] = (w[0], hi, cls, w[3] + m)
                    else:
                        wrong.append((lo, hi, cls, m))
        cells.append((t, nxt, brk if nxt < tau else None, wrong))
        t = nxt
    return cells


def components(cells):
    """Merge wrong intervals across cell boundaries into connected components (lo, hi, mass, dominant class)."""
    comps = []
    for lo, hi, brk, wrong in cells:
        for wlo, whi, cls, m in wrong:
            if comps and abs(comps[-1]["hi"] - wlo) < 1e-12:
                c = comps[-1]; c["hi"] = whi; c["mass"] += m; c["cm"][cls] = c["cm"].get(cls, 0) + m
            else:
                comps.append(dict(lo=wlo, hi=whi, mass=m, cm={cls: m}))
    for c in comps:
        c["cls"] = max(c["cm"], key=c["cm"].get)
    return comps


def label_point(model, y, a, b, p1, tau, kind):
    T = len(y)
    cells = walk_one(model, y, a, b, tau, kind)
    comps = components(cells)
    # gate flips (cell ends) strictly inside a component
    breaks = {round(hi, 12): brk for lo, hi, brk, _ in cells if brk is not None}
    for c in comps:
        c["flips"] = [breaks[k] for k in sorted(breaks) if c["lo"] + 1e-12 < k < c["hi"] - 1e-12]
    F = sum(c["mass"] for c in comps)
    if not comps:
        return F, None
    c = max(comps, key=lambda c: c["mass"])
    u = y[a]
    if kind == "23":
        partner = b if c["lo"] >= 0 else a       # t>0 target a: wrong=b is the tie flip
        wt = "tie" if c["cls"] == partner else "max" if c["cls"] == p1 else "other"
        third = None
    else:
        lowmask = np.ones(T, bool); lowmask[[a, b]] = False
        third = int(np.flatnonzero(lowmask)[np.argmax(y[lowmask])]) if lowmask.any() else -1
        partner = a if c["lo"] >= 0 else b
        wt = "tie" if c["cls"] == partner else "third" if c["cls"] == third else "other"
    ub = 0 if u <= 0 else 1 if u <= 0.25 else 2 if u <= 0.5 else 3
    side = "+" if c["lo"] >= 0 else "-"
    return F, dict(kind=kind, a=int(a), b=int(b), p1=int(p1) if kind == "23" else int(third),
                   ub=ub, wt=wt, side=side, flips=[list(map(int, f)) for f in c["flips"][:3]],
                   nflips=len(c["flips"]), width=float(c["hi"] - c["lo"]), ncomp=len(comps),
                   npos=int((y > 0).sum()), u=float(u))


def run(name, logn=14, rules=(700, 701), tail_frac=0.05):
    model = get_model(name)
    T = model[2].shape[0]
    _, Z23, Z12 = C.tables(T)
    n = 2 ** logn
    out = dict(model=name, T=T, n=n, rules={})
    for rs in rules:
        G, meta = [], []
        for kind, Z in (("23", Z23), ("12", Z12)):
            U = C.sobol(T, n, rs + (0 if kind == "23" else 7))
            cfg = C.configs(T, kind)
            for i in range(0, n, 4096):
                Ui = U[i:i + 4096]
                y, a, b, tau = C.face_points(T, kind, Ui)
                ci = np.minimum((Ui[:, 0] * len(cfg)).astype(int), len(cfg) - 1)
                p1 = cfg[ci, 2]
                m, _, _ = C.walk_mass(model, y, a, b, tau, kind)
                G.append(Z * m)
                meta += [(kind, y[j], a[j], b[j], p1[j], tau[j]) for j in range(len(m))]
        g = np.concatenate(G); g2 = g ** 2
        order = np.argsort(-g2); ntail = int(np.ceil(tail_frac * len(g)))
        tail = []
        for j in order[:ntail]:
            kind, y, a, b, p1, tau = meta[j]
            F, lab = label_point(model, y, a, b, p1, tau, kind)
            Z = Z23 if kind == "23" else Z12
            lab = lab or {}
            lab.update(g2=float(g2[j]), Fcheck=float(abs(Z * F - g[j]) / max(g[j], 1e-300)))
            tail.append(lab)
        s = np.sort(g2)[::-1]
        out["rules"][str(rs)] = dict(
            err=float(g.mean()), err_note="mean g over 2n points; error estimate = 2*err", sum_g2=float(g2.sum()), tail_g2=float(s[:ntail].sum()),
            top1pct_share=float(s[:int(0.01 * len(g))].sum() / g2.sum()),
            top5pct_share=float(s[:ntail].sum() / g2.sum()), nzero=int((g == 0).sum()), tail=tail)
        print(name, rs, "err_est", 2 * g.mean(), "top5%", s[:ntail].sum() / g2.sum(),
              "max Fcheck", max(t["Fcheck"] for t in tail), flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", f"{name}.json"), "w"))


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 14)
