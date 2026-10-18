"""T272 part (2): integrated in-band-flip effect E[F] - E[F0] (paired on the same Sobol faces).
F0a = gates frozen at the face point (kinks.F0_frozen); F0c = exact walk, but each misclassified component (split at
t = 0) is replaced by the frozen-linear continuation of its entry cell (the end nearer t = 0), extended away from 0.
usage: python f0.py MODEL [n_a rules_a n_c rules_c]  -> results/f0_MODEL.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import ndtr

import core as C
import kinks as K
from tail import HERE, get_model

SQ2 = np.sqrt(2.0)


def cells_one(model, y, a, b, tau, kind, eps=1e-10, max_cells=4000):
    """Scalar exact walk: list of (lo, hi, tc, L, dL)."""
    Wi, Wh, Wo = model
    T = len(y); H = Wh.shape[0]
    d = np.zeros(T); d[a] = 1 / SQ2; d[b] = -1 / SQ2
    t = -tau; out = []
    while t < tau and len(out) < max_cells:
        tc = t + eps * (1 + abs(t))
        h = np.zeros(H); dh = np.zeros(H); nxt = tau
        for s in range(T):
            z = Wh @ h + (y[s] + tc * d[s]) * Wi
            dz = Wh @ dh + d[s] * Wi
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tc - z / dz
            r = r[(r > tc) & np.isfinite(r)]
            if len(r) and r.min() < nxt:
                nxt = r.min()
            g = z > 0
            h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
        out.append((t, nxt, tc, Wo @ h, Wo @ dh))
        t = nxt
    return out


def wrong_pieces(L, dL, tc, k, p_lo, p_hi):
    clo, chi, ok, _, _ = K.correct_set(L[None], dL[None], np.array([tc]), np.array([k]), np.array([p_lo]),
                                       np.array([p_hi]))
    clo, chi, ok = clo[0], chi[0], ok[0]
    if not ok:
        return [(p_lo, p_hi)]
    return [(lo, hi) for lo, hi in ((p_lo, clo), (chi, p_hi)) if hi > lo]


def F_and_F0c(model, y, a, b, tau, kind):
    tpos, tneg = (a, b) if kind == "23" else (b, a)
    cl = cells_one(model, y, a, b, tau, kind)
    F = 0.0; F0 = 0.0
    for side, k, edge in ((0, tneg, -tau), (1, tpos, tau)):
        # wrong intervals on this side, in order moving AWAY from t = 0
        segs = []
        for lo, hi, tc, L, dL in cl:
            p_lo, p_hi = (lo, min(hi, 0.0)) if side == 0 else (max(lo, 0.0), hi)
            if p_hi <= p_lo:
                continue
            for wlo, whi in wrong_pieces(L, dL, tc, k, p_lo, p_hi):
                segs.append((wlo, whi, tc, L, dL))
        segs.sort(key=lambda s_: -s_[1] if side == 0 else s_[0])
        # merge into components; entry = end nearer 0, entry cell = the first piece met from the 0 side
        comps = []
        for s_ in segs:
            near = s_[1] if side == 0 else s_[0]; far = s_[0] if side == 0 else s_[1]
            if comps and abs(comps[-1]["far"] - near) < 1e-12:
                comps[-1]["far"] = far; comps[-1]["mass"] += ndtr(s_[1]) - ndtr(s_[0])
            else:
                comps.append(dict(near=near, far=far, mass=ndtr(s_[1]) - ndtr(s_[0]), cell=s_[2:]))
        for c in comps:
            F += c["mass"]
            tc, L, dL = c["cell"]
            # frozen continuation from the entry point away from 0, within the window
            p_lo, p_hi = (edge, c["near"]) if side == 0 else (c["near"], edge)
            for wlo, whi in wrong_pieces(L, dL, tc, k, p_lo, p_hi):
                if (side == 0 and abs(whi - c["near"]) < 1e-12) or (side == 1 and abs(wlo - c["near"]) < 1e-12):
                    F0 += ndtr(whi) - ndtr(wlo)
    return F, F0


def main():
    name = sys.argv[1]
    na, ra, nc, rc = (int(v) for v in sys.argv[2:6]) if len(sys.argv) > 5 else (2 ** 14, 8, 2 ** 13, 4)
    model = get_model(name) if not name.startswith("z10s") else C.zoo(16, 10, seed=int(name[4:]))
    T = model[2].shape[0]; _, Z23, Z12 = C.tables(T); t0 = time.time()
    out = dict(model=name, Z=[Z23, Z12], a=[], c=[])
    for r in range(ra):                                     # F and F0a, vectorised
        e = {}
        for kind, Z in (("23", Z23), ("12", Z12)):
            U = C.sobol(T, na, 700 + r + (0 if kind == "23" else 7))
            Fs, F0s = [], []
            for i in range(0, na, 4096):
                y, a, b, tau = C.face_points(T, kind, U[i:i + 4096])
                Fs.append(C.walk_mass(model, y, a, b, tau, kind)[0]); F0s.append(K.F0_frozen(model, y, a, b, tau, kind))
            e[kind] = (Z * np.concatenate(Fs).mean(), Z * np.concatenate(F0s).mean())
        out["a"].append(dict(errF=e["23"][0] + e["12"][0], errF0a=e["23"][1] + e["12"][1]))
    print(name, "F0a done", round(time.time() - t0), flush=True)
    for r in range(rc):                                     # F and F0c, scalar
        e = {}
        for kind, Z in (("23", Z23), ("12", Z12)):
            U = C.sobol(T, nc, 800 + r + (0 if kind == "23" else 7))
            y, a, b, tau = C.face_points(T, kind, U)
            v = np.array([F_and_F0c(model, y[i], a[i], b[i], tau[i], kind) for i in range(nc)])
            if r == 0:                                      # scalar walk == vectorised walk
                out.setdefault("check_F_scalar_vs_vector", []).append(
                    float(np.abs(v[:, 0] - C.walk_mass(model, y, a, b, tau, kind)[0]).max()))
            e[kind] = (Z * v[:, 0].mean(), Z * v[:, 1].mean())
        out["c"].append(dict(errF=e["23"][0] + e["12"][0], errF0c=e["23"][1] + e["12"][1]))
        print(name, "F0c rule", r, out["c"][-1], round(time.time() - t0), flush=True)
    for key, f0 in (("a", "errF0a"), ("c", "errF0c")):
        d = np.array([x["errF"] - x[f0] for x in out[key]])
        out[f"bias_{key}_pp"] = float(100 * d.mean()); out[f"bias_{key}_se_pp"] = float(100 * d.std(ddof=1) / np.sqrt(len(d)))
        out[f"errF_{key}"] = float(np.mean([x["errF"] for x in out[key]]))
        out[f"errF0_{key}"] = float(np.mean([x[f0] for x in out[key]]))
    out["sec"] = time.time() - t0
    json.dump(out, open(os.path.join(HERE, "results", f"f0_{name}.json"), "w"), indent=1)
    print(name, {k: v for k, v in out.items() if k.startswith(("bias", "errF", "check"))}, flush=True)


if __name__ == "__main__":
    main()
