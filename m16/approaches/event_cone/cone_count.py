"""T373: exact activation regions of the first k steps of orig inside a triple's TRUE event cone.
Derived from T344's cone_count.py (verbatim copy: cone_count_t344.py, sha256 0b87cd21...). Changes (README,
pre-registration):
 1. cone(): the homogeneous ordering cone O = {x_p1 >= x_p2 >= x_p3 >= every low} (no far-gap proxy), or, with
    affine=True, the affine K = O with x_p2 - x_p3 >= t0, for the equal-count check (rows G x >= b).
 2. Triples: pre-registered random ones (rng 373); model orig only (m16/common.py weights).
 3. Optional speed-up (fast=True): each cell carries a strictly interior point xi; the sign of c.xi tells which side
    of a neuron's hyperplane is non-empty, so only the other side's LP is solved. Decisions are the same as T344's
    two-LP rule (hi <= TOL -> off; lo >= -TOL -> on; else split); validated against fast=False for k <= 4.
 4. Cap on regions (= cells: histories are distinct by construction), as in T344.
 5. (post-launch, memory only) cells of the last step keep only their history; the event check uses the last
    COMPLETE step's region set.
usage: python cone_count.py KMAX i0,i1,...  (indices into TRIPLES) [--slow] [--affine BOX]"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.join(ROOT, "m16"))
TOL = 1e-9
D = 10
T0 = 1.5
TRIPLES = ((1, 2, 0), (9, 7, 8), (8, 9, 6), (2, 8, 7), (5, 2, 7), (2, 8, 5), (1, 7, 8), (1, 0, 8), (4, 1, 3), (0, 4, 2))


def model():
    import common
    return common.Wi, common.Wh, common.Wo


def cone(tri, affine=False):
    p1, p2, p3 = tri; lows = [i for i in range(D) if i not in tri]; e = np.eye(D)
    rows = [e[p1] - e[p2], e[p2] - e[p3]] + [e[p3] - e[l] for l in lows]
    b = np.zeros(len(rows))
    if affine:
        b[1] = T0                                                  # x_p2 - x_p3 >= t0
    return np.array(rows), b


def interior(tri):
    x = np.zeros(D); p1, p2, p3 = tri; lows = [i for i in range(D) if i not in tri]
    x[p1], x[p2], x[p3] = .9, .6, .3
    x[lows] = -np.linspace(.05, .9, len(lows))
    return x


def lp(c, G, b, box, sign):
    """sign=+1: max c.x; sign=-1: min c.x over {G x >= b, |x_i| <= box}. Returns value (nan if infeasible)."""
    r = linprog(-sign * c, A_ub=-G, b_ub=-b, bounds=[(-box, box)] * D, method="highs")
    return -sign * r.fun if r.status == 0 else np.nan, (r.x if r.status == 0 else None)


def split_plain(G, b, c, box):
    nc = np.linalg.norm(c)
    if nc == 0:
        return "off", None
    hi, _ = lp(c / nc, G, b, box, 1); lo, _ = lp(c / nc, G, b, box, -1)
    if hi <= TOL:
        return "off", None
    if lo >= -TOL:
        return "on", None
    return "split", None


def split_fast(G, xi, c):
    """Same decision as split_plain, using the interior point xi of {G x >= 0, |x| <= 1}. Returns
    (decision, interior point for the child that needs a new one)."""
    nc = np.linalg.norm(c)
    if nc == 0:
        return "off", None, None
    u = c / nc; s = u @ xi; z = np.zeros(len(G))
    if s > TOL:                                                    # hi > TOL for sure
        lo, xs = lp(u, G, z, 1.0, -1)
        if lo >= -TOL:
            return "on", None, None
        lam = (-lo / 2) / (s - lo)                                 # u.x = lo/2 < 0, strictly inside G
        return "split", xi, lam * xi + (1 - lam) * xs
    if s < -TOL:                                                   # lo < -TOL for sure
        hi, xs = lp(u, G, z, 1.0, 1)
        if hi <= TOL:
            return "off", None, None
        lam = (hi / 2) / (hi - s)
        return "split", lam * xi + (1 - lam) * xs, xi
    d, _ = split_plain(G, z, c, 1.0)
    if d != "split":
        return d, None, None
    _, xp = lp(u, G, z, 1.0, 1); _, xm = lp(u, G, z, 1.0, -1)       # |s| <= TOL: rebuild both children
    return "split", .5 * (xi + xp), .5 * (xi + xm)


def count(tri, kmax, maxregions=200_000, fast=True, affine_box=None, return_cells=False):
    Wi, Wh, _ = model(); H = len(Wi); res = []; last_complete = (0, set())
    G0, b0 = cone(tri, affine=affine_box is not None)
    box = affine_box or 1.0
    cells = [(G0, b0, interior(tri), np.zeros((H, D)), b"")]
    for t in range(kmax):
        t0 = time.time(); new = []; ok = True; nlp = 0
        for G, b, xi, A, hist in cells:
            Z = Wh @ A; Z[:, t] += Wi
            pieces = [(G, b, xi, [])]
            for j in range(H):
                nxt = []
                for Gp, bp, xp, g in pieces:
                    if fast:
                        d, xpos, xneg = split_fast(Gp, xp, Z[j])
                    else:
                        d, _ = split_plain(Gp, bp, Z[j], box); xpos = xneg = None
                    if d == "off":
                        nxt.append((Gp, bp, xp, g + [0]))
                    elif d == "on":
                        nxt.append((Gp, bp, xp, g + [1]))
                    else:
                        nxt.append((np.vstack([Gp, Z[j]]), np.append(bp, 0.), xpos, g + [1]))
                        nxt.append((np.vstack([Gp, -Z[j]]), np.append(bp, 0.), xneg, g + [0]))
                pieces = nxt
            for Gp, bp, xp, g in pieces:
                g = np.array(g, float); hb = hist + np.packbits(g > 0).tobytes()
                if t == kmax - 1:                                  # last step: keep only the history (memory)
                    new.append((None, None, None, None, hb))
                else:
                    new.append((Gp, bp, xp, g[:, None] * Z, hb))
            if len(new) > maxregions:
                ok = False; break
        if ok:
            last_complete = (t + 1, {c[4] for c in new})
        cells = new
        hs = {c[4] for c in cells}
        res.append(dict(k=t + 1, complete=ok, cells=len(cells), regions=len(hs), sec=round(time.time() - t0, 1)))
        print(tri, res[-1], flush=True)
        if not ok:
            break
    if return_cells:
        return res, cells
    return res, last_complete


def prefixes(x, k):
    """6-step (k-step) gate histories of inputs x, packed as in count()."""
    import common
    hs = common.run(x)[:, 1:k + 1] > 0                             # (N, k, H)
    return [b"".join(np.packbits(hs[n, t]).tobytes() for t in range(k)) for n in range(len(x))]


def event_check(tri, regions, k, n=2_000_000, seed=0, chunk=200_000):
    """Proposal samples in K; check every k-prefix is enumerated; count event (pick = p3) prefixes."""
    import common
    rng = np.random.default_rng(seed + 1000 * tri[0] + 100 * tri[1] + 10 * tri[2])
    p1, p2, p3 = tri; lows = [i for i in range(D) if i not in tri]
    miss = 0; ev = 0; evp = set(); allp = set()
    for _ in range(n // chunk):
        x = np.zeros((chunk, D)); x[:, lows] = rng.standard_normal((chunk, len(lows)))
        x[:, p3] = x[:, lows].max(1) + rng.exponential(1, chunk)
        x[:, p2] = x[:, p3] + T0 + rng.exponential(1, chunk)
        x[:, p1] = x[:, p2] + rng.exponential(1, chunk)
        hs = common.run(x)
        pick = (hs[:, -1] @ common.Wo.T).argmax(1)
        pre = hs[:, 1:k + 1] > 0
        packed = np.packbits(pre, axis=2).reshape(chunk, -1)
        keys = [r.tobytes() for r in packed]
        for kk, isev in zip(keys, pick == p3):
            if kk not in regions:
                miss += 1
            allp.add(kk)
            if isev:
                ev += 1; evp.add(kk)
    return dict(samples=n, missing_prefixes=miss, distinct_sample_prefixes=len(allp), event_samples=ev,
                event_prefixes=len(evp), event_feasible_frac_lb=len(evp) / max(len(regions), 1))


if __name__ == "__main__":
    kmax = int(sys.argv[1]); idx = [int(i) for i in sys.argv[2].split(",")]
    fast = "--slow" not in sys.argv
    abox = float(sys.argv[sys.argv.index("--affine") + 1]) if "--affine" in sys.argv else None
    tag = f"k{kmax}_{'fast' if fast else 'slow'}{'_aff' if abox else ''}_{sys.argv[2].replace(',', '-')}"
    out = {}; path = os.path.join(HERE, f"results_{tag}.json")
    for i in idx:
        tri = TRIPLES[i]; t1 = time.time()
        res, (klast, regs) = count(tri, kmax, fast=fast, affine_box=abox)
        out[str(tri)] = dict(counts=res, cpu_sec=round(time.time() - t1, 1))
        if "--events" in sys.argv:
            out[str(tri)]["events"] = ev = event_check(tri, regs, klast); ev["k"] = klast
            print(tri, ev, flush=True)
        json.dump(out, open(path, "w"), indent=1)
