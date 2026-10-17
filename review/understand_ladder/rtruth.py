"""Independent truth for T97 (review of E38). Written from scratch by the reviewer; shares no code with
m16/approaches/understand_ladder or arc_m43_repro.

* load(H, T, seed): zoo weights straight from alg_zoo (float64 copies) + the torch module itself.
* mc_torch(module, n, seed): accuracy of the actual torch module on n N(0, I_T) inputs (labels = alg_zoo task).
* exact(model): exact accuracy of a bias-free ReLU RNN with T = 3 by spherical-polygon enumeration:
  8 octants x 6 orderings -> spherical triangles; each recursively split by the sign of every pre-activation row
  of step 2 and 3 (step-1 pattern = sign(Wi * sign x0)); in each linear cone the correct set is the polygon
  clipped by L_truth - L_d > 0 (dead pattern -> logits 0 -> argmax 0). Solid angle by the Van Oosterom-Strackee
  formula over a fan. Returns (accuracy, n_regions = distinct (sign x0, g2, g3) patterns, n_leaves).
"""
import math

import numpy as np

E3 = np.eye(3)
PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def load(H, T, seed):
    import alg_zoo
    m = alg_zoo.zoo_2nd_argmax(H, T, seed=seed)
    sd = {k: v.detach().double().numpy().copy() for k, v in m.state_dict().items()}
    return (sd["rnn.weight_ih_l0"][:, 0], sd["rnn.weight_hh_l0"], sd["linear.weight"]), m


def mc_torch(module, n, seed, chunk=1_000_000):
    import torch
    from alg_zoo.tasks import task_2nd_argmax
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(seed)
    ok = 0; done = 0
    with torch.no_grad():
        while done < n:
            m = min(chunk, n - done)
            x = torch.randn(m, module.seq_len, generator=g, dtype=module.dtype)
            ok += int((module(x).argmax(-1) == task_2nd_argmax(x)).sum()); done += m
    return ok / n


def mc_np(model, n, seed, chunk=1_000_000):
    """Own float64 numpy forward pass (h_t = relu(Wh h_{t-1} + Wi x_t), logits = Wo h_T)."""
    Wi, Wh, Wo = model
    rng = np.random.default_rng(seed); ok = 0; done = 0
    while done < n:
        m = min(chunk, n - done)
        x = rng.standard_normal((m, Wo.shape[0]))
        h = np.zeros((m, len(Wi)))
        for t in range(x.shape[1]):
            h = np.maximum(h @ Wh.T + np.outer(x[:, t], Wi), 0.0)
        ok += int(((h @ Wo.T).argmax(1) == np.argsort(x, 1)[:, -2]).sum()); done += m
    return ok / n


def clip(poly, nrm):
    """Spherical convex polygon (list of 3-vectors, within a hemisphere) intersected with {nrm . x > 0}."""
    if not poly:
        return []
    d = [float(nrm @ v) for v in poly]
    if min(d) >= 0:
        return poly
    if max(d) <= 0:
        return []
    out = []
    k = len(poly)
    for i in range(k):
        a, b = poly[i], poly[(i + 1) % k]; da, db = d[i], d[(i + 1) % k]
        if da >= 0:
            out.append(a)
        if da * db < 0:
            v = a + (da / (da - db)) * (b - a)
            out.append(v / np.linalg.norm(v))
    return out if len(out) >= 3 else []


def tri(a, b, c):
    num = abs(float(a @ np.cross(b, c)))
    den = 1.0 + float(a @ b) + float(b @ c) + float(c @ a)      # unit vectors
    return 2.0 * math.atan2(num, den)


def area(poly):
    if len(poly) < 3:
        return 0.0
    return sum(tri(poly[0], poly[i], poly[i + 1]) for i in range(1, len(poly) - 1))


def split(polys, A):
    """Split each (poly, pattern-so-far) by the sign of every row of A (H x 3)."""
    out = []
    for poly in polys:
        cur = [(poly, ())]
        for r in A:
            nxt = []
            for p, g in cur:
                pp = clip(p, r); pm = clip(p, -r)
                if pp:
                    nxt.append((pp, g + (1,)))
                if pm:
                    nxt.append((pm, g + (0,)))
            cur = nxt
        out += cur
    return out


def octant_tris():
    out = []
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                v = [sx * E3[0], sy * E3[1], sz * E3[2]]
                # orient counter-clockwise seen from outside
                if float(v[0] @ np.cross(v[1], v[2])) < 0:
                    v = [v[0], v[2], v[1]]
                out.append((sx, v))
    return out


def exact(model):
    Wi, Wh, Wo = model
    tot = 0.0; pats = set(); leaves = 0; dead = 0.0
    for s, oct_ in octant_tris():
        g1 = ((Wi * s) > 0).astype(float)
        A2 = Wh @ np.outer(g1 * Wi, E3[0]) + np.outer(Wi, E3[1])
        for (hi, mid, lo) in PERMS:                       # x_hi > x_mid > x_lo, 2nd argmax = mid
            poly = clip(clip(oct_, E3[hi] - E3[mid]), E3[mid] - E3[lo])
            if not poly:
                continue
            for p2, g2 in split([poly], A2):
                A3 = Wh @ (np.array(g2, float)[:, None] * A2) + np.outer(Wi, E3[2])
                for p3, g3 in split([p2], A3):
                    leaves += 1; pats.add((s, g2, g3))
                    g = np.array(g3, float)
                    if g.sum() == 0:
                        if mid == 0:
                            tot += area(p3)
                        dead += area(p3)
                        continue
                    L = Wo @ (g[:, None] * A3)
                    q = p3
                    for d in range(3):
                        if d != mid:
                            q = clip(q, L[mid] - L[d])
                    tot += area(q)
    return tot / (4 * math.pi), len(pats), leaves
