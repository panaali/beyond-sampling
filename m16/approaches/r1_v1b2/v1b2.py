"""T313: R1W V1 with exact B2 first-cell prefix sharing and instrumented FLOPs (TASKS Directive 8 cost convention).
The estimate is computed by the SAME code path as frozen V1 (r1_walk/r1w2.est); only the cost is recounted here.

Convention (Directive 8): charge every weight-dependent operation; Phi at weight-dependent points = 20 FLOPs; model-free
precomputed tables free (GL nodes and weights, phi(v2 nodes), transverse points, the v4 Gauss node, Phi at
model-free points such as v2 + tmax and the blind "g12 assumed correct" profile); forwards at instrumented FLOPs.

Per walk step-evaluation (value + tangent + gate-flip roots), counted from walk.walk_line as implemented:
  z = h Wh^T + x Wi : 2H^2 + 2H | gate compare : H | h = where : H
  dz = dh Wh^T + d Wi : 2H^2 + 2H | dh = where : H
  roots r = tc - z/dz : 2H | r > tc and isfinite : 2H | min over neurons : H | update nxt/rstep : 2
Per cell readout + correct set: L = h Wo^T, dL = dh Wo^T : 2 x 2 C H | correct_interval ~ 14 C + 6 | cost bookkeeping 4
B2: in the FIRST cell of each line every row sits at the same t, so rows sharing an (input, tangent) prefix share the
step-evaluations up to that step; the first cell is charged by distinct prefixes. Later cells: per row, from the
restart step (the step whose gate flipped) to T-1, no sharing (rows are at different t)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("r1_walk", "r0_walk", "r0_integral"):
    sys.path.insert(0, os.path.join(HERE, "..", p))
import r0  # noqa: E402
import walk as W  # noqa: E402
import r1w  # noqa: E402
import r1w2  # noqa: E402

H, T, C = 16, 10, 10
F_STEP = (2 * H * H + 2 * H) + H + H + (2 * H * H + 2 * H) + H + 2 * H + 2 * H + H + 2      # = 1218
F_CELL = 2 * 2 * C * H + 14 * C + 6 + 4                                                       # = 790
F_PHI = 20


def distinct_prefix_steps(x, dx):
    """sum over steps s of #distinct (x[:, :s+1], dx[:, :s+1]) prefixes across rows."""
    tot = 0
    for s in range(T):
        tot += len({(tuple(np.round(x[i, :s + 1], 12)), tuple(np.round(dx[i, :s + 1], 12))) for i in range(len(x))})
    return tot


def walk_counted(model, x0, d, k, tmax):
    """Re-run walk.walk_line's loop only to count step-evaluations per cell (same arithmetic, same breakpoints)."""
    Wi, Wh, Wo = model; n = len(x0)
    t = np.zeros(n); restart = np.zeros(n, int); act = np.arange(n); it = 0; flops = 0.0; eps = 1e-10
    first_steps = distinct_prefix_steps(x0, d)
    while len(act) and it < 400:
        it += 1
        X0, D = x0[act], d[act]; tc = t[act] + eps * (1 + np.abs(t[act]))
        h = np.zeros((len(act), H)); dh = np.zeros_like(h); nxt = np.full(len(act), tmax); rstep = np.full(len(act), T)
        for s in range(T):
            z = h @ Wh.T + (X0[:, s] + tc * D[:, s])[:, None] * Wi[None]
            dz = dh @ Wh.T + D[:, s:s + 1] * Wi[None]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = tc[:, None] - z / dz
            r = np.where((r > tc[:, None]) & np.isfinite(r), r, np.inf).min(1)
            upd = r < nxt; nxt = np.where(upd, r, nxt); rstep = np.where(upd, s, rstep)
            g = z > 0; h = np.where(g, z, 0.0); dh = np.where(g, dz, 0.0)
        if it == 1:
            flops += first_steps * F_STEP                       # B2: shared prefixes in the first cell
        else:
            flops += (T - restart[act]).sum() * F_STEP           # restart from the flipping step
        flops += len(act) * F_CELL
        t[act] = nxt; restart[act] = rstep
        act = act[nxt < tmax]
    return flops


def phi_flops(pieces, nodes, kind, tmax):
    """Weight-dependent Phi evaluations in walk._profile: per interval-node, Phi(v2 +- a) and Phi(v2 +- b) unless the end
    is model-free (a = tmax tail start, b = infinity); '23' adds the ^8 (3 mults) per Phi; 2 adds per interval-node."""
    f = 0.0
    for rows, lo, hi in pieces:
        m = hi > lo; cnt = m.sum()
        if not cnt:
            continue
        a_free = np.isclose(lo[m], tmax); b_free = hi[m] >= W.INF - 1e-12
        nphi = (~a_free).sum() + (~b_free).sum()
        f += len(nodes) * (nphi * (F_PHI + (3 if kind == "23" else 0)) + 2 * cnt)
    return f


def est(model, budget=None):
    """Identical estimate to frozen V1 (r1w2.est with defaults); cost = instrumented FLOPs with B2."""
    acc, _ = r1w2.est(model)
    return acc, cost(model)


def cost(model):
    model = tuple(np.asarray(w, np.float64) for w in model)
    TRI = r1w.TRI; NT = len(TRI); k = TRI[:, 1]; flops = 0.0
    for name, lo, hi, (v2s, g12s, g23s) in r1w2.REG[False]:
        nodes, _ = W._gl(lo, hi)
        v3s = v2s - g23s; v4s = float(r1w.gauss_v4(v3s, 1)[0][0]); delta = v3s - v4s
        m4 = float(r0.mu(np.array([v4s]))[0]); dm4 = float(r1w.dmu(v4s)[0])
        tail = (np.arange(NT), np.full(NT, r1w2.TMAX), np.full(NT, W.INF))
        x, dx = r1w.inputs(v2s + g12s, v2s, v2s, v2s - delta, m4 + dm4 * (v2s - delta - v4s), 0.0, -1.0, -1.0, -dm4)
        flops += walk_counted(model, x, dx, k, r1w2.TMAX)
        p23, _ = W.walk_line(model, x, dx, k, r1w2.TMAX); flops += phi_flops(p23 + [tail], nodes, "23", r1w2.TMAX)
        if name.startswith("nb"):
            x, dx = r1w.inputs(v2s, v2s, v3s, v4s, m4, 1.0, 0.0, 0.0, 0.0)
            flops += walk_counted(model, x, dx, k, r1w2.TMAX)
            p12, _ = W.walk_line(model, x, dx, k, r1w2.TMAX); flops += phi_flops(p12 + [tail], nodes, "12", r1w2.TMAX)
        flops += NT * len(nodes) * 4                              # I12 * I23 * pdf(nodes), dot with GL weights
    flops += NT                                                    # mean over triples
    return float(flops)
