"""Deterministic-seeded, best-first walk over ERROR cones; sum their Gaussian masses.
estimate = 1 - sum(enumerated error-cone masses)   (uncovered error mass = bias)."""
import sys, heapq, time
sys.path.insert(0, '/home/user/clean-neurons/m16'); sys.path.insert(0, '/home/user/clean-neurons/m16/approaches/cone_enum')
import numpy as np
from cones import *
import arc_eval as ae


def seeds(model, C):
    """Canonical rank-order inputs (expected order statistics) pushed across 2nd/3rd and 1st/2nd ties by
    bisection (as in est_probe) -> points where the model is wrong.  No random inputs."""
    out = []
    for p1 in range(T):
        for p2 in range(T):
            for p3 in range(T):
                if len({p1, p2, p3}) < 3: continue
                base = ae._canon((p1, p2, p3), 0); v2 = base[p2]
                for mv, wrong in [(lambda b, g: ae._set(b, p3, v2 * (1 - g)), p3),
                                  (lambda b, g: ae._set(b, p1, v2 / (1 - g)), p1)]:
                    g, nn = ae._bisect(model, base, mv, wrong); C.flops += nn * FWD
                    if g > 0: out.append((g, mv(base, g)))
    out.sort(key=lambda t: -t[0])        # wider flip thresholds first (heuristically larger cones)
    return [p for _, p in out]


def is_error(model, x, C):
    C.flops += FWD
    return forward_point(model, x) != int(np.argsort(x)[-2])


def walk(model, max_cones, C, log_every=None, max_seeds=None, max_fwd=np.inf):
    seen = set(); heap = []; masses = []; keys = []; cnt = 0
    S = seeds(model, C)
    if max_seeds: S = S[:max_seeds]
    for k, p in enumerate(S):
        heapq.heappush(heap, (-1.0, cnt, p)); cnt += 1        # seeds get top priority
    trace = []
    while heap and len(masses) < max_cones and C.fwd_equiv < max_fwd:
        pr, _, x = heapq.heappop(heap)
        key, A = cone_of(model, x, C)
        if key in seen: continue
        seen.add(key)
        if key[1] == key[3]: continue                       # not an error cone
        r = cheb(A, C)
        if r is None: continue
        c, rho = r
        m, rse, (Y, Lb, b, w, cm) = cone_mass_ahr(A, c, C, seed=len(masses))
        masses.append(m); keys.append(key); C.cones += 1
        # facet crossings: one per binding row, at its highest-weight IS point; priority = flux share
        tot = w.sum() + 1e-300
        for row in np.unique(b):
            sel = np.flatnonzero(b == row); j = sel[w[sel].argmax()]
            xs = Lb[j] * cm + Y[j]
            xo = xs - 1e-7 * np.linalg.norm(xs) * A[row]
            if is_error(model, xo, C):
                heapq.heappush(heap, (-m * w[sel].sum() / tot, cnt, xo)); cnt += 1
        if log_every and len(masses) % log_every == 0:
            trace.append((len(masses), sum(masses), C.fwd_equiv)); print(trace[-1], flush=True)
    walk.keys = keys
    return float(sum(masses)), masses, trace


if __name__ == "__main__":
    from common import Wi, Wh, Wo
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    C = Counter(); t0 = time.time()
    tot, masses, tr = walk((Wi, Wh, Wo), n, C, log_every=max(n // 10, 1))
    print('cones', len(masses), 'error mass covered', tot, 'of true 0.046821', 'fwd-equiv', C.fwd_equiv,
          'LPs', C.lps, 'time', time.time() - t0)
    np.save(f'masses_{n}.npy', np.array(masses))


def est_cone(model, budget, rng=None):
    """arc_eval interface. Deductive (no random model inputs; rng unused). budget and the returned
    'evals' are in forward-pass equivalents of FLOPs (cone construction + LPs + integrator + crossing checks)."""
    C = Counter(); tot, _, _ = walk(model, 10**9, C, max_fwd=budget)
    return 1 - tot, C.fwd_equiv
