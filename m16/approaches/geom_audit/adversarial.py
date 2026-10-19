"""T420 phase C: adversarial polytopes for geom_fast (enumerate_fast, pulling_triangulation_fast, volumes_iv).
Each case: identity against rigour.Poly.enumerate / rigour.pulling_triangulation / q2iv_ref(exact volume), the
independent checks of indep.py, and the same under qhull sabotage (no proposals, half, noisy, junk) and with the
arb volume path forced. usage: python adversarial.py [seed] -> results_c_<seed>.jsonl"""
import itertools
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "event_cone_rigour")); sys.path.insert(0, HERE)
import geom_fast as G  # noqa: E402
import rigour as R  # noqa: E402
import indep  # noqa: E402
from flint import fmpq  # noqa: E402

P61 = 2 ** 61 - 1


def Q(x):
    if isinstance(x, fmpq):
        return x
    if isinstance(x, Fraction):
        return fmpq(x.numerator, x.denominator)
    if isinstance(x, float):
        return fmpq(*x.as_integer_ratio())
    return fmpq(x)


def poly(A, b):
    return [[Q(x) for x in r] for r in A], [Q(x) for x in b]


def box(n, lo=-1, hi=1):
    A, b = [], []
    for i in range(n):
        e = [0] * n; e[i] = 1; A.append(e); b.append(lo)
        e = [0] * n; e[i] = -1; A.append(e); b.append(-hi)
    return A, b


def h_of_points(P):
    """H-rep (A y >= b) of conv(P) by cddlib (exact); P rational points."""
    import cdd
    import cdd.gmp as cg
    mat = cg.matrix_from_array([[1] + [Fraction(x) for x in p] for p in P], rep_type=cdd.RepType.GENERATOR)
    h = cg.copy_inequalities(cg.polyhedron_from_matrix(mat))
    if h.lin_set:
        raise ValueError("not full-dimensional")
    return [[x for x in r[1:]] for r in h.array], [-r[0] for r in h.array]


def families(rng):
    """(label, A, b) cases."""
    out = []
    for n in range(2, 10):
        out.append(("cube", *box(n)))
    for n in range(2, 7):                                  # cross-polytope (n >= 7: the reference's
        # degenerate-edge search enumerates C(2^(n-1), n-1) row subsets per vertex; infeasible): every vertex has 2^(n-1) tight rows
        A = [list(s) for s in itertools.product([-1, 1], repeat=n)]
        out.append(("cross", [[-x for x in r] for r in A], [-1] * len(A)))
    for n in range(3, 10):                                 # pyramid over a cube: apex has 2(n-1) tight rows
        A = [[0] * (n - 1) + [1]]; b = [0]
        for i in range(n - 1):
            for s in (1, -1):
                r = [0] * n; r[i] = -s; r[-1] = -1; A.append(r); b.append(-1)
        out.append(("pyramid", A, b))
    for n in range(3, 7):                                  # random 0/1 polytopes (very degenerate; n = 7 stalls
        # in the reference's Poly.edges, which enumerates C(|T|, n - 1) row subsets at a degenerate vertex)
        for _ in range(3):
            pts = {tuple(int(x) for x in rng.integers(0, 2, n)) for _ in range(3 * n)}
            try:
                A, b = h_of_points(sorted(pts))
                out.append(("01poly", A, b))
            except ValueError:
                pass
    for n in range(3, 10):                                 # cube cut through vertices
        A, b = box(n, 0, 1); A.append([-1] * n); b.append(-(n - 2)); out.append(("cube_cut_vertices", A, b))
    for n in (3, 5, 7):                                    # near-parallel rows
        for e in (12, 20, 30):
            A, b = box(n)
            r = [Fraction(1)] * n; A.append([-x for x in r]); b.append(-Fraction(n - 1, 2))
            r2 = [Fraction(1)] * n; r2[0] += Fraction(1, 10 ** e)
            A.append([-x for x in r2]); b.append(-Fraction(n - 1, 2) - Fraction(1, 10 ** (e + 1)))
            out.append(("near_parallel_1e-%d" % e, A, b))
    for n in (3, 5, 7):                                    # tiny facets: corner cut at distance eps
        for e in (15, 25, 40):
            A, b = box(n); A.append([-1] * n); b.append(-(n - Fraction(1, 10 ** e))); out.append(("tiny_facet_1e-%d" % e, A, b))
    for n in (3, 5, 7):                                    # huge / mixed scales
        for sc in (20, 100):
            A, b = box(n)
            s = [Fraction(10) ** int(rng.integers(-sc, sc + 1)) for _ in range(n)]
            A = [[x / s[i] if x else 0 for i, x in enumerate(r)] for r in A]
            A.append([Fraction(-1) / s[i] for i in range(n)]); b.append(-Fraction(n - 1, 2))
            out.append(("mixed_scale_1e%d" % sc, A, b))
    for n in (3, 5, 7):                                    # big denominators
        A, b = box(n)
        for _ in range(4):
            A.append([Fraction(int(rng.integers(-1000, 1000)), 2 ** 200 + int(rng.integers(1, 99))) for _ in range(n)])
            b.append(-Fraction(sum(abs(x) for x in A[-1])) / 2)
        out.append(("big_denominators", A, b))
    for n in (3, 5, 7):                                    # modular attacks on 2^61 - 1
        A, b = box(n, 0, Fraction(1, P61)); A.append([-1] * n); b.append(-Fraction(n - 1, P61)); out.append(("den_p", A, b))
        A, b = box(n, 0, P61); A.append([-1] * n); b.append(-(n - 1) * P61); out.append(("diff_mult_p", A, b))
        A, b = box(n, 0, P61); A.append([-1] * n); b.append(-(n - 1) * P61)
        A.append([P61] + [1] * (n - 1)); b.append(0); out.append(("rows_mult_p", A, b))
    for n in range(3, 10):                                 # random polytopes, random rational normals
        for _ in range(2):
            A, b = box(n, -3, 3)
            for _ in range(2 * n + 6):
                a = [Fraction(int(x), int(rng.integers(1, 50))) for x in rng.integers(-20, 21, n)]
                if any(a):
                    A.append(a); b.append(-sum(abs(x) for x in a) * Fraction(int(rng.integers(30, 120)), 100))
            out.append(("random", A, b))
    return out


SABOTAGE = {
    "none": None,
    "no_qhull": lambda P, f: [],
    "half": lambda P, f: [y for j, y in enumerate(f) if j % 2],
    "noisy": lambda P, f: [y + np.random.default_rng(1).normal(0, 1e-3 * (1 + np.abs(y))) for y in f],
    "junk": lambda P, f: [np.random.default_rng(2).normal(0, 1, P.n) for _ in range(len(f))],
    "nonfinite": lambda P, f: [np.full(P.n, np.nan), np.full(P.n, np.inf)] + list(f),
}


def run_case(label, A, b):
    A, b = poly(A, b)
    Pl = R.Poly(A, b); n = Pl.n
    v0 = Pl.start()
    if v0 is None or v0 == "infeasible":
        W, _ = indep.cdd_vertices(A, b); v0 = list(W[0])
    res = dict(label=label, n=n, m=len(A))
    Vr, incr = Pl.enumerate(v0)
    Sr, dr = R.pulling_triangulation(Vr, incr, n)
    res.update(nV=len(Vr), nS=len(Sr), dim=dr)
    exact = indep.simplex_vols(indep.homog(Vr), Sr, n)
    ref_iv = [G.q2iv_ref(v) for v in exact]
    orig = G._float_vertices
    mism = {}
    for name, sab in SABOTAGE.items():
        if sab is not None:
            G._float_vertices = lambda P, sab=sab: sab(P, orig(P))
        st0 = dict(G.STATS)
        try:
            Vf, incf = G.enumerate_fast(Pl, v0)
        finally:
            G._float_vertices = orig
        Sf, df = G.pulling_triangulation_fast(Vf, incf, n, A)
        mm = [k for k, x, y in (("V", Vf, Vr), ("inc", incf, incr), ("S", Sf, Sr), ("dim", df, dr)) if x != y]
        if name == "none":
            Sf_noA, _ = G.pulling_triangulation_fast(Vf, incf, n, None)
            if Sf_noA != Sr:
                mm.append("S_noA")
            res["stats"] = {a: G.STATS[a] - st0[a] for a in st0}
        if mm:
            mism[name] = mm
    if dr == n and Sr:
        lo, hi = G.volumes_iv(Vr, Sr, n)
        if [(a, c) for a, c in zip(lo, hi)] != ref_iv:
            mism["vol_iv"] = True
        ec = G.EXACT_COST; G.EXACT_COST = -1                 # force the arb path
        st0 = G.STATS["vol_exact"]
        try:
            lo2, hi2 = G.volumes_iv(Vr, Sr, n)
        finally:
            G.EXACT_COST = ec
        res["arb_vol_exact_fallbacks"] = G.STATS["vol_exact"] - st0
        bad = sum(not (Q(float(l)) <= v <= Q(float(h))) for v, l, h in zip(exact, lo2, hi2))
        res["arb_path_not_q2iv_ref"] = sum((a, c) != r for a, c, r in zip(lo2, hi2, ref_iv))
        if bad:
            mism["arb_enclosure"] = bad
        balls = G.volumes_arb(Vr, Sr, n)
        if sum(not (G._dyadic(b_.lower()) <= v <= G._dyadic(b_.upper())) for b_, v in zip(balls, exact)):
            mism["volumes_arb"] = True
        ind = indep.check_cell(A, b, A, Vr, incr, Sr, lo, hi, n)
        res["indep"] = ind
        if not ind["ok"]:
            mism["indep"] = True
    res["mismatch"] = mism
    return res


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 420
    rng = np.random.default_rng(seed)
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0     # resume: skip the first `start` cases
    path = os.path.join(HERE, "results_c_%d.jsonl" % seed)
    with open(path, "a" if start else "w") as fh:
        for ci, (label, A, b) in enumerate(families(rng)):
            if ci < start:
                continue
            t = time.perf_counter()
            try:
                r = run_case(label, A, b)
            except Exception as e:  # noqa: BLE001
                import traceback
                r = dict(label=label, error=repr(e)[:300], tb=traceback.format_exc()[-800:])
            r["t"] = time.perf_counter() - t; r["case"] = ci
            fh.write(json.dumps(r, default=str) + "\n"); fh.flush()
            print(r["label"], r.get("n"), r.get("nV"), r.get("nS"), r.get("mismatch"), r.get("error", ""), "%.1fs" % r["t"], flush=True)


if __name__ == "__main__":
    main()
