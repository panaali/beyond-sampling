"""T412 soundness / tightness / speed tests of m5t_fast against the arb M5t (rigour._bounds_tangent).
usage:
  python t412_test.py fns                 transcendental-function error models vs arb; E_exp vs arb on random knots
  python t412_test.py simplex t0          test 1 on data/t412_rec_<t0>.npz (t412_record.py) -> results_t412_simplex_<t0>.json
  python t412_test.py cell t0             test 2: mass_cell_fast on the bench cells -> results_t412_cell_<t0>.json
  python t412_test.py explain t0 i [n]    re-run arb on cell i with a tighter _E_exp (eps 1e-30, 300 bits) and the arb
                                          tangent centre: is fast hi >= arb hi (1 - 1e-12) then?
  python t412_test.py region idx t0 [budget]  time mass_cell_fast on data/regions_1-8-2.npz[idx] (event=True)
  python t412_test.py explainregion idx t0 n  the same for the first n simplices of a 10-D region (event=True)
  python t412_test.py calib               arb per-simplex time on cells 3 and 44 (t0 2.25) run alone vs as recorded
  python t412_test.py regionarb idx t0 zbins  arb mass_cell on the same region with per-simplex fast vs arb comparison"""
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import m5t_fast as F  # noqa: E402
import rigour as R  # noqa: E402
from flint import arb, ctx, fmpq  # noqa: E402

BENCH = "data/bench_182_k4.npz"


def cell_params(tri, k, t0):
    I = sorted(set(range(k)) | set(tri)); d = len(I)
    return F.Params(d, R.D - d, I.index(tri[2]), t0)


def gates_of(dat, i, k):
    g = np.full((10, 16), -1); g[:k] = np.unpackbits(np.frombuffer(dat["hist"][i], np.uint8)).reshape(k, 16)
    return g


def test_fns():
    from scipy.special import erfcx, log_ndtr
    rng = np.random.default_rng(1); ctx.prec = 200
    Q = lambda v: arb(fmpq(*float(v).as_integer_ratio()))  # noqa: E731
    out = {}
    xs = np.r_[rng.uniform(-60, 40, 20000), rng.uniform(-8, 8, 20000)]
    out["log_ndtr_abs_over_1+x2"] = max(float(abs(Q(v) - ((-Q(x)) / arb(2).sqrt()).erfc().log() + arb(2).log()).upper()) / (1 + x * x)
                                        for x, v in zip(xs, log_ndtr(xs)))
    xs = rng.uniform(0, 60, 20000)
    out["erfcx_rel"] = max(float((abs(Q(v) - (Q(x) ** 2).exp() * Q(x).erfc()) / ((Q(x) ** 2).exp() * Q(x).erfc())).upper()) for x, v in zip(xs, erfcx(xs)))
    xs = rng.uniform(-745, 700, 20000)                  # model: |err| <= REL_T exp(x) + 4 * 2^-1074 (subnormals)
    out["exp_rel_beyond_tiny"] = max(max(0.0, float((abs(Q(v) - Q(x).exp()) - arb(F.TINY)).upper())) / float(Q(x).exp().lower()) for x, v in zip(xs, np.exp(xs)))
    xs = np.exp(rng.uniform(-700, 700, 20000))
    out["log_rel"] = max(float((abs(Q(v) - Q(x).log()) / abs(Q(x).log())).upper()) for x, v in zip(xs, np.log(xs)))
    import numba

    @numba.njit
    def nexp(x):
        return np.exp(x)

    @numba.njit
    def nlog(x):
        return np.log(x)
    xs = rng.uniform(-745, 700, 20000)
    out["numba_exp_rel_beyond_tiny"] = max(max(0.0, float((abs(Q(nexp(x)) - Q(x).exp()) - arb(F.TINY)).upper())) / float(Q(x).exp().lower()) for x in xs)
    xs = np.exp(rng.uniform(-700, 700, 20000))
    out["numba_log_rel"] = max(float((abs(Q(nlog(x)) - Q(x).log()) / abs(Q(x).log())).upper()) for x in xs)
    out["assumed"] = dict(REL_T=F.REL_T, LNDTR_ABS=F.LNDTR_ABS)
    # E_exp: n! e[t] vs arb (Hermite-Genocchi, distinct knots, 600 bits)
    ctx.prec = 8000
    viol = 0; worst = 0.0; cnt = 0
    for n in (4, 9):
        for _ in range(3000):
            sc = 10 ** rng.uniform(-12, 3.7)
            t = rng.uniform(-5, 5) + sc * rng.random(n + 1)
            if rng.random() < 0.3:                       # clusters
                t[: (n + 1) // 2] = t[0] + sc * 1e-9 * rng.random((n + 1) // 2)
            Tu, Lu = F.E_exp(t[None], True); Tl, Ll = F.E_exp(t[None], False)
            T = [Q(v) for v in t]
            if len(set(t)) < len(t) or np.ptp(t) > F.WMAX:
                continue
            dd = [(-x).exp() for x in T]
            for lev in range(1, n + 1):
                dd = [(dd[i + 1] - dd[i]) / (T[i + lev] - T[i]) for i in range(len(dd) - 1)]
            E = dd[0] * math.factorial(n) * (-1) ** n
            if float(E.rad()) > 1e-20 * float(E.mid()):
                continue
            up_ = (Q(-Tu[0]) + Q(Lu[0])).exp(); lo_ = (Q(-Tl[0]) + Q(Ll[0])).exp()
            cnt += 1
            if up_ < E or lo_ > E:
                viol += 1
            worst = max(worst, float((up_ / E).upper()) - 1)
    out["E_exp"] = dict(cases=cnt, violations=viol, worst_rel_excess_upper=worst)
    print(json.dumps(out, indent=1))
    json.dump(out, open("results_t412_fns.json", "w"), indent=1)


def test_simplex(t0):
    rec = np.load(f"data/t412_rec_{t0}.npz")
    tri = tuple(int(v) for v in rec["tri"]); k = int(rec["k"]); P = cell_params(tri, k, t0)
    cell = rec["cell"]
    m = np.nonzero(cell == cell[0])[0]                                                     # compile + warm up
    F.bounds_batch(rec["Mlo"][m], rec["Mhi"][m], rec["vlo"][m], rec["vhi"][m], P)
    l0, h0 = F._bounds_batch_np(rec["Mlo"][m[:500]], rec["Mhi"][m[:500]], rec["vlo"][m[:500]], rec["vhi"][m[:500]], P)
    l1, h1 = F.bounds_batch(rec["Mlo"][m[:500]], rec["Mhi"][m[:500]], rec["vlo"][m[:500]], rec["vhi"][m[:500]], P)
    port = float(np.max(np.abs(h1 / h0 - 1)))                                                # numba port vs numpy reference
    lo = np.empty(len(cell)); hi = np.empty(len(cell)); per = []
    for i in np.unique(cell):
        m = np.nonzero(cell == i)[0]
        t = time.perf_counter()
        lo[m], hi[m] = F.bounds_batch(rec["Mlo"][m], rec["Mhi"][m], rec["vlo"][m], rec["vhi"][m], P)
        tf = time.perf_counter() - t
        per.append(dict(i=int(i), n=len(m), t_fast=tf, t_arb=float(rec["t"][m].sum())))
    alo, ahi = rec["alo"], rec["ahi"]
    r = hi / ahi
    res = dict(t0=t0, simplices=len(cell), cells=len(per), port_max_rel_diff_hi=port,
               viol_hi_lt_arb_lo=int((hi < alo).sum()), viol_lo_gt_arb_hi=int((lo > ahi).sum()),
               nonfinite=int((~np.isfinite(hi)).sum()),
               frac_hi_within_1p01=float((hi <= 1.01 * ahi).mean()),
               frac_hi_ge_arb_1m1e12=float((hi >= ahi * (1 - 1e-12)).mean()),
               n_hi_lt_arb_1m1e12=int((hi < ahi * (1 - 1e-12)).sum()),
               n_hi_lt_arb_1m1e6=int((hi < ahi * (1 - 1e-6)).sum()),
               hi_ratio_quantiles={q: float(np.quantile(r, q)) for q in (0, 0.001, 0.01, 0.5, 0.99, 0.999, 1)},
               lo_ratio_median=float(np.median(lo / np.maximum(rec["alo"], 1e-300))),
               t_arb_per_simplex=float(rec["t"].sum() / len(cell)), t_fast_per_simplex=sum(p["t_fast"] for p in per) / len(cell),
               speedup=float(rec["t"].sum() / sum(p["t_fast"] for p in per)),
               speedup_cell_median=float(np.median([p["t_arb"] / p["t_fast"] for p in per])),
               speedup_cell_min=float(min(p["t_arb"] / p["t_fast"] for p in per)))
    print(json.dumps(res, indent=1))
    json.dump(dict(summary=res, cells=per), open(f"results_t412_simplex_{t0}.json", "w"), indent=0)


def test_cell(t0, budget=2000):
    dat = np.load(BENCH, allow_pickle=True); tri = tuple(int(v) for v in dat["tri"]); k = int(dat["k"])
    ref = json.load(open(f"results_a_m5t_{t0}_2000.json"))
    out = []
    for o in ref:
        g = gates_of(dat, o["i"], k)
        t = time.perf_counter(); geo = F.cell_geometry(tri, g, k); tg = time.perf_counter() - t
        t = time.perf_counter(); lb, ub, info = F.mass_cell_fast(tri, g, k, t0, budget=budget, return_info=True, geom=geo)
        tb = time.perf_counter() - t
        se = o["mc"] / math.sqrt(o["hits"])
        out.append(dict(i=o["i"], lb=lb, ub=ub, mc=o["mc"], hits=o["hits"], se=se, ok=ub >= o["mc"] - 4 * se,
                        ub_arb=o["ub"], lb_arb=o["lb"], t_geom=tg, t_bounds=tb, t_arb=o["t"], simplices=info["simplices"]))
        print(out[-1], flush=True)
    s = dict(t0=t0, cells=len(out), violations=sum(not o["ok"] for o in out),
             ub_over_mc_median=float(np.median([o["ub"] / o["mc"] for o in out])),
             ub_over_ub_arb=[float(np.min([o["ub"] / o["ub_arb"] for o in out])), float(np.median([o["ub"] / o["ub_arb"] for o in out])), float(np.max([o["ub"] / o["ub_arb"] for o in out]))],
             t_cell_median=float(np.median([o["t_geom"] + o["t_bounds"] for o in out])), t_arb_cell_median=float(np.median([o["t_arb"] for o in out])))
    print(json.dumps(s, indent=1))
    json.dump(dict(summary=s, cells=out), open(f"results_t412_cell_{t0}.json", "w"), indent=0)


ORIG_E = R._E_exp


TIGHT_FALLBACK = [0]
ORIG_G = R._G_arb


def tight_G(d, s, a, b):
    """rigour._G_arb at 300 bits (its 53-bit gamma_upper ball has relative radius up to ~1e-8)."""
    old = ctx.prec; ctx.prec = 300
    try:
        return ORIG_G(d, s, a, b)
    finally:
        ctx.prec = old


def tight_on():
    R._E_exp = tight_E; R._G_arb = tight_G


def tight_off():
    R._E_exp = ORIG_E; R._G_arb = ORIG_G


def tight_E(tv, upper):
    """rigour._E_exp with knot separation 1e-30 (not 1e-9) and arithmetic at 300, 1500 or 5000 bits (the first that
    gives a ball of relative radius <= 1e-20); trivial fallback only if none does (counted in TIGHT_FALLBACK)."""
    old = ctx.prec
    try:
        n = len(tv) - 1
        pts = [x.lower() if upper else x.upper() for x in tv]
        lo_ = min(pts, key=float); hi_ = max(pts, key=float)
        trivial = (-lo_).exp() if upper else (-hi_).exp()
        if n == 0:
            return trivial
        for prec in (300, 1500, 5000):
            ctx.prec = prec
            eps = arb(fmpq(1, 10 ** 30)) * (1 + (hi_ - lo_))
            order = sorted(range(len(pts)), key=lambda j: float(pts[j]))
            T = [None] * len(pts)
            for r_, j in enumerate(order):
                T[r_] = pts[j] - r_ * eps if upper else pts[j] + r_ * eps
            dd = [(-x).exp() for x in T]
            for lev in range(1, n + 1):
                dd = [(dd[j + 1] - dd[j]) / (T[j + lev] - T[j]) for j in range(len(dd) - 1)]
            E = dd[0] * math.factorial(n) * (-1) ** n
            if E.is_finite() and E > 0 and arb(E.rad()) <= arb(E.mid()) * arb(fmpq(1, 10 ** 20)):
                return E.upper() if upper else E.lower()
        TIGHT_FALLBACK[0] += 1
        return trivial
    finally:
        ctx.prec = old


def test_explain(t0, i, nmax=400):
    """Exceptions hi_fast < hi_arb (1 - 1e-12): rerun arb on cell i, and per simplex compare fast hi (with the arb
    tangent centre) to arb hi computed with tight_E (knot separation 1e-30 instead of 1e-9, up to 5000 bits) and tight_G
    (300 bits)."""
    from scipy.optimize import nnls
    dat = np.load(BENCH, allow_pickle=True); tri = tuple(int(v) for v in dat["tri"]); k = int(dat["k"])
    P = cell_params(tri, k, t0); g = gates_of(dat, i, k)

    rows = []
    orig = R._bounds_tangent

    def wrapped(Ys, Ms, vol, d, zg, mlow, ip3, c0):
        lo, hi = orig(Ys, Ms, vol, d, zg, mlow, ip3, c0)
        if len(rows) < nmax:
            tight_on()
            try:
                lo2, hi2 = orig(Ys, Ms, vol, d, zg, mlow, ip3, c0)
            finally:
                tight_off()
            Pf = np.array([[float(v) for v in p] for p in Ms])
            beta, _ = nnls(np.vstack([Pf.T, 1e3 * np.ones(len(Ms))]), np.r_[np.zeros(Pf.shape[1]), 1e3])
            bq = [R.q(v) for v in beta]; tot = sum(bq); bq = [v / tot for v in bq]
            mc = np.array([float(sum(bq[j] * Ms[j][kk] for j in range(len(Ms)))) for kk in range(len(Ms[0]))])
            iv = [[F.q2iv(x) for x in m] for m in Ms]; vv = F.q2iv(vol)
            Ml = np.array([[[x[0] for x in m] for m in iv]]); Mh = np.array([[[x[1] for x in m] for m in iv]])
            fl, fh = F.bounds_batch(Ml, Mh, np.array([vv[0]]), np.array([vv[1]]), P, mc=mc[None])
            fl0, fh0 = F.bounds_batch(Ml, Mh, np.array([vv[0]]), np.array([vv[1]]), P)
            rows.append((float(hi.upper()), float(hi2.upper()), float(fh[0]), float(fh0[0])))
        return lo, hi
    R._bounds_tangent = wrapped
    R.mass_cell(tri, g, k, t0, budget=2000, tangent=True, zbins=24, zratio=fmpq(23, 20))
    R._bounds_tangent = orig
    a = np.array(rows)
    res = dict(t0=t0, cell=i, simplices=len(a), tight_fallbacks=TIGHT_FALLBACK[0],
               n_fast_lt_arb=int((a[:, 3] < a[:, 0] * (1 - 1e-12)).sum()),
               n_fastmc_lt_arb=int((a[:, 2] < a[:, 0] * (1 - 1e-12)).sum()),
               n_fastmc_lt_arbtight=int((a[:, 2] < a[:, 1] * (1 - 1e-12)).sum()),
               arbtight_over_arb_min=float((a[:, 1] / a[:, 0]).min()),
               fastmc_over_arbtight=[float((a[:, 2] / a[:, 1]).min()), float(np.median(a[:, 2] / a[:, 1])), float((a[:, 2] / a[:, 1]).max())],
               fast_over_fastmc=[float((a[:, 3] / a[:, 2]).min()), float((a[:, 3] / a[:, 2]).max())])
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"results_t412_explain_{t0}_{i}.json", "w"), indent=1)


def test_region(idx, t0, budget):
    dat = np.load("data/regions_1-8-2.npz"); tri = tuple(int(v) for v in dat["tri"]); g = dat["gates"][idx].astype(int)
    t = time.perf_counter(); geo = F.cell_geometry(tri, g, 10, event=True); tg = time.perf_counter() - t
    t = time.perf_counter(); lb, ub, info = F.mass_cell_fast(tri, g, 10, t0, budget=budget, event=True, return_info=True, geom=geo)
    tb = time.perf_counter() - t
    res = dict(region=idx, t0=t0, budget=budget, lb=lb, ub=ub, t_geom=tg, t_bounds=tb, **info)
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"results_t412_region_{idx}_{t0}_{budget}.json", "w"), indent=1)


def test_calib():
    dat = np.load(BENCH, allow_pickle=True); tri = tuple(int(v) for v in dat["tri"]); k = int(dat["k"])
    rec = np.load("data/t412_rec_2.25.npz"); orig = R._bounds_tangent; ts = []

    def w(*a):
        t = time.perf_counter(); r = orig(*a); ts.append(time.perf_counter() - t); return r
    R._bounds_tangent = w
    out = {}
    for i in (3, 44):
        ts.clear()
        R.mass_cell(tri, gates_of(dat, i, k), k, 2.25, budget=2000, tangent=True, zbins=24, zratio=fmpq(23, 20))
        out[i] = dict(alone=float(np.mean(ts)), recorded=float(rec["t"][rec["cell"] == i].mean()), n=len(ts))
    R._bounds_tangent = orig
    print(json.dumps(out))
    json.dump(out, open("results_t412_arb_timing_calibration.json", "w"))


def test_explain_region(idx, t0, nmax):
    """Region simplices: fast hi vs arb hi with the tightened _E_exp (first nmax root simplices, zbins 24)."""
    dat = np.load("data/regions_1-8-2.npz"); tri = tuple(int(v) for v in dat["tri"]); g = dat["gates"][idx].astype(int)
    P = F.Params(10, 0, sorted(set(range(10)) | set(tri)).index(tri[2]), t0)
    orig = R._bounds_tangent; rows = []

    def wrapped(Ys, Ms, vol, *a):
        if len(rows) >= nmax:
            raise StopIteration
        lo, hi = orig(Ys, Ms, vol, *a)
        tight_on()
        try:
            lo2, hi2 = orig(Ys, Ms, vol, *a)
        finally:
            tight_off()
        iv = [[F.q2iv(x) for x in m] for m in Ms]; vv = F.q2iv(vol)
        fl, fh = F.bounds_batch(np.array([[[x[0] for x in m] for m in iv]]), np.array([[[x[1] for x in m] for m in iv]]),
                                np.array([vv[0]]), np.array([vv[1]]), P)
        rows.append((float(hi.upper()), float(hi2.upper()), float(fh[0]), float(lo2.lower()), float(fl[0])))
        return lo, hi
    R._bounds_tangent = wrapped
    try:
        R.mass_cell(tri, g, 10, t0, tangent=True, event=True, zbins=24, zratio=fmpq(23, 20))
    except StopIteration:
        pass
    R._bounds_tangent = orig
    a = np.array(rows)
    res = dict(region=idx, simplices=len(a), tight_fallbacks=TIGHT_FALLBACK[0], n_fast_lt_arb=int((a[:, 2] < a[:, 0] * (1 - 1e-12)).sum()),
               n_fast_lt_arbtight=int((a[:, 2] < a[:, 1] * (1 - 1e-12)).sum()),
               n_fast_lo_gt_arbtight_hi=int((a[:, 4] > a[:, 1]).sum()), n_fast_hi_lt_arbtight_lo=int((a[:, 2] < a[:, 3]).sum()),
               arb_over_arbtight=[float(np.min(a[:, 0] / a[:, 1])), float(np.median(a[:, 0] / a[:, 1])), float(np.max(a[:, 0] / a[:, 1]))],
               fast_over_arbtight=[float(np.min(a[:, 2] / a[:, 1])), float(np.median(a[:, 2] / a[:, 1])), float(np.max(a[:, 2] / a[:, 1]))])
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"results_t412_explain_region_{idx}.json", "w"), indent=1)


def test_region_arb(idx, t0, zbins):
    zr = fmpq(23, 20) if zbins == 24 else fmpq(27, 25)
    dat = np.load("data/regions_1-8-2.npz"); tri = tuple(int(v) for v in dat["tri"]); g = dat["gates"][idx].astype(int)
    P = F.Params(10, 0, sorted(set(range(10)) | set(tri)).index(tri[2]), t0, zbins, zr)
    orig = R._bounds_tangent; rows = []; M = []

    def wrapped(Ys, Ms, vol, *a):
        t = time.perf_counter(); lo, hi = orig(Ys, Ms, vol, *a); t = time.perf_counter() - t
        iv = [[F.q2iv(x) for x in m] for m in Ms]; vv = F.q2iv(vol)
        M.append((np.array([[x[0] for x in m] for m in iv]), np.array([[x[1] for x in m] for m in iv]), vv))
        rows.append((float(lo.lower()), float(hi.upper()), t))
        return lo, hi
    R._bounds_tangent = wrapped
    t = time.perf_counter(); lb, ub = R.mass_cell(tri, g, 10, t0, tangent=True, event=True, zbins=zbins, zratio=zr); ta = time.perf_counter() - t
    R._bounds_tangent = orig
    a = np.array(rows)
    Ml = np.array([m[0] for m in M]); Mh = np.array([m[1] for m in M])
    vl = np.array([m[2][0] for m in M]); vh = np.array([m[2][1] for m in M])
    F.bounds_batch(Ml[:50], Mh[:50], vl[:50], vh[:50], P)
    t = time.perf_counter(); lo, hi = F.bounds_batch(Ml, Mh, vl, vh, P); tf = time.perf_counter() - t
    res = dict(region=idx, t0=t0, zbins=zbins, arb_lb=lb, arb_ub=ub, arb_t_total=ta, arb_t_bounds=float(a[:, 2].sum()), simplices=len(a),
               fast_t_bounds=tf, speedup_bounds=float(a[:, 2].sum() / tf), fast_sum_hi=float(F.up(math.fsum(hi))),
               viol_hi_lt_arb_lo=int((hi < a[:, 0]).sum()), viol_lo_gt_arb_hi=int((lo > a[:, 1]).sum()),
               frac_hi_within_1p01=float((hi <= 1.01 * a[:, 1]).mean()), hi_ratio_q=[float(np.quantile(hi / a[:, 1], q)) for q in (0, .01, .5, .99, 1)],
               arb_hi_share_of_simplices_with_ratio_lt_0p99=float(a[hi < 0.99 * a[:, 1], 1].sum() / a[:, 1].sum()))
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"results_t412_regionarb_{idx}_{t0}_{zbins}.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fns":
        test_fns()
    elif cmd == "simplex":
        test_simplex(float(sys.argv[2]))
    elif cmd == "cell":
        test_cell(float(sys.argv[2]))
    elif cmd == "explain":
        test_explain(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 400)
    elif cmd == "explainregion":
        test_explain_region(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "calib":
        test_calib()
    elif cmd == "regionarb":
        test_region_arb(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "region":
        test_region(int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 4000)
