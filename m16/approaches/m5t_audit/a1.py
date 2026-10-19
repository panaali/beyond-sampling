"""T417: assumption A1 of m5t_fast (libm / numpy / scipy accuracy) measured against mpmath (60 digits).
(1) Inputs actually used: scipy log_ndtr and erfcx are wrapped while bounds_batch runs on real and adversarial
    simplices (gen.py families), and every argument is recorded (subsampled) and checked.
(2) numba (libm) exp / log and numpy exp / log on sweeps covering the arguments the code can produce, with dense
    sampling near the subnormal range of exp.
(3) Environment checks the error analysis relies on: subnormals not flushed (FTZ/DAZ), no FMA contraction or SVML
    in the compiled numba functions.
usage: python a1.py NCASES SEED -> results/a1_<seed>.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "event_cone_rigour"))
sys.path.insert(0, HERE)
import m5t_fast as F  # noqa: E402
import mpmath as mp  # noqa: E402
import numba  # noqa: E402
import scipy.special as sp  # noqa: E402

import gen  # noqa: E402

mp.mp.dps = 60
REC = dict(log_ndtr=[], erfcx=[])


def rec(name, f):
    def g(x):
        a = np.asarray(x, float).ravel()
        REC[name].append(a[np.isfinite(a)])
        return f(x)
    return g


def lndtr_ref(x):
    return mp.log(mp.ncdf(mp.mpf(x)))


def erfcx_ref(x):
    x = mp.mpf(x)
    return mp.exp(x * x) * mp.erfc(x)


@numba.njit
def nexp(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = np.exp(x[i])
    return out


@numba.njit
def nlog(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = np.log(x[i])
    return out


@numba.njit
def nsub():
    return 2.0 ** -1074 + 0.0, (2.0 ** -1022) / 4.0


def main():
    N, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = np.random.default_rng(41700 + seed)
    F.log_ndtr = rec("log_ndtr", sp.log_ndtr); F.erfcx = rec("erfcx", sp.erfcx)
    seeds = gen.Seeds(); n = 0
    while n < N:
        c = gen.make(gen.FAMILIES[n % len(gen.FAMILIES)], rng, seeds)
        if c is None:
            continue
        n += 1
        P = F.Params(c["d"], c["mlow"], c["jp3"], c["t0"], c["zbins"], c["zratio"])
        try:
            F.bounds_batch(c["Mlo"][None], c["Mhi"][None], np.array([c["vlo"]]), np.array([c["vhi"]]), P, mc=c["mc"][None])
        except Exception:  # noqa: BLE001
            pass
    res = dict(cases=N)
    for name, ref_f, kind in (("log_ndtr", lndtr_ref, "abs1x2"), ("erfcx", erfcx_ref, "rel")):
        x = np.unique(np.concatenate(REC[name])) if REC[name] else np.zeros(0)
        res[name + "_inputs"] = dict(count=int(len(x)), min=float(x.min()) if len(x) else None, max=float(x.max()) if len(x) else None)
        sel = x if len(x) <= 40000 else np.r_[np.sort(x)[:2000], np.sort(x)[-2000:], rng.choice(x, 36000, replace=False)]
        f = sp.log_ndtr if name == "log_ndtr" else sp.erfcx
        v = f(sel); worst = 0.0; wx = None
        for xi, vi in zip(sel, v):
            r = ref_f(float(xi))
            err = float(abs(mp.mpf(float(vi)) - r))
            m = err / (1 + xi * xi) if kind == "abs1x2" else err / float(abs(r))
            if m > worst:
                worst, wx = m, float(xi)
        res[name] = dict(measured_worst=worst, at=wx, assumed=1e-13, margin=1e-13 / worst if worst else None, checked=int(len(sel)))
    # (2) exp / log sweeps
    xs = np.r_[rng.uniform(-745.2, 709.7, 200000), rng.uniform(-745.2, -700, 50000), rng.uniform(-50, 50, 50000)]
    for nm, f in (("exp_numba", nexp), ("exp_numpy", np.exp)):
        v = f(xs); worst = 0.0; wx = None
        for xi, vi in zip(xs[::5], v[::5]):
            r = mp.exp(mp.mpf(float(xi)))
            err = max(0.0, float(abs(mp.mpf(float(vi)) - r)) - F.TINY)
            m = float(mp.mpf(err) / r)
            if m > worst:
                worst, wx = m, float(xi)
        res[nm] = dict(measured_worst_rel_beyond_tiny=worst, at=wx, assumed=1e-13, margin=1e-13 / worst if worst else None, checked=int(len(xs[::5])))
    ys = np.r_[np.exp(rng.uniform(-745, 709, 100000)), 1 + 10 ** rng.uniform(-16, 0, 20000)]
    for nm, f in (("log_numba", nlog), ("log_numpy", np.log)):
        v = f(ys); worst = 0.0; wx = None
        for yi, vi in zip(ys[::3], v[::3]):
            r = mp.log(mp.mpf(float(yi)))
            m = float(abs(mp.mpf(float(vi)) - r) / abs(r)) if r != 0 else 0.0
            if m > worst:
                worst, wx = m, float(yi)
        res[nm] = dict(measured_worst_rel=worst, at=wx, assumed=1e-13, margin=1e-13 / worst if worst else None, checked=int(len(ys[::3])))
    # (3) environment
    a, b = nsub()
    res["subnormals_numba"] = bool(a > 0 and b > 0)
    res["subnormals_numpy"] = bool(np.float64(2.0 ** -1074) + 0.0 > 0)
    ir = ""
    for fn in (F._e1, F._emat, F._prep2, F._bins, F._bin_terms):
        for sig in fn.signatures:
            ir += fn.inspect_llvm(sig)
    res["llvm_fma"] = ir.count("llvm.fma") + ir.count("fmuladd")
    res["llvm_contract_flags"] = ir.count(" contract ")
    res["llvm_fast_flags"] = ir.count(" fast ")
    res["svml"] = ir.count("__svml")
    res["numba_config"] = dict(fastmath_default=False, svml_enabled=bool(getattr(numba.config, "USING_SVML", False)))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", f"a1_{seed}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
