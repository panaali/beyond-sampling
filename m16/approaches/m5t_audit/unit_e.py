"""T417: unit audit of m5t_fast._e1 (the E_exp bound: series, scaling-squaring, cap, convexity, Jensen, trivial)
against flint's rigorous arb_mat.exp divided differences (ref.logE) on adversarial knot vectors.
usage: python unit_e.py N SEED  -> results/unit_e_<seed>.json"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "event_cone_rigour"))
sys.path.insert(0, HERE)
import m5t_fast as F  # noqa: E402
from flint import arb, ctx, fmpq  # noqa: E402

import ref  # noqa: E402


def fq(x):
    return fmpq(*float(x).as_integer_ratio())


def knots(rng):
    n = int(rng.integers(1, 11)); V = n + 1
    r = rng.random()
    if r < 0.25:
        w = 10 ** rng.uniform(-320, 5.5)
    elif r < 0.45:
        w = rng.uniform(20, 28)                      # series / scaling-squaring switch at 24
    elif r < 0.6:
        w = rng.uniform(4900, 5100)                  # WMAX = 5000 switch
    elif r < 0.7:
        w = rng.uniform(9500, 10500)                 # 2 WMAX (cap path)
    else:
        w = 10 ** rng.uniform(-3, 3)
    pat = int(rng.integers(6))
    if pat == 0:
        u = rng.random(V)
    elif pat == 1:                                   # clusters, intra-gaps tiny or exactly 0
        g = rng.integers(1, V + 1, V); cen = rng.random(V + 1)
        u = cen[g] + (rng.random(V) * 10 ** -rng.uniform(0, 17) if rng.random() < 0.6 else 0)
    elif pat == 2:                                   # one outlier
        u = np.zeros(V); u[rng.integers(V)] = 1.0
        u += rng.random(V) * 10 ** -rng.uniform(0, 17) * (rng.random() < 0.5)
    elif pat == 3:                                   # all equal but one tiny gap
        u = np.zeros(V); u[rng.integers(V)] = 10 ** -rng.uniform(0, 17)
    elif pat == 4:
        u = np.round(rng.random(V) * 4) / 4
    else:
        u = np.geomspace(1, 10 ** rng.uniform(1, 15), V); rng.shuffle(u)
    u = (u - u.min()) / max(u.max() - u.min(), 1e-300)
    off = [0.0, rng.uniform(-10, 10), rng.choice([-1, 1]) * 1e3, rng.choice([-1, 1]) * 10 ** rng.uniform(6, 9)][rng.integers(4)]
    t = off + w * u
    return np.ascontiguousarray(t, dtype=float)


def main():
    N, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = np.random.default_rng(4170 + seed)
    ctx.prec = 256
    s = np.empty(11); a = np.empty(11); b = np.empty(11)
    st = dict(cases=0, viol_up=0, viol_lo=0, nonfinite=0, max_up_excess=0.0, max_lo_gap=0.0, worst=[])
    t0 = time.time()
    while st["cases"] < N:
        t = knots(rng)
        if not np.all(np.isfinite(t)):
            continue
        st["cases"] += 1
        m = int(rng.integers(4))
        upper = m != 1
        cap = m in (0, 1)
        wcut = -1.0 if m == 3 else F.WMAX
        T, LF, w = F._e1(t, upper, F.REL_T, wcut, s, a, b, cap)
        if not (np.isfinite(T) and np.isfinite(LF)):
            st["nonfinite"] += 1; st["worst"].append(dict(t=t.tolist(), m=m, T=T, LF=LF)); continue
        L = ref.logE([arb(fq(x)) for x in t])
        e = -fq(T) + fq(LF)
        if upper and arb(e) < L.lower():
            st["viol_up"] += 1; st["worst"].append(dict(kind="up", t=t.tolist(), m=m, e=float(e), ref=float(L.mid())))
        if not upper and arb(e) > L.upper():
            st["viol_lo"] += 1; st["worst"].append(dict(kind="lo", t=t.tolist(), m=m, e=float(e), ref=float(L.mid())))
        gap = float((arb(e) - L).mid())
        if upper:
            st["max_up_excess"] = max(st["max_up_excess"], gap) if m in (0, 2) and w <= F.WMAX else st["max_up_excess"]
        else:
            st["max_lo_gap"] = max(st["max_lo_gap"], -gap) if w <= F.WMAX else st["max_lo_gap"]
    st["worst"] = st["worst"][:50]; st["elapsed"] = time.time() - t0
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(st, open(os.path.join(HERE, "results", f"unit_e_{seed}.json"), "w"), indent=0)
    print(json.dumps({k: v for k, v in st.items() if k != "worst"}))


if __name__ == "__main__":
    main()
