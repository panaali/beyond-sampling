"""T324 addendum A (DIAGNOSTIC SAMPLING, paired): combination rules for the 2-D residue on 10 DEV2 models.
usage: python addendum.py [N=1000000]"""
import json, os, sys
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm
NN = int(float(sys.argv[1])) if len(sys.argv) > 1 else 1_000_000
sys.argv = sys.argv[:1]
import budget as B  # noqa: E402
from budget import Point, verdict, mu, dmu, node, dev2, forward, T  # noqa: E402

# exact-law region / half-region points (cont_d.exact_points), halves split at the exact median of v2 in the region
v = np.linspace(-6.5, 6.5, 2_600_001); f = 90 * norm.pdf(v) * ndtr(-v) * ndtr(v) ** 8
cf = np.concatenate([[0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(v))])
def median(lo, hi):
    a, b = np.interp([lo, hi], v, cf); return float(np.interp((a + b) / 2, cf, v))
MED = {"nb": median(0.0, 6.5), "b": median(-6.5, 0.0)}
HALF = {"nb1": (0.0, MED["nb"]), "nb2": (MED["nb"], 6.5), "b1": (-6.5, MED["b"]), "b2": (MED["b"], 0.0)}
PT1 = {k: B.PTS[k] for k in ("nb", "b")}
PT2 = {k: B.cont_d.exact_points(lo, hi) for k, (lo, hi) in HALF.items()}


def anchor_from(P, pts, half):
    if half:
        key = np.where(P.b, np.where(P.v2 <= MED["b"], "b1", "b2"), np.where(P.v2 <= MED["nb"], "nb1", "nb2"))
    else:
        key = np.where(P.b, "b", "nb")
    arr = np.array([pts[k] for k in key])                      # (n,3)
    v2a, g12s, g23s = arr.T; v3a = v2a - g23s; v4a = node(v3a)
    return v2a, g12s, g23s, v3a, v4a


def lines(model, P, A, exact=False):
    v2a, g12s, g23s, v3a, v4a = A
    v3 = v2a - P.g23
    if exact:
        v4 = node(v3); low = mu(v4)
    else:
        v4 = v3 - (v3a - v4a); low = mu(v4a) + dmu(v4a) * (v4 - v4a)
    s23 = verdict(model, P, v2a + g12s, v2a, v3, v4, low)
    s12 = verdict(model, P, v2a + P.g12, v2a, v3a, v4a, mu(v4a))
    J = verdict(model, P, v2a + P.g12, v2a, v3, v4, low)
    return s23, s12, J


if __name__ == "__main__":
    tr = json.load(open(os.path.join(B.ROOT, "data/oracle_floor/dev2_truths.json")))["truths"]
    out = {"points1": PT1, "points2": PT2, "median": MED}; models = dev2()
    for idx in range(0, 80, 8):
        name, model = models[idx]; rng = np.random.default_rng(324500 + idx)
        acc = {}; cls = {}
        for c in range(NN // 250_000):
            x = rng.standard_normal((250_000, T)); P = Point(x)
            Tg = B.closure_c(model, P)
            s23, s12, J1 = lines(model, P, anchor_from(P, PT1, False))
            _, _, J1x = lines(model, P, anchor_from(P, PT1, False), exact=True)
            t23, t12, J2 = lines(model, P, anchor_from(P, PT2, True))
            R = {"P1": s23 * s12, "OR": np.maximum(s23, s12), "L23": s23, "L12": s12, "AVG": (s23 + s12) / 2,
                 "P2": t23 * t12, "OR2": np.maximum(t23, t12), "J1": J1, "J1_exact": J1x, "J2": J2, "T": Tg}
            tri = P.tri[:, 0] * 100 + P.tri[:, 1] * 10 + P.tri[:, 2]
            for k, r in R.items():
                acc[k] = acc.get(k, 0.0) + r.sum()
                cls[k] = cls.get(k, 0) + np.bincount(tri, weights=r - Tg, minlength=1000)
            cls["_n"] = cls.get("_n", 0) + np.bincount(tri, minlength=1000)
        n = cls["_n"]; valid = n > 0
        out[name] = dict(truth=tr[name]["acc"], **{k: float(a / NN) for k, a in acc.items()},
                         class_res_rms_pp={k: float(100 * np.sqrt(np.mean((cls[k][valid] / n[valid]) ** 2))) for k in acc},
                         class_n_mean=float(n[valid].mean()))
        print(idx, name, {k: round(100 * (out[name][k] - out[name]["T"]), 3) for k in acc}, flush=True)
        json.dump(out, open(os.path.join(B.HERE, "addendum.json"), "w"), indent=1)
    names = [k for k in out if k not in ("points1", "points2", "median")]
    S = {}
    for k in out[names[0]]["class_res_rms_pp"]:
        res = np.array([100 * (out[n][k] - out[n]["T"]) for n in names]); err = np.array([100 * (out[n][k] - out[n]["truth"]) for n in names])
        S[k] = dict(res_bias=float(res.mean()), res_rms=float(np.sqrt((res ** 2).mean())), vs_truth_bias=float(err.mean()), vs_truth_rms=float(np.sqrt((err ** 2).mean())),
                    class_res_rms=float(np.mean([out[n]["class_res_rms_pp"][k] for n in names])))
    base = S["P1"]["res_rms"]
    for k in S:
        S[k]["ratio_to_P1"] = S[k]["res_rms"] / base if base > 0 else None
    zero = ("OR", "L23", "L12", "AVG")
    S["verdict"] = "GO" if any(S[k]["ratio_to_P1"] <= .6 for k in zero) else ("STOP (only exact 2-D)" if min(S["J1"]["ratio_to_P1"], S["J2"]["ratio_to_P1"]) <= .6 else "STOP (nothing reaches .6)")
    json.dump(dict(per_model=out, summary=S), open(os.path.join(B.HERE, "addendum.json"), "w"), indent=1)
    for k, s in S.items():
        print(k, s if isinstance(s, str) else {a: round(b, 3) for a, b in s.items()})
