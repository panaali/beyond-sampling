"""T361 S2 (DIAGNOSTIC SAMPLING): oracle floors of the homogeneous closure c_h (mean / spread lows) vs closure c (E1, E0)
on 30 DEV3 models, paired 2e5 exact-law inputs (seed 356000 + index, as floors_dev3.py)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.argv = sys.argv[:1]
import budget as B  # noqa: E402
D = os.path.join(HERE, "../../../../data/oracle_floor"); z = np.load(os.path.join(D, "dev3_weights.npz"))
tr = json.load(open(os.path.join(D, "dev3_truths.json")))["truths"]; ref = json.load(open(os.path.join(HERE, "floors_dev3.json")))["per_model"]
d, c, sr = 0.169777, 0.622476, 0.438411
PAR = (-1.0) ** np.arange(10)


def verdict_pos(m, P, v1, v2, v3, v4, lowmat):
    x = np.empty((P.n, 10)); r = np.arange(P.n)
    x[r, P.tri[:, 0]] = v1; x[r, P.tri[:, 1]] = v2; x[r, P.tri[:, 2]] = v3; x[r, P.p4] = v4
    for j in range(6):
        x[r, P.lows[:, j]] = lowmat(P.lows[:, j])
    return (B.forward(m, x).argmax(1) == P.tri[:, 1]).astype(float)


rows = {}
for k in range(0, 90, 3):
    nm = str(z["names"][k]); m = tuple(z[a][k].astype(np.float64) for a in ("Wi", "Wh", "Wo"))
    x = np.random.default_rng(356000 + k).standard_normal((B.N, 10)); P = B.Point(x)
    v1 = P.v2 + P.g12; v3 = P.v2 - P.g23; tv4 = -np.sort(-x, 1)[:, 3]
    E1 = B.closure_c(m, P).mean(); E0 = B.verdict(m, P, v1, P.v2, v3, tv4, B.mu(tv4)).mean()
    v4h = v3 - d * v1
    Hm = verdict_pos(m, P, v1, P.v2, v3, v4h, lambda pos: v4h - c * v1).mean()
    Hs = verdict_pos(m, P, v1, P.v2, v3, v4h, lambda pos: v4h - (c - (-1.0) * PAR[pos] * sr) * v1).mean()
    rows[nm] = dict(truth=tr[nm]["acc"], E1=float(E1), E0=float(E0), H_mean=float(Hm), H_spread=float(Hs),
                    chk_E0=float(abs(E0 - ref[nm]["E0"])), chk_E1=float(abs(E1 - ref[nm]["E1"])))
    print(nm, {kk: round(100 * (v - rows[nm]["truth"]), 3) for kk, v in rows[nm].items() if kk in ("E1", "E0", "H_mean", "H_spread")}, flush=True)
p = np.array([r["truth"] for r in rows.values()]); S = {"n": len(rows), "max_chk_E0_pp": 100 * max(r["chk_E0"] for r in rows.values()),
                                                       "max_chk_E1_pp": 100 * max(r["chk_E1"] for r in rows.values())}
for kk in ("E1", "E0", "H_mean", "H_spread"):
    e = 100 * (np.array([r[kk] for r in rows.values()]) - p); S[kk] = dict(bias=float(e.mean()), sd=float(e.std(ddof=1)), rms=float(np.sqrt((e ** 2).mean())))
fh = min(S["H_mean"]["rms"], S["H_spread"]["rms"]); S["verdict"] = "KILL" if fh > .45 else ("GO stage 2" if fh <= .40 else "inconclusive")
json.dump(dict(per_model=rows, summary=S), open(os.path.join(HERE, "s2.json"), "w"), indent=1); print(S)
