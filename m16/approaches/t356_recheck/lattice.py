"""T357 step 2 (pre-registered in README): scheme L (rank-1 Korobov lattice quadrature of closure c in order-statistic
coordinates) vs a high-precision paired-MC E0 and a plain-MC control, on DEV3 dev3_s500..509. -> lattice.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from itertools import permutations
from scipy.special import ndtr, ndtri
from scipy.stats import norm
T, NL, NREF = 10, 3203, 2_000_000
mu = lambda v: -norm.pdf(v) / np.maximum(ndtr(v), 1e-300)
TRI = np.array(list(permutations(range(T), 3)))
FREE = np.array([[p for p in range(T) if p not in t] for t in TRI]); P4 = FREE[:, -1]


def p2_crit(a, N, d=5):
    """Korobov P2 figure of merit (model-free), smaller is better"""
    k = np.arange(N); z = np.array([pow(a, j, N) for j in range(d)])
    x = (np.outer(k, z) % N) / N; B2 = x * x - x + 1 / 6
    return float(np.prod(1 + 2 * np.pi ** 2 * B2, 1).mean() - 1)


A = min(range(2, NL // 2), key=lambda a: p2_crit(a, NL))
U = ((np.outer(np.arange(NL), [pow(A, j, NL) for j in range(5)]) % NL) + 0.5) / NL


def lattice_inputs(U):
    tri = np.minimum((U[:, 0] * 720).astype(int), 719)
    v1 = ndtri(U[:, 1] ** (1 / 10)); v2 = ndtri(ndtr(v1) * U[:, 2] ** (1 / 9)); v3 = ndtri(ndtr(v2) * U[:, 3] ** (1 / 8))
    v4 = ndtri(ndtr(v3) * U[:, 4] ** (1 / 7))
    X = np.repeat(mu(v4)[:, None], T, 1); n = np.arange(len(U)); t = TRI[tri]
    X[n, t[:, 0]] = v1; X[n, t[:, 1]] = v2; X[n, t[:, 2]] = v3; X[n, P4[tri]] = v4
    return X, t[:, 1]


import e0e1 as E  # noqa: E402  (closure() and ok(); module-level run guarded below)

if __name__ == "__main__":
    z = np.load(sys.argv[1]); names = [str(n) for n in z["names"]][:10]; XL, yL = lattice_inputs(U); out = dict(a=A, N=NL, models={})
    C = (NL * 5948 + NL * 4 * 40) / 5948
    for i, n in enumerate(names):
        m = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))
        L = float(E.ok(m, XL, yL).mean())
        acc = 0
        for c in range(4):
            X = np.random.default_rng(3570 + i * 10 + c).standard_normal((NREF // 4, T)); X0, y = E.closure(X, -np.sort(-X, 1)[:, 3]); acc += E.ok(m, X0, y).sum()
        E0 = acc / NREF
        X = np.random.default_rng(3571 + i).standard_normal((NL, T)); X0, y = E.closure(X, -np.sort(-X, 1)[:, 3]); MC = float(E.ok(m, X0, y).mean())
        out["models"][n] = dict(L=L, E0=E0, MC=MC); print(n, out["models"][n], flush=True)
    d = np.array([v["L"] - v["E0"] for v in out["models"].values()]); dm = np.array([v["MC"] - v["E0"] for v in out["models"].values()])
    out.update(C_fpe=C, rms_L_pp=100 * float(np.sqrt((d ** 2).mean())), mean_L_pp=100 * float(d.mean()), rms_MC_pp=100 * float(np.sqrt((dm ** 2).mean())),
               mean_MC_pp=100 * float(dm.mean()), E0_noise_pp=100 * float(np.sqrt(np.mean([v["E0"] * (1 - v["E0"]) for v in out["models"].values()]) / NREF)))
    out["overturn"] = bool(out["rms_L_pp"] <= 0.15 and abs(out["mean_L_pp"]) <= 0.1 and C <= 3300)
    print({k: v for k, v in out.items() if k != "models"}); json.dump(out, open(os.path.join(HERE, "lattice.json"), "w"), indent=1)
