"""D2: per-face records (config, u, tau, cells, F, wrong-output rank) on iid faces, T=4 and 5, seeds 0-4."""
import sys
import numpy as np
from scipy.special import ndtr, ndtri
import core as C

N = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
K = 24   # points per segment for the wrong-output-rank attribution
for T in (4, 5):
    for s in range(5):
        m = C.zoo(16, T, s)
        rec = {}
        for kind in ("23", "12"):
            U = np.random.default_rng(7 + T * 10 + s + (0 if kind == "23" else 5)).random((N, T))
            cfg = C.configs(T, kind)
            ci = np.minimum((U[:, 0] * len(cfg)).astype(int), len(cfg) - 1)
            Fs, cs, ns, ys, taus, As, Bs = [], [], [], [], [], [], []
            for i in range(0, N, 8192):
                y, a, b, tau = C.face_points(T, kind, U[i:i + 8192])
                F, c, n = C.walk_mass(m, y, a, b, tau, kind)
                Fs.append(F); cs.append(c); ns.append(n); ys.append(y); taus.append(tau); As.append(a); Bs.append(b)
            F = np.concatenate(Fs); y = np.concatenate(ys); tau = np.concatenate(taus)
            a = np.concatenate(As); b = np.concatenate(Bs)
            # wrong-output rank on faces with F>0: K phi-quantile points in window
            wr = np.zeros((N, T + 1))
            idx = np.nonzero(F > 0)[0]
            if len(idx):
                q = (np.arange(K) + 0.5) / K
                lo, hi = ndtr(-tau[idx]), ndtr(tau[idx])
                t = ndtri(lo[:, None] + q[None] * (hi - lo)[:, None])          # (n,K)
                d = np.zeros((len(idx), T)); r = np.arange(len(idx))
                d[r, a[idx]] = 1 / C.SQ2; d[r, b[idx]] = -1 / C.SQ2
                X = y[idx][:, None, :] + t[..., None] * d[:, None, :]
                X = X.reshape(-1, T)
                pred = C.forward(m, X).argmax(1)
                rk = np.argsort(np.argsort(-X, 1), 1)[np.arange(len(X)), pred] + 1
                rk = rk.reshape(len(idx), K)
                for k in range(1, T + 1):
                    wr[idx, k] = (rk == k).mean(1)
            rec[kind] = dict(F=F, cost=np.concatenate(cs), cells=np.concatenate(ns), ci=ci, tau=tau,
                             u=y[np.arange(N), a], a=a, b=b, wr=wr, y=y)
        np.savez_compressed(f"../../../data/minimal_pair_d2_T{T}_s{s}.npz",
                            **{f"{k}_{f}": v for k, d in rec.items() for f, v in d.items()})
        _, Z23, Z12 = C.tables(T)
        print(T, s, "err", Z23 * rec["23"]["F"].mean() + Z12 * rec["12"]["F"].mean(), flush=True)
