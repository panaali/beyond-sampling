"""T80: ceiling of E35's own read-outs. Feed ARC's propagation (vendored code via E35's arc_kprop helpers) the
*exact* (MC, large n) joint cumulants of the final state s_T = (h_T, x), orders 1..K, and run only the read-out
layers (Heaviside networks: 'uncond' = E35 uncond read-out; 'cell' = per-(p1,p2) cell read-out, with exact
cumulants of s_T | cell). If this misses, no improvement of the hidden-state propagation can rescue E35's
estimator: the failure is in expressing the 0/1 target through cumulants.
Usage: python readout_ceiling.py T K-list mode(uncond|cell) n [model_name]"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, os.path.join(ROOT, "m16")); sys.path.insert(0, os.path.join(ROOT, "m16/approaches/arc_cumulants"))
import leaderboard as LB  # noqa: E402
import arc_kprop as A  # noqa: E402  (E35's wrapper around ARC's vendored code; used read-only)

T = int(sys.argv[1]); KS = [int(k) for k in sys.argv[2].split(",")]; MODE = sys.argv[3]; N = int(sys.argv[4])
MNAME = sys.argv[5] if len(sys.argv) > 5 else None
H = 16


def moments_to_cumulants(S, order):
    """S: list of chunks (n_i, D) -> cumulants 1..order (central moments; kappa4 = m4 - 3 sym(C C))."""
    n = sum(len(s) for s in S)
    mu = sum(s.sum(0) for s in S) / n
    D = len(mu)
    acc = {2: 0.0, 3: 0.0, 4: 0.0}
    for s in S:
        c = s - mu
        acc[2] = acc[2] + c.T @ c
        if order >= 3:
            c2 = (c[:, :, None] * c[:, None, :]).reshape(len(c), -1)
            acc[3] = acc[3] + (c2.T @ c).reshape(D, D, D)
            if order >= 4:
                acc[4] = acc[4] + (c2.T @ c2).reshape(D, D, D, D)
    out = {1: mu, 2: acc[2] / n}
    if order >= 3:
        out[3] = acc[3] / n
    if order >= 4:
        C = out[2]
        out[4] = acc[4] / n - (np.einsum("ij,kl->ijkl", C, C) + np.einsum("ik,jl->ijkl", C, C)
                               + np.einsum("il,jk->ijkl", C, C))
    return out


def final_states(model, x):
    Wi, Wh, Wo = model
    h = np.zeros((len(x), H))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    return np.concatenate([h, x], 1)


def main():
    if MNAME is None:
        name, model = LB._base(H, T)
    elif MNAME.startswith("seed"):
        name, model = MNAME, LB._zoo(H, T, int(MNAME[4:]))
    else:
        name, model = MNAME, LB.models(H, T)[MNAME]
    import json
    d = json.load(open(os.path.join(ROOT, "results", "leaderboard_truths.json" if T == 10 else f"leaderboard_truths_H16_T{T}.json")))
    tr = d.get(name, {}).get("acc", float("nan"))
    r = np.random.default_rng(99)
    ch = 50_000 if T > 5 else 100_000
    kmax = max(KS)
    print(f"model {name} T={T} truth {tr:.5f}; mode {MODE}; exact cumulants of (h_T, x) from {N} draws", flush=True)
    if MODE == "uncond":
        S = [final_states(model, r.standard_normal((ch, T))) for _ in range(N // ch)]
        cum = moments_to_cumulants(S, min(kmax, 4))
        layers = A.readout_uncond(model)
        for K in KS:
            Kin = {k: cum[k] for k in cum if k <= A.KH.get_d_max(K, A.KH.SIMPLE)}
            est = float(A.propagate(layers, Kin, K)[0])
            print(f"  K={K}: estimate {est:.5f} ({100*(est-tr):+.2f} pp)", flush=True)
    else:
        res = {K: 0.0 for K in KS}
        for p1 in range(T):
            for p2 in range(T):
                if p1 == p2:
                    continue
                S = []
                for _ in range(N // ch):
                    x = -np.sort(-r.standard_normal((ch, T)), axis=1)
                    rest = x[:, 2:].copy(); r.permuted(rest, axis=1, out=rest)
                    others = [i for i in range(T) if i not in (p1, p2)]
                    y = np.empty_like(x); y[:, p1] = x[:, 0]; y[:, p2] = x[:, 1]; y[:, others] = rest
                    S.append(final_states(model, y))
                cum = moments_to_cumulants(S, min(kmax, 4))
                layers = A.readout_cell(model, p2)
                for K in KS:
                    Kin = {k: cum[k] for k in cum if k <= A.KH.get_d_max(K, A.KH.SIMPLE)}
                    res[K] += float(A.propagate(layers, Kin, K)[0]) / (T * (T - 1))
        for K in KS:
            print(f"  K={K}: estimate {res[K]:.5f} ({100*(res[K]-tr):+.2f} pp)", flush=True)


if __name__ == "__main__":
    main()
