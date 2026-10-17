"""Diagnosis (MC reference): does ARC's propagation work on *random* RNNs, and does that depend on the input width?
h_{t+1} = ReLU(Wh h_t + Wi x_t), x_t ~ N(0, I_d) fresh each step, Wh ~ N(0, 2/H), Wi ~ N(0, 2/d) (He-type), output
Wo h_T with Wo ~ N(0, 1/H). d = 1 is the AlgZoo architecture (scalar input); d = H is the wide-MLP-like analogue.
Input injection: kappa_2 += Wi Wi^T after each linear step (x_t is Gaussian and independent of h_t, so this is exact).
Reports variance-normalised MSE over output units and seeds (sampling with S samples has 1/S).
Usage: python diag_width.py T H1,H2,... K1,K2,..."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import arc_kprop as A
from mlp_kprop import kprop_harmonic as KH

T = int(sys.argv[1]); Hs = [int(h) for h in sys.argv[2].split(",")]; Ks = [int(k) for k in sys.argv[3].split(",")]
N = 1_000_000


def prop_rnn(Wi, Wh, Wo, K):
    H = Wh.shape[0]
    kind = KH.SIMPLE
    sk = not KH.metric_needed(K, kind)
    Kt = KH.coerce_input({1: torch.zeros(H, dtype=A.DT), 2: torch.zeros(H, H, dtype=A.DT)}, k_max=K, kind=kind)
    inj = torch.as_tensor(Wi @ Wi.T, dtype=A.DT)
    for t in range(T):
        Kt = KH.linear_kprop(Kt, torch.as_tensor(Wh, dtype=A.DT), k_max=K, set_metric=1.0 if sk else None)
        Kt[2] = Kt[2].clone(); Kt[2].core = Kt[2].core + inj
        Kt = KH.nonlin_kprop(Kt, nonlin_wick_coef=A.relu_wick_coef, k_max=K, kind=kind)
    Kt = KH.linear_kprop(Kt, torch.as_tensor(Wo, dtype=A.DT), k_max=K, d_max=1, set_metric=1.0 if sk else None)
    return Kt[1].to_tensor().numpy()


for H in Hs:
    for d in (1, H):
        res = {K: [] for K in Ks}
        for seed in range(3):
            r = np.random.default_rng(seed)
            Wi = r.standard_normal((H, d)) * math.sqrt(2 / d)
            Wh = r.standard_normal((H, H)) * math.sqrt(2 / H)
            Wo = r.standard_normal((H, H)) * math.sqrt(1 / H)
            s1 = np.zeros(H); s2 = np.zeros(H)
            for _ in range(N // 250_000):
                h = np.zeros((250_000, H))
                for t in range(T):
                    h = np.maximum(h @ Wh.T + r.standard_normal((250_000, d)) @ Wi.T, 0)
                y = h @ Wo.T; s1 += y.sum(0); s2 += (y ** 2).sum(0)
            mc = s1 / N; v = s2 / N - mc ** 2
            for K in Ks:
                e = prop_rnn(Wi, Wh, Wo, K)
                res[K].append(np.mean((e - mc) ** 2 / v))
        print(f"T={T} H={H:4d} input dim d={d:4d}: " + "  ".join(
            f"K={K}: vMSE={np.mean(res[K]):.1e} (~{1/np.mean(res[K]):.0f} samples)" for K in Ks) + f"   [MC floor {1/N:.0e}]", flush=True)
