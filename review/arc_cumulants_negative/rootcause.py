"""T80 part 1: independent re-test of E35's root cause on random RNNs.

h_{t+1} = relu(Wh h_t + Wi x_t), x_t ~ N(0, I_d) fresh each step, Wh ~ N(0, 2/H), Wi ~ N(0, 2/d), linear read-out
y = Wo h_T, Wo ~ N(0, 1/H). d = 1 is the AlgZoo architecture. Metric: variance-normalised MSE of E[y] over output
units and seeds (S Monte Carlo samples give 1/S), and the mean |off-diagonal correlation| of h_T (MC).

Propagators: arc2 (ARC K=2 rule, own code), gauss (exact Gaussian closure, own code), arcK<k> (ARC's vendored library,
K=k, SIMPLE; only for H <= 64; driven by own loop).
Usage: python rootcause.py T H1,H2 d-list(e.g. 1,2,4,16,H) [N] [vendored K list, e.g. 2,3]"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cumprop import rnn_step_inject  # noqa: E402

T = int(sys.argv[1]); Hs = [int(h) for h in sys.argv[2].split(",")]; dspec = sys.argv[3].split(",")
N = int(sys.argv[4]) if len(sys.argv) > 4 else 1_000_000
VK = [int(k) for k in sys.argv[5].split(",")] if len(sys.argv) > 5 else []
USE_K3 = bool(VK)
SEEDS = 3

if USE_K3:
    ROOT = os.path.abspath(os.path.join(HERE, "../.."))
    for p in (os.path.join(ROOT, "data/arc_kprop_deps"), os.path.join(ROOT, "m16/approaches/arc_cumulants/vendor")):
        sys.path.insert(0, p)
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from mlp_kprop import kprop_harmonic as KH
    from mlp_kprop.wick import relu_wick_coef
    logging.getLogger("mlp_kprop").setLevel(logging.ERROR)
    torch.set_num_threads(1)

    def arc_vendor(Wi, Wh, Wo, K):
        H = Wh.shape[0]; DT = torch.float64
        sk = not KH.metric_needed(K, KH.SIMPLE)
        Kt = KH.coerce_input({1: torch.zeros(H, dtype=DT), 2: torch.zeros(H, H, dtype=DT)}, k_max=K, kind=KH.SIMPLE)
        Q = torch.as_tensor(Wi @ Wi.T, dtype=DT)
        for _ in range(T):
            Kt = KH.linear_kprop(Kt, torch.as_tensor(Wh, dtype=DT), k_max=K, set_metric=1.0 if sk else None)
            Kt[2] = Kt[2].clone(); Kt[2].core = Kt[2].core + Q
            Kt = KH.nonlin_kprop(Kt, nonlin_wick_coef=relu_wick_coef, k_max=K, kind=KH.SIMPLE)
        Kt = KH.linear_kprop(Kt, torch.as_tensor(Wo, dtype=DT), k_max=K, d_max=1, set_metric=1.0 if sk else None)
        return Kt[1].to_tensor().numpy()


def own(Wi, Wh, Wo, mode):
    H = Wh.shape[0]
    mu, C = np.zeros(H), np.zeros((H, H))
    Q = Wi @ Wi.T
    for _ in range(T):
        mu, C = rnn_step_inject(mu, C, Wh, Q, mode)
    return Wo @ mu


def mc(Wi, Wh, Wo, r):
    H, d = Wi.shape
    s1 = np.zeros(H); s2 = np.zeros(H); hs = np.zeros(H); hh = np.zeros((H, H))
    ch = 50_000 if H > 64 else 200_000
    for _ in range(N // ch):
        h = np.zeros((ch, H))
        for _t in range(T):
            h = np.maximum(h @ Wh.T + r.standard_normal((ch, d)) @ Wi.T, 0)
        y = h @ Wo.T
        s1 += y.sum(0); s2 += (y ** 2).sum(0); hs += h.sum(0); hh += h.T @ h
    m = s1 / N; v = s2 / N - m ** 2
    hm = hs / N; hc = hh / N - np.outer(hm, hm)
    sd = np.sqrt(np.maximum(np.diag(hc), 1e-30)); corr = hc / np.outer(sd, sd)
    off = np.abs(corr[~np.eye(H, dtype=bool)]).mean()
    return m, v, off


meths = ["arc2", "gauss"] + [f"arcK{k}" for k in VK]
print(f"T={T}  N={N}  seeds={SEEDS}  (MC floor ~{1/N:.0e})")
for H in Hs:
    ds = sorted({H if s == "H" else int(s) for s in dspec})
    for d in ds:
        res = {k: [] for k in meths}; offs = []
        for seed in range(SEEDS):
            r = np.random.default_rng(1000 * H + 10 * d + seed)
            Wi = r.standard_normal((H, d)) * math.sqrt(2 / d)
            Wh = r.standard_normal((H, H)) * math.sqrt(2 / H)
            Wo = r.standard_normal((H, H)) * math.sqrt(1 / H)
            m, v, off = mc(Wi, Wh, Wo, r); offs.append(off)
            for k in meths:
                e = arc_vendor(Wi, Wh, Wo, int(k[4:])) if k.startswith("arcK") else own(Wi, Wh, Wo, k)
                res[k].append(np.mean((e - m) ** 2 / v))
        print(f"H={H:4d} d={d:4d} mean|corr(h_T)|={np.mean(offs):.3f}  " + "  ".join(
            f"{k}: {np.mean(res[k]):.1e} (~{1/np.mean(res[k]):.0f})" for k in meths), flush=True)
