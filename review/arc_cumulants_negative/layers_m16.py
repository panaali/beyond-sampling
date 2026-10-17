"""T80 part 1b: layer-by-layer error of propagated hidden-state moments on the trained M_16,T (base model), vs MC.
Diagnosis only (uses random inputs for the reference).

Propagators: arc2, gauss (own code); arcK3, arcK4 (ARC's vendored library, own driver; scalar input injected
exactly into kappa_2 as in E35's diag_width); rim(m,n,closure) = recent-input mixture (cumprop.rim_hidden).
Columns per layer: rms error of the mean / rms MC mean, rms error of the variances / rms MC variance,
rms error of the off-diagonal covariances / rms MC off-diagonal covariance.
Usage: python layers_m16.py T methods(comma; e.g. arc2,gauss,arcK3,rim:1:8:gauss)"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "m16"))
import leaderboard as LB  # noqa: E402
from cumprop import rnn_step_inject, rim_hidden  # noqa: E402

T = int(sys.argv[1]); METHODS = sys.argv[2].split(",")
H = 16
name, (Wi, Wh, Wo) = LB._base(H, T)
Wi2 = Wi[:, None]


def vendor_trace(K):
    for p in (os.path.join(ROOT, "data/arc_kprop_deps"), os.path.join(ROOT, "m16/approaches/arc_cumulants/vendor")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import logging, warnings, torch
    warnings.filterwarnings("ignore")
    from mlp_kprop import kprop_harmonic as KH
    from mlp_kprop.wick import relu_wick_coef
    logging.getLogger("mlp_kprop").setLevel(logging.ERROR)
    torch.set_num_threads(1)
    DT = torch.float64
    sk = not KH.metric_needed(K, KH.SIMPLE)
    Kt = KH.coerce_input({1: torch.zeros(H, dtype=DT), 2: torch.zeros(H, H, dtype=DT)}, k_max=K, kind=KH.SIMPLE)
    Q = torch.as_tensor(np.outer(Wi, Wi), dtype=DT)
    out = []
    for _ in range(T):
        Kt = KH.linear_kprop(Kt, torch.as_tensor(Wh, dtype=DT), k_max=K, set_metric=1.0 if sk else None)
        Kt[2] = Kt[2].clone(); Kt[2].core = Kt[2].core + Q
        Kt = KH.nonlin_kprop(Kt, nonlin_wick_coef=relu_wick_coef, k_max=K, kind=KH.SIMPLE)
        out.append((Kt[1].to_tensor().numpy().copy(), Kt[2].to_tensor().numpy().copy()))
    return out


def own_trace(mode):
    mu, C = np.zeros(H), np.zeros((H, H)); out = []
    for _ in range(T):
        mu, C = rnn_step_inject(mu, C, Wh, np.outer(Wi, Wi), mode)
        out.append((mu.copy(), C.copy()))
    return out


# MC reference
r = np.random.default_rng(0)
N, ch = 2_000_000, 250_000
S1 = [np.zeros(H) for _ in range(T)]; S2 = [np.zeros((H, H)) for _ in range(T)]
for _ in range(N // ch):
    h = np.zeros((ch, H)); x = r.standard_normal((ch, T))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
        S1[t] += h.sum(0); S2[t] += h.T @ h
ref = [(S1[t] / N, S2[t] / N - np.outer(S1[t] / N, S1[t] / N)) for t in range(T)]
off = ~np.eye(H, dtype=bool)
rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
print(f"model {name} H={H} T={T}; MC reference {N} draws. entries: relative rms error mean / var / off-diag cov")
for meth in METHODS:
    if meth.startswith("arcK"):
        tr = vendor_trace(int(meth[4:]))
    elif meth.startswith("rim"):
        _, m, n, cl = meth.split(":"); tr = []
        rim_hidden(Wi, Wh, T, int(m), int(n), cl, record=tr)
    else:
        tr = own_trace(meth)
    cells = []
    for t in range(T):
        (mu, C), (rm, rC) = tr[t], ref[t]
        cells.append(f"{rms(mu - rm) / rms(rm):.3f}/{rms(np.diag(C) - np.diag(rC)) / rms(np.diag(rC)):.3f}/"
                     f"{rms((C - rC)[off]) / rms(rC[off]):.3f}")
    print(f"{meth:18s} " + "  ".join(f"h{t+1}:{c}" for t, c in enumerate(cells)), flush=True)
