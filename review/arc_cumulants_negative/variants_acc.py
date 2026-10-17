"""T80 part 3: accuracy estimates of the untried variants on M_16,T (bias vs truth; deterministic, no model calls on
random inputs). Variants (spec strings):
  joint:<closure>          (b) Gaussian closure (closure = gauss: exact pairwise ReLU moments as E13, or arc2) of the
                           joint state (h, x), unconditional; endpoint: P(correct) under one Gaussian for (l, x).
  cell:<closure>           (b) the same per (p1,p2) cell, x | cell entering with its exact (model-free) mean/cov;
                           endpoint: Gaussian orthant on the T-1 margins.
  arccell:<K>              (b)+(d) ARC's vendored propagation (SIMPLE, order K) of the RNN layers per cell (E35's cell
                           inputs), then margins' cumulants -> Gaussian / Edgeworth orthant endpoint (= T40's
                           argmax endpoint d/da E[max(l+a)] with the Edgeworth density).
  rim:<m>:<n>:<closure>    (a) recent-input mixture (structure vs randomness): the last m inputs exact on an n-node grid
                           (n<0: split half-line rule), older inputs as a Gaussian residual jointly with h; endpoint
                           per component: Gaussian over (l, x_old) with x_recent fixed. m >= T is plain quadrature.
Endpoint integrals use fixed scrambled-Sobol Gaussian points (deterministic; explicit densities only).
Cost: FLOPs counted analytically (see cost notes in README), reported in FPE.
Usage: python variants_acc.py T spec1,spec2,... model1,model2,...|base"""
import os
import sys

import numpy as np
from scipy.stats import norm, qmc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "m16"))
import leaderboard as LB  # noqa: E402
from cumprop import relu_layer, relu_layer_b, gh_nodes, rnn_step_joint  # noqa: E402
from edgeworth import endpoint  # noqa: E402

H = 16
_SOB = {}


def sobol_normal(k, m=17):
    if (k, m) not in _SOB:
        u = qmc.Sobol(k, scramble=True, seed=31).random_base2(m)
        _SOB[(k, m)] = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    return _SOB[(k, m)]


def gauss_correct(mu, C, T, m=17):
    """P(argmax l = position of 2nd-largest x) for (l, x) ~ N(mu, C) (2T-dim; may be degenerate)."""
    ev, U = np.linalg.eigh((C + C.T) / 2)
    L = U * np.sqrt(np.maximum(ev, 0))
    z = sobol_normal(2 * T, m) @ L.T + mu
    return float((z[:, :T].argmax(1) == np.argsort(z[:, T:], 1)[:, -2]).mean())


def cost_closure(H, D, closure):
    """FLOPs of one closure step on a state of dim D (analytic estimate): linear maps + ReLU moments."""
    lin = 2 * H * D * D + 2 * H * H * D
    if closure == "gauss":
        return lin + (H * (H - 1) // 2) * 80 * 30 + 50 * H
    return lin + 4 * H * H + 50 * H


# ----------------------------------------------------------------- (b) joint and per-cell Gaussian closure
def est_joint(model, closure):
    Wi, Wh, Wo = model; T = Wo.shape[0]; D = H + T
    mu = np.zeros(D); C = np.zeros((D, D)); C[H:, H:] = np.eye(T)
    for t in range(T):
        mu, C = rnn_step_joint(mu, C, Wh, Wi, t, H, closure)
    A = np.zeros((2 * T, D)); A[:T, :H] = Wo; A[T:, H:] = np.eye(T)
    return gauss_correct(A @ mu, A @ C @ A.T, T), T * cost_closure(H, D, closure) + 2 * (2 * T) * D * D


def cell_inputs(T):
    sys.path.insert(0, os.path.join(ROOT, "m16/approaches/arc_cumulants"))
    import arc_kprop as A
    return A, A.cell_base_cumulants(T, order=4)


def margins_matrix(Wo, p2):
    T = Wo.shape[0]
    return np.stack([Wo[p2] - Wo[i] for i in range(T) if i != p2])     # (T-1, H)


def est_cell(model, closure):
    Wi, Wh, Wo = model; T = Wo.shape[0]; D = H + T
    A, base = cell_inputs(T)
    tot, flops = 0.0, 0
    for p1 in range(T):
        for p2 in range(T):
            if p1 == p2:
                continue
            K = A.cell_input(H, T, p1, p2, 2, base)
            mu, C = K[1].copy(), K[2].copy()
            for t in range(T):
                mu, C = rnn_step_joint(mu, C, Wh, Wi, t, H, closure)
            M = margins_matrix(Wo, p2)
            e = endpoint(M @ mu[:H], M @ C[:H, :H] @ M.T, np.zeros((T - 1,) * 3), np.zeros((T - 1,) * 4))
            tot += e["gauss"] / (T * (T - 1))
            flops += T * cost_closure(H, D, closure)
    return tot, flops


def est_arccell(model, K):
    """ARC's propagation of the RNN layers per cell (vendored), margins cumulants -> Gaussian/Edgeworth endpoints."""
    Wi, Wh, Wo = model; T = Wo.shape[0]
    A, base = cell_inputs(T)
    import torch
    KH = A.KH
    tot = {"gauss": 0.0, "edge3": 0.0, "edge4": 0.0}; flops = 0
    layers = A.rnn_layers(model)
    for p1 in range(T):
        for p2 in range(T):
            if p1 == p2:
                continue
            Kin = A.cell_input(H, T, p1, p2, K, base)

            def run():
                Kt = KH.coerce_input({d: torch.as_tensor(v, dtype=A.DT) for d, v in Kin.items()}, k_max=K, kind=KH.SIMPLE)
                sk = not KH.metric_needed(K, KH.SIMPLE)
                for W, b, act in layers:
                    Kt = KH.linear_kprop(Kt, torch.as_tensor(W, dtype=A.DT), k_max=K, set_metric=1.0 if sk else None)
                    Kt = KH.nonlin_kprop(Kt, nonlin_wick_coef=A.mixed_wick(act), k_max=K, kind=KH.SIMPLE)
                return {d: Kt[d].to_tensor().numpy() for d in Kt if d <= 4}
            cum, raw, _ = A.counted(run)
            flops += raw
            M = margins_matrix(Wo, p2); k = T - 1
            mu = M @ cum[1][:H]; C = M @ cum[2][:H, :H] @ M.T
            k3 = np.einsum("ia,jb,kc,abc->ijk", M, M, M, cum[3][:H, :H, :H]) if K >= 3 else np.zeros((k,) * 3)
            k4 = (np.einsum("ia,jb,kc,ld,abcd->ijkl", M, M, M, M, cum[4][:H, :H, :H, :H])
                  if K >= 4 else np.zeros((k,) * 4))
            e = endpoint(mu, C, k3, k4)
            for kk in tot:
                tot[kk] += e[kk] / (T * (T - 1))
    return tot, flops


# ----------------------------------------------------------------- (a) recent-input mixture, joint with x
def est_rim(model, m, n, closure):
    Wi, Wh, Wo = model; T = Wo.shape[0]; D = H + T
    z, w = gh_nodes(n); nn = len(z)
    Wt = np.ones(1); MU = np.zeros((1, D)); C = np.zeros((1, D, D)); C[0, H:, H:] = np.eye(T)
    keys = [()]; flops = 0
    for t in range(T):
        B = len(Wt)
        MUn = np.repeat(MU, nn, 0); Cn = np.repeat(C, nn, 0)
        xi = np.tile(z, B)
        # set x_t to the node value (variance 0, no covariance)
        MUn[:, H + t] = xi; Cn[:, H + t, :] = 0; Cn[:, :, H + t] = 0
        mu_u = MUn[:, :H] @ Wh.T + xi[:, None] * Wi[None]
        Cu = np.einsum("ij,bjk,lk->bil", Wh, Cn[:, :H, :H], Wh)
        Cux = np.einsum("ij,bjk->bik", Wh, Cn[:, :H, H:])
        if not Cn.any():
            mm, cc, c = np.maximum(mu_u, 0), np.zeros_like(Cu), (mu_u > 0).astype(float)
        else:
            parts = [relu_layer_b(mu_u[i:i + 1024], Cu[i:i + 1024], closure) for i in range(0, B * nn, 1024)]
            mm, cc, c = (np.concatenate([p[j] for p in parts]) for j in range(3))
        flops += B * nn * cost_closure(H, D, closure)
        MUn[:, :H] = mm; Cn[:, :H, :H] = cc
        Cn[:, :H, H:] = c[:, :, None] * Cux; Cn[:, H:, :H] = Cn[:, :H, H:].transpose(0, 2, 1)
        Wn = (Wt[:, None] * w[None, :]).reshape(-1)
        kn = [k + (j,) for k in keys for j in range(nn)]
        if len(kn[0]) > m:
            kn = [k[1:] for k in kn]
            uniq = {}
            for i, k in enumerate(kn):
                uniq.setdefault(k, []).append(i)
            keys = list(uniq)
            W2 = np.empty(len(keys)); MU2 = np.empty((len(keys), D)); C2 = np.empty((len(keys), D, D))
            for a, k in enumerate(keys):
                ii = uniq[k]; ww = Wn[ii]; s = ww.sum()
                mu_ = (ww[:, None] * MUn[ii]).sum(0) / s; dm = MUn[ii] - mu_
                C2[a] = (ww[:, None, None] * (Cn[ii] + dm[:, :, None] * dm[:, None, :])).sum(0) / s
                W2[a] = s; MU2[a] = mu_
            Wt, MU, C = W2, MU2, C2
            flops += len(kn) * 3 * D * D
        else:
            keys, Wt, MU, C = kn, Wn, MUn, Cn
    A = np.zeros((2 * T, D)); A[:T, :H] = Wo; A[T:, H:] = np.eye(T)
    tot = 0.0
    det = not C.any()
    for b in range(len(Wt)):
        mu_b = A @ MU[b]
        if det:
            tot += Wt[b] * float(mu_b[:T].argmax() == np.argsort(mu_b[T:])[-2])
        else:
            tot += Wt[b] * gauss_correct(mu_b, A @ C[b] @ A.T, T, m=12)
    return tot, flops


def run_spec(spec, model):
    p = spec.split(":")
    if p[0] == "joint":
        return est_joint(model, p[1])
    if p[0] == "cell":
        return est_cell(model, p[1])
    if p[0] == "arccell":
        return est_arccell(model, int(p[1]))
    if p[0] == "rim":
        return est_rim(model, int(p[1]), int(p[2]), p[3])
    raise ValueError(spec)


def main():
    import json
    T = int(sys.argv[1]); specs = sys.argv[2].split(","); mnames = sys.argv[3].split(",")
    truths = json.load(open(os.path.join(ROOT, "results", "leaderboard_truths.json" if T == 10 else f"leaderboard_truths_H16_T{T}.json")))
    allm = None
    fpe = LB.fpe_flops(H, T)
    for mn in mnames:
        if mn == "base":
            name, model = LB._base(H, T)
        elif mn.startswith("seed"):
            name, model = mn, LB._zoo(H, T, int(mn[4:]))
        else:
            allm = allm or LB.models(H, T); name, model = mn, allm[mn]
        tr = truths[name]["acc"]
        for spec in specs:
            est, fl = run_spec(spec, model)
            if isinstance(est, dict):
                s = "  ".join(f"{k}={v:.5f} ({100*(v-tr):+.2f} pp)" for k, v in est.items())
            else:
                s = f"{est:.5f} ({100*(est-tr):+.2f} pp)"
            print(f"T={T} {name:10s} truth {tr:.5f}  {spec:20s} {s}   cost ~{fl/fpe:,.0f} FPE", flush=True)


if __name__ == "__main__":
    main()
