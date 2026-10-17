"""T79 check 1: fidelity of reductions of the low inputs given a top-K skeleton (DIAGNOSIS ONLY, random inputs).
Independent re-implementation of E31's diag_cond/diag_condk (analytic Jacobian instead of finite differences,
truncnorm sampler, Genz orthant probability instead of sampled Gaussians, different seeds/models).

Per skeleton S: q = P(correct|S) from M paired low draws; each reduction evaluated on the SAME draws.
  R0      low inputs at the conditional mean
  lin     exact linear map of the gate pattern at the conditional-mean point
  add     cut-HDMR additive surrogate (cross-check of E31)
  gauss   Gaussian law of the 9 margins with their TRUE conditional mean/cov (Genz orthant probability)
  slin    statistical linearisation: OLS of logits on the low inputs over the M draws (oracle, in-sample)
  slin_r  slin + independent Gaussian residual with the true residual covariance (oracle 'linear + noise' law)
  lin_r   lin + independent Gaussian residual with the true residual covariance of lin (oracle)
Also: leverage = Var_S(q)/p(1-p) and its inverse, the conditional-MC gain ceiling.
usage: python review/stoch_margin/fidelity.py MODEL K [NS] [M] [seed]
"""
import json
import sys
import time

import numpy as np
from scipy.stats import multivariate_normal as mvn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rv  # noqa: E402

name, K = sys.argv[1], int(sys.argv[2])
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
M = int(sys.argv[4]) if len(sys.argv) > 4 else 256
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 101
model = rv.dev_model(name)
rng = np.random.default_rng(seed)
nl = rv.T - K
CH = 100
cols = {k: [] for k in ["q", "R0", "lin", "add", "gauss", "slin", "slin_r", "lin_r", "v2", "v3"]}
t0 = time.time()


def frac_ok(L, lab):
    return (L.argmax(-1) == lab[:, None]).mean(1)


def orthant(Lmat, lab):
    """P(all margins > 0) for Gaussian logits-law given by sample moments of Lmat (M,10)."""
    others = [k for k in range(10) if k != lab]
    mar = Lmat[:, [lab]] - Lmat[:, others]
    mu, C = mar.mean(0), np.cov(mar.T) + 1e-8 * np.eye(9)
    return float(mvn.cdf(np.zeros(9), mean=-mu, cov=C, abseps=1e-6, releps=1e-5))


for c in range(NS // CH):
    pos, val = rv.skeletons(CH, K, rng)
    free = rv.free_positions(pos)
    low = rv.low_draws(val[:, -1], M, nl, rng)
    x = rv.assemble(pos, val, free, low)
    lab = pos[:, 1]
    L = rv.run(model, x.reshape(-1, rv.T))[0].reshape(CH, M, 10)
    mu = rv.cond_mean(val[:, -1])
    xb = rv.assemble(pos, val, free, np.repeat(mu[:, None, None], nl, 2))[:, 0]
    L0, pre0 = rv.run(model, xb, keep_pre=True)
    J = rv.jacobian(model, pre0)                                       # (CH,10,T)
    Jl = np.take_along_axis(J, free[:, None, :].repeat(10, 1), 2)      # (CH,10,nl)
    dlow = low - mu[:, None, None]
    Llin = L0[:, None] + np.einsum("nkj,nmj->nmk", Jl, dlow)
    Ladd = np.repeat(L0[:, None], M, 1)
    for j in range(nl):
        xj = np.repeat(xb[:, None], M, 1)
        xj[np.arange(CH), :, free[:, j]] = low[:, :, j]
        Ladd += rv.run(model, xj.reshape(-1, rv.T))[0].reshape(CH, M, 10) - L0[:, None]
    # statistical linearisation (oracle): per skeleton OLS of L on [1, low]
    A = np.concatenate([np.ones((CH, M, 1)), low], 2)
    coef = np.linalg.solve(np.einsum("nmi,nmj->nij", A, A) + 1e-10 * np.eye(nl + 1), np.einsum("nmi,nmk->nik", A, L))
    Lsl = np.einsum("nmi,nik->nmk", A, coef)
    res = L - Lsl
    resl = L - Llin
    noise = rng.standard_normal((CH, M, 10))
    Lsl_r, Llin_r = np.empty_like(L), np.empty_like(L)
    for i in range(CH):
        for src, dst, R in ((Lsl, Lsl_r, res), (Llin, Llin_r, resl)):
            Rc = R[i] - R[i].mean(0)
            C = Rc.T @ Rc / (M - 1)
            w, V = np.linalg.eigh(C)
            dst[i] = src[i] + R[i].mean(0) + noise[i] @ (V * np.sqrt(np.clip(w, 0, None))).T
        cols["gauss"].append(orthant(L[i], lab[i]))
    cols["q"] += list(frac_ok(L, lab))
    cols["R0"] += list((L0.argmax(1) == lab).astype(float))
    cols["lin"] += list(frac_ok(Llin, lab))
    cols["add"] += list(frac_ok(Ladd, lab))
    cols["slin"] += list(frac_ok(Lsl, lab))
    cols["slin_r"] += list(frac_ok(Lsl_r, lab))
    cols["lin_r"] += list(frac_ok(Llin_r, lab))
    cols["v2"] += list(val[:, 1]); cols["v3"] += list(val[:, 2])
    if c % 10 == 0:
        print(c, round(time.time() - t0), flush=True)

R = {k: np.array(v) for k, v in cols.items()}
q = R["q"]
n = len(q)
p = q.mean()
var_q = q.var(ddof=1) - (q * (1 - q)).mean() / (M - 1)       # unbiased Var_S(E[f|S])
out = dict(model=name, K=K, NS=n, M=M, seed=seed, acc=float(p), leverage=float(var_q / (p * (1 - p))),
           cmc_gain_ceiling=float(p * (1 - p) / var_q), secs=round(time.time() - t0))
for k in ["R0", "lin", "add", "gauss", "slin", "slin_r", "lin_r"]:
    out[k] = rv.summarize(q, R[k], n)
print(json.dumps(out, indent=1))
np.savez(f"{rv.OUT}/fid_{name}_K{K}_s{seed}.npz", **R)
json.dump(out, open(f"{__file__.rsplit('/', 1)[0]}/fid_{name}_K{K}.json", "w"), indent=1)
