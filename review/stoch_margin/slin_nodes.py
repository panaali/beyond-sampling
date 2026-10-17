"""T79 variant C: statistical linearisation + Gaussian residual ('slin_r'), the one reduction found faithful at
K = 3 as an ORACLE in fidelity.py, now with its moments computed from a FIXED deterministic node set of n points
per skeleton (no oracle; DIAGNOSIS ONLY for the truth). Per skeleton S (top-K positions + values):
  nodes  : the first n points of an unscrambled Sobol sequence in [0,1]^(10-K), shifted by 1/(2n) and mapped to the
           conditional law N(0,1) | < v_K by the inverse CDF (model-independent, deterministic);
  fit    : OLS of the 10 logits on [1, x_low] over the n nodes -> a, B; residual covariance S_r (n-1-nl dof);
  law    : L = a + B x_low + N(0, S_r), x_low iid N(0,1)|<v_K (model-free; evaluated here with the M paired truth
           draws as a proxy for its exact law, plus independent Gaussian noise).
Reports the bias against q(S) from M paired truth draws, for n in NLIST. Cost of q(S): n FPE per skeleton.
usage: python review/stoch_margin/slin_nodes.py MODEL K [NS] [M]
"""
import json
import sys
import time

import numpy as np
from scipy.stats import norm, qmc

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rv  # noqa: E402

name, K = sys.argv[1], int(sys.argv[2])
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
M = int(sys.argv[4]) if len(sys.argv) > 4 else 256
NLIST = (16, 32, 64, 128)
model = rv.dev_model(name)
rng = np.random.default_rng(505)
nl = rv.T - K
U = {n: (qmc.Sobol(nl, scramble=False).random(n) + 0.5 / n) % 1.0 for n in NLIST}
CH = 100
res = {k: [] for k in ["q"] + [f"n{n}" for n in NLIST] + ["oracle"]}
t0 = time.time()


def law_q(L_low, coef, Sr, low, lab, noise):
    """fraction of paired draws where a + B x + residual has argmax lab."""
    A = np.concatenate([np.ones((low.shape[0], 1)), low], 1)
    w, V = np.linalg.eigh(Sr)
    Ls = A @ coef + noise @ (V * np.sqrt(np.clip(w, 0, None))).T
    return (Ls.argmax(1) == lab).mean()


def fit(X, Lx):
    A = np.concatenate([np.ones((len(X), 1)), X], 1)
    coef, *_ = np.linalg.lstsq(A, Lx, rcond=None)
    R = Lx - A @ coef
    return coef, R.T @ R / max(len(X) - 1 - nl, 1)


for c in range(NS // CH):
    pos, val = rv.skeletons(CH, K, rng)
    vK = val[:, -1]
    free = rv.free_positions(pos)
    low = rv.low_draws(vK, M, nl, rng)
    lab = pos[:, 1]
    L = rv.run(model, rv.assemble(pos, val, free, low).reshape(-1, rv.T))[0].reshape(CH, M, 10)
    res["q"] += list((L.argmax(2) == lab[:, None]).mean(1))
    noise = rng.standard_normal((CH, M, 10))
    Ln = {}
    for n in NLIST:
        Xn = norm.ppf(U[n][None] * norm.cdf(vK)[:, None, None])         # (CH,n,nl)
        Ln[n] = (Xn, rv.run(model, rv.assemble(pos, val, free, Xn).reshape(-1, rv.T))[0].reshape(CH, n, 10))
    for i in range(CH):
        for n in NLIST:
            coef, Sr = fit(Ln[n][0][i], Ln[n][1][i])
            res[f"n{n}"].append(law_q(None, coef, Sr, low[i], lab[i], noise[i]))
        coef, Sr = fit(low[i], L[i])
        res["oracle"].append(law_q(None, coef, Sr, low[i], lab[i], noise[i]))
    if c % 5 == 0:
        print(c, round(time.time() - t0), flush=True)
q = np.array(res["q"])
out = dict(model=name, K=K, NS=len(q), M=M, acc=float(q.mean()))
for k in res:
    if k != "q":
        out[k] = rv.summarize(q, np.array(res[k]), len(q))
out["secs"] = round(time.time() - t0)
print(json.dumps(out, indent=1))
json.dump(out, open(f"{__file__.rsplit('/', 1)[0]}/slinn_{name}_K{K}.json", "w"), indent=1)
