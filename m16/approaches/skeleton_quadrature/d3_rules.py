"""D3: skeleton rules vs Monte Carlo on the skeleton integral p = (1/720) sum_t int q(t,U) dU (DIAGNOSIS of the rule;
q at every node from M iid inner draws, an unbiased stand-in for an exact q). Each rule uses N = 720 k nodes and is
randomised R times (scramble/shift/assignment), so its MSE = variance across randomisations minus the known inner
noise sum w^2 q(1-q)/(M-1); for rules with a deterministic part (Gauss nodes) the MSE uses the truth instead.
Rules:
  mc        iid triple, iid U                                   (the baseline: Var_S(q)/N)
  strat     every triple k times, iid U
  sob4      scrambled Sobol in 4-D, coordinate 0 -> triple = floor(720 u0)
  shared3   every triple k times; the N U-points are ONE scrambled 3-D Sobol set, randomly assigned to triples
            (the triple-averaged function is integrated by an N-point QMC set: structure shared across triples)
  pertri3   every triple k times; each triple gets its OWN scrambled k-point 3-D Sobol set
  tieGL     every triple; U3 by a k3-point Gauss-Legendre rule (tie direction, deterministic), (U1,U2) by a
            per-triple scrambled 2-D Sobol set with k/k3 points
usage: python m16/approaches/skeleton_quadrature/d3_rules.py MODEL [R] [M] [k list] [kinds] [tag]
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rv  # noqa: E402
from d1_shape_lib import vals_from_U  # noqa: E402

name = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 8
M = int(sys.argv[3]) if len(sys.argv) > 3 else 256
KS = [int(k) for k in sys.argv[4].split(",")] if len(sys.argv) > 4 else [2, 4, 8]
model = rv.dev_model(name)
rng = np.random.default_rng(5005)
TR = np.array(list(itertools.permutations(range(10), 3)))
D = __file__.rsplit("/", 1)[0]
truth = json.load(open(f"{D}/truth_{name}.json"))["p"] if os.path.exists(f"{D}/truth_{name}.json") else None


def qnodes(tri, U):
    pos = TR[tri]; val = vals_from_U(np.clip(U, 1e-12, 1 - 1e-12)); free = rv.free_positions(pos)
    q = np.empty(len(tri))
    for a in range(0, len(tri), 64):
        b = slice(a, a + 64)
        low = rv.low_draws(val[b, 2], M, 7, rng)
        x = rv.assemble(pos[b], val[b], free[b], low)
        q[b] = (rv.run(model, x.reshape(-1, rv.T))[0].reshape(-1, M, 10).argmax(2) == pos[b, 1:2]).mean(1)
    return q


def sob(d, n):
    return qmc.Sobol(d, scramble=True, seed=rng).random(n)


def rule(kind, k):
    N = 720 * k
    if kind == "mc":
        tri, U, w = rng.integers(0, 720, N), rng.uniform(size=(N, 3)), np.full(N, 1 / N)
    elif kind == "strat":
        tri, U, w = np.repeat(np.arange(720), k), rng.uniform(size=(N, 3)), np.full(N, 1 / N)
    elif kind == "sob4":
        P = sob(4, N); tri = np.minimum((P[:, 0] * 720).astype(int), 719); U = P[:, 1:]; w = np.full(N, 1 / N)
    elif kind == "shared3":
        tri = np.repeat(np.arange(720), k); U = sob(3, N)[rng.permutation(N)]; w = np.full(N, 1 / N)
    elif kind == "pertri3":
        tri = np.repeat(np.arange(720), k); U = np.concatenate([sob(3, k) for _ in range(720)]); w = np.full(N, 1 / N)
    elif kind.startswith("tieGL"):
        k3 = int(kind[5:]); k12 = k // k3
        xg, wg = np.polynomial.legendre.leggauss(k3)
        tri, U, w = [], [], []
        for t in range(720):
            P = sob(2, k12)
            for i in range(k12):
                tri += [t] * k3
                U += [[P[i, 0], P[i, 1], (x + 1) / 2] for x in xg]
                w += list(wg / 2 / k12 / 720)
        tri, U, w = np.array(tri), np.array(U), np.array(w)
    q = qnodes(tri, U)
    return float(w @ q), float(np.sum(w ** 2 * q * (1 - q)) / (M - 1)), float(np.var(q))


out = dict(model=name, R=R, M=M, truth=truth, rules={})
kinds = sys.argv[5].split(",") if len(sys.argv) > 5 else ["mc", "strat", "sob4", "shared3", "pertri3", "tieGL2", "tieGL4"]
TAG = sys.argv[6] if len(sys.argv) > 6 else ""
for k in KS:
    for kind in kinds:
        if kind.startswith("tieGL") and k % int(kind[5:]):
            continue
        est = [rule(kind, k) for _ in range(R)]
        e = np.array([a for a, _, _ in est]); nv = np.mean([b for _, b, _ in est])
        var_r = max(e.var(ddof=1) - nv, 0.0)
        mse_t = float(np.mean((e - truth) ** 2) - nv) if truth is not None else None
        out["rules"][f"{kind}_k{k}"] = dict(N=720 * k, mean=float(e.mean()), var_rand=var_r, inner_noise=nv,
                                            mse_vs_truth=mse_t, var_q=float(np.mean([c for _, _, c in est])))
        print(kind, k, out["rules"][f"{kind}_k{k}"], flush=True)
        json.dump(out, open(f"{D}/d3{TAG}_{name}.json", "w"), indent=1)
