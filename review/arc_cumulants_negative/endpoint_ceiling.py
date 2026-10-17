"""T80 variants (c), (d): ceiling of ANY cumulant propagation that ends in a cumulant endpoint.

Feed the endpoint the *exact* (large-MC) cumulants of the final quantities and see how far its accuracy is from the
truth. A propagation can at best deliver these cumulants, so if the ceiling misses by more than sampling's error at
the relevant budgets, that combination of conditioning + endpoint cannot compete (barring lucky cancellation).

Conditionings ("cells"), the target being P(argmax l = p2 | cell):
  p1p2   : positions of the largest and 2nd-largest input (T(T-1) cells; what E35 'cell' and E13 used)
  perm   : full ranking of x (T! cells; T <= 5)
  p1p2s  : p1p2 x sign pattern of x_0 (gate structure of step 1: h_1 = relu(Wi) x0+ + relu(-Wi) x0- exactly)
Endpoints on the T-1 margins m_i = l_p2 - l_i (i != p2):
  gauss  : Gaussian orthant with exact mean/cov
  edge3  : + kappa_3 Edgeworth term
  edge4  : + kappa_4 and kappa_3^2 terms (full 2nd-order Edgeworth)
This orthant probability under the Edgeworth density is the same number as ARC's argmax endpoint
q_p2 = d/da_p2 E[max_j(l_j + a_j)] evaluated with the Edgeworth density of l (T40), since the margins are linear in l.
Also 'joint': one Gaussian for (l, x) jointly, unconditional (what a K=2 propagation of the joint state gives).

The orthant integrals are done by plain Gaussian draws (2^22) of the explicit endpoint density, not model calls.
Usage: python endpoint_ceiling.py T cellkind1,cellkind2 n_per_cell [model_name]"""
import itertools
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, os.path.join(ROOT, "m16")); sys.path.insert(0, HERE)
import leaderboard as LB  # noqa: E402

T = int(sys.argv[1]); KINDS = sys.argv[2].split(","); NPC = int(sys.argv[3])
MNAME = sys.argv[4] if len(sys.argv) > 4 else None
H = 16


def get_model():
    if MNAME is None:
        return LB._base(H, T)
    if MNAME.startswith("seed"):
        return MNAME, LB._zoo(H, T, int(MNAME[4:]))
    return MNAME, LB.models(H, T)[MNAME]


def truth_of(name):
    import json
    p = os.path.join(ROOT, "results", "leaderboard_truths.json" if T == 10 else f"leaderboard_truths_H16_T{T}.json")
    d = json.load(open(p))
    return d[name]["acc"] if name in d else float("nan")


from edgeworth import cumulants, endpoint, NG  # noqa: E402


def forward(model, x):
    return LB.forward(model, x)


def cells(kind):
    if kind == "p1p2":
        return [(p1, p2) for p1 in range(T) for p2 in range(T) if p1 != p2]
    if kind == "perm":
        return list(itertools.permutations(range(T)))
    if kind == "p1p2s":
        return [(p1, p2, s) for p1 in range(T) for p2 in range(T) if p1 != p2 for s in (0, 1)]
    raise ValueError(kind)


def sample_cell(kind, cell, n, r):
    """Exact draws of x | cell (every draw used) and the cell's probability."""
    x = -np.sort(-r.standard_normal((n, T)), axis=1)  # descending
    if kind == "perm":
        y = np.empty_like(x); y[:, list(cell)] = x       # cell[r] = position of rank r
        return y, 1 / math.factorial(T)
    p1, p2 = cell[0], cell[1]
    rest = x[:, 2:].copy(); r.permuted(rest, axis=1, out=rest)
    others = [i for i in range(T) if i not in (p1, p2)]
    y = np.empty_like(x); y[:, p1] = x[:, 0]; y[:, p2] = x[:, 1]; y[:, others] = rest
    pc = 1 / (T * (T - 1))
    if kind == "p1p2s":
        s = cell[2]
        keep = (y[:, 0] > 0) == bool(s)
        frac = keep.mean()
        return y[keep], pc * frac           # P(cell) estimated (model-free); reported separately below
    return y, pc


def main():
    name, model = get_model()
    tr = truth_of(name)
    r = np.random.default_rng(2024)
    print(f"model {name} (H={H}, T={T}) truth {tr:.5f}; {NPC} exact draws per cell; orthants by {NG} Gaussian draws")
    for kind in KINDS:
        acc = {"mc": 0.0, "gauss": 0.0, "edge3": 0.0, "edge4": 0.0}; tot_p = 0.0
        for cell in cells(kind):
            x, pc = sample_cell(kind, cell, NPC, r)
            p2 = cell[1] if kind != "perm" else cell[1]
            l = forward(model, x)
            m = l[:, [p2]] - l[:, [i for i in range(T) if i != p2]]
            mu, C, k3, k4 = cumulants(m)
            e = endpoint(mu, C, k3, k4)
            acc["mc"] += pc * (m > 0).all(1).mean()
            for kk in e:
                acc[kk] += pc * e[kk]
            tot_p += pc
        print(f"  cells={kind:6s} ({len(cells(kind))} cells, sum P={tot_p:.4f}): "
              + "  ".join(f"{k}={v:.5f} ({100*(v-tr):+.2f} pp)" for k, v in acc.items()), flush=True)
    if "joint" in sys.argv[5:] or (len(sys.argv) > 5 and sys.argv[5] == "joint"):
        n = NPC * 4
        x = r.standard_normal((n, T)); l = forward(model, x)
        z = np.concatenate([l, x], 1); mu = z.mean(0); C = np.cov(z.T)
        g = np.random.default_rng(5).multivariate_normal(mu, C, size=NG, method="eigh")
        pj = (g[:, :T].argmax(1) == np.argsort(g[:, T:], 1)[:, -2]).mean()
        print(f"  joint Gaussian (l, x), exact mean/cov from {n} draws: {pj:.5f} ({100*(pj-tr):+.2f} pp)")


if __name__ == "__main__":
    main()
