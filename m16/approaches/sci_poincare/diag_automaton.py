"""Diagnostic: is the gate sequence produced by a small finite automaton driven by the record symbols?
Symbol r_t = min(relative rank of x_t among x_0..x_t, K+1) (independent uniform by Renyi's theorem).
Reports per step: plug-in conditional entropy H(G_t | history), H(G_t | G_{t-1}, r_t),
and predictability (mass of the modal G_t within each conditioning class)."""
import sys; sys.path.insert(0, "/home/user/clean-neurons/m16")
import numpy as np
from collections import Counter, defaultdict
from arc_eval import Wi0, Wh0, Wo0
rng = np.random.default_rng(4); N = 400_000
x = rng.standard_normal((N, 10)); h = np.zeros((N, 16)); G = []
for t in range(10):
    pre = h @ Wh0.T + x[:, t:t+1] * Wi0[None]; G.append((pre > 0) @ (1 << np.arange(16))); h = np.maximum(pre, 0)
G = np.stack(G, 1)
rel = np.array([(x[:, :t+1] > x[:, t:t+1]).sum(1) + 1 for t in range(10)]).T  # 1 = new max
def H_cond(keys, vals):
    d = defaultdict(Counter)
    for k, v in zip(keys, vals): d[k][v] += 1
    H = 0; modal = 0
    for c in d.values():
        n = sum(c.values()); p = np.array(list(c.values())) / n
        H += n * -(p * np.log2(p)).sum(); modal += max(c.values())
    return H / len(vals), modal / len(vals), len(d)
for K in (2, 3):
    r = np.minimum(rel, K + 1)
    print(f"K={K}")
    for t in range(10):
        hist = [tuple(row) for row in r[:, :t+1]]
        Hh, mh, nh = H_cond(hist, G[:, t])
        Hm, mm, nm = H_cond(list(zip(G[:, t-1] if t else np.zeros(N, int), r[:, t])), G[:, t])
        Hu, mu, _ = H_cond([0]*N, G[:, t])
        print(f" t={t}: H(G)={Hu:.2f}  H(G|hist)={Hh:.2f} modal={mh:.3f} (#hist {nh})   H(G|G-,r)={Hm:.2f} modal={mm:.3f}")
