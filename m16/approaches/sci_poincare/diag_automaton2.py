"""Markov-automaton test with sign symbol: H(G_t | G_{t-1}, rank symbol, sign x_t), and per-neuron uncertainty."""
import sys; sys.path.insert(0, "/home/user/clean-neurons/m16")
import numpy as np
from collections import Counter, defaultdict
from arc_eval import Wi0, Wh0, Wo0
rng = np.random.default_rng(5); N = 400_000
x = rng.standard_normal((N, 10)); h = np.zeros((N, 16)); B = []
for t in range(10):
    pre = h @ Wh0.T + x[:, t:t+1] * Wi0[None]; B.append(pre > 0); h = np.maximum(pre, 0)
B = np.stack(B, 1); G = B @ (1 << np.arange(16))
rel = np.array([(x[:, :t+1] > x[:, t:t+1]).sum(1) + 1 for t in range(10)]).T
r = np.minimum(rel, 4) * 2 + (x > 0)
tot_unc = np.zeros(16)
for t in range(1, 10):
    key = G[:, t-1] * 16 + r[:, t]
    # per-neuron: fraction of samples where neuron's gate differs from the modal value within its class
    _, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv); on = np.stack([np.bincount(inv, B[:, t, j]) for j in range(16)], 1)
    err = np.minimum(on, cnt[:, None] - on).sum(0) / N
    tot_unc += err
    d = defaultdict(Counter)
    for k, v in zip(key, G[:, t]): d[k][v] += 1
    modal = sum(max(c.values()) for c in d.values()) / N
    print(f"t={t}: modal next-pattern given (G-,rank,sign) = {modal:.3f}; per-neuron unpredictable frac:", np.round(err, 3))
print("summed over steps:", np.round(tot_unc, 3))
