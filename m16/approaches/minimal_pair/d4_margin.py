"""D4b: median relative readout margin (target logit - best other, / max|logit|) by number of positive inputs k."""
import numpy as np
import core as C
for T in (4, 5):
  for s in range(5):
    m = C.zoo(16, T, s); x = np.random.default_rng(4).standard_normal((400_000, T))
    L = C.forward(m, x); tgt = np.argsort(x, 1)[:, -2]
    mar = L[np.arange(len(L)), tgt] - np.where(np.eye(T, dtype=bool)[tgt], -np.inf, L).max(1)
    rel = mar / np.abs(L).max(1); k = (x > 0).sum(1)
    print(f"T{T} s{s}: median rel margin by k=0..{T}: " + " ".join(f"{np.median(rel[k==j]):.3f}" for j in range(T + 1)))
