"""Large orig dataset for learning curves: python build_big.py -> data/orig_big.npz (300k lines, seed 77)."""
import numpy as np, wtool as W
M = W.get_model("orig"); parts = []
for s in range(5):
    R = W.build(M, 60000, seed=77 + s)
    parts.append({k: R[k] for k in ["y", "a", "b", "p1", "v", "o1", "delta", "mass", "madj", "delta_lin", "s", "m"]})
    print(s, flush=True)
np.savez_compressed("data/orig_big.npz", **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
