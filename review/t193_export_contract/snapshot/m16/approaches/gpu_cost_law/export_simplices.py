"""T192 -> Codex lane 34: deterministic sample of the simplicial cones integrated by exact enumeration at (8,5).

Halves-start exact mode (as E48), halves processed in index order, simplices in estT's emission order; every
STRIDE-th simplex is kept. Saved per simplex: rays R (rows, unit), |det R|, mass by the estT rule (estG batched,
= estT._orth5 to ~1e-13) and a reference mass (sliver bisection to pairwise cos >= 0.6, then thin rule 10^4 nodes /
Plackett 256 log nodes). usage: python export_simplices.py seed [stride] -> results/simplices/H8T5_zoo{seed}.npz"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
import estP  # noqa: E402
import estG  # noqa: E402
import estT_base as B  # noqa: E402

seed = int(sys.argv[1]); stride = int(sys.argv[2]) if len(sys.argv) > 2 else 430
model = estP.load_model(8, 5, str(seed)); estP._init(model, "exact", 0.1)
keep = []; n = 0
for i in range(len(B._HALVES[5])):
    R = estP._half(i)[1].get(5)
    if R is None:
        continue
    for r in R:
        if n % stride == 0:
            keep.append(r)
        n += 1
R = np.array(keep)
m_estT = estG.masses_each(R, 5)
B.set_rules5(10, 256); estG._CACHE.clear(); estG.SLIVER[0] = 0.6
m_ref = np.array([estG.masses(r[None], 5)[0] for r in R])
os.makedirs(os.path.join(HERE, "results", "simplices"), exist_ok=True)
np.savez_compressed(os.path.join(HERE, "results", "simplices", f"H8T5_zoo{seed}.npz"), rays=R,
                    absdet=np.abs(np.linalg.det(R)), mass_estT=m_estT, mass_ref=m_ref, stride=stride, total=n)
print(seed, "kept", len(R), "of", n, "sum estT %.12e ref %.12e" % (m_estT.sum(), m_ref.sum()),
      "max rel diff %.2e" % np.max(np.abs(m_estT - m_ref) / m_ref))
