"""Region counts for the scaling question: (a) distinct gate patterns among n Gaussian inputs (a lower bound on the
number of linear regions of R^T; regions of tiny mass are missed), (b) exact gate cells on the T = 4 boundary
faces used by fibre4 (2-D arrangement), (c) arc segments for T = 3. usage: python count_regions.py H T [n]"""
import sys

import numpy as np

from zoo import zoo_model


def patterns(model, n, rng, chunk=200_000):
    Wi, Wh, Wo = model
    T = Wo.shape[0]; H = len(Wi)
    seen = set(); done = 0
    w = (1 << np.arange(H * T, dtype=np.uint64)).astype(np.uint64) if H * T <= 64 else None
    while done < n:
        m = min(chunk, n - done); x = rng.standard_normal((m, T))
        h = np.zeros((m, H)); bits = []
        for t in range(T):
            z = h @ Wh.T + x[:, t:t + 1] * Wi[None]; bits.append(z > 0); h = np.maximum(z, 0)
        B = np.concatenate(bits, 1)
        if w is not None:
            seen.update(np.unique((B.astype(np.uint64) * w).sum(1)).tolist())
        else:
            seen.update(map(bytes, np.packbits(B, 1)))
        done += m
    return len(seen)


if __name__ == "__main__":
    H, T = int(sys.argv[1]), int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 2_000_000
    for s in range(5):
        m = zoo_model(H, T, s)
        r = [patterns(m, k, np.random.default_rng(1)) for k in (n // 4, n)]
        extra = ""
        if T == 4:
            import fibre4
            from fiber3 import Ops
            nc = sum(len(fibre4.cells_on_face(m, poly, fam["B"], Ops())) for fam in fibre4.FAMS4 for poly in fam["polys"])
            extra = f" face cells {nc}"
        print(f"H={H} T={T} seed {s}: distinct patterns in {n // 4} / {n} samples: {r[0]} / {r[1]}{extra}", flush=True)
