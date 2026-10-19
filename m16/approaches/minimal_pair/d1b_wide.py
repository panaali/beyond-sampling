"""D1b: where are the WIDE errors (relevant gap > thr) at T=5? positions + values of all inputs."""
import sys
import numpy as np
import core as C
thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
for T in (4, 5):
  for s in range(5):
    m = C.zoo(16, T, s); rng = np.random.default_rng(900 + s + 10*T)
    W = []; tot = 0
    for _ in range(16):
        x = rng.standard_normal((250_000, T))
        pred = C.forward(m, x).argmax(1)
        order = np.argsort(-x, 1); rk = np.argsort(order, 1)[np.arange(len(x)), pred] + 1
        xs = -np.sort(-x, 1)
        g = np.where(rk == 1, xs[:, 0] - xs[:, 1], np.where(rk == 3, xs[:, 1] - xs[:, 2], 9))
        bad = rk != 2; tot += bad.sum()
        w = bad & (g > thr)
        W.append(np.c_[rk[w], order[w, :3], xs[w, :3], g[w], x[w]])
    W = np.concatenate(W); n = 4_000_000
    print(f"T{T} s{s}: err {tot/n:.5f} wide(>{thr}) {len(W)/n:.6f} = {len(W)/tot:.3f} of errors")
    if len(W) < 5: continue
    # pattern: predicted rank, position of r1, r2, r3
    from collections import Counter
    c = Counter(f"pr{int(r[0])} pos{int(r[1])}{int(r[2])}{int(r[3])}" for r in W)
    print("   ", c.most_common(8))
    print("    median top3 values", np.median(W[:, 4:7], 0).round(2), " last-pos value med", np.median(W[:, 8+T-1]).round(2),
          " #positives med", np.median((W[:, 8:8+T] > 0).sum(1)))
