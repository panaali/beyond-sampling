"""Sanity check of CondLaw against brute-force rejection from N(0, I_10) (model-free, no scoring)."""
import numpy as np
import core
for t0 in (1.25, 2.75):
    L = core.CondLaw(t0); rng = np.random.default_rng(1)
    acc = []; n = 0
    for _ in range(40):
        z = rng.standard_normal((1_000_000, 10)); s = -np.sort(-z, 1); m = s[:, 1] - s[:, 2] > t0; acc.append(s[m]); n += len(z)
    a = np.concatenate(acc); x, pos = L.draw(np.random.default_rng(2), 200_000); s = -np.sort(-x, 1)
    assert np.all(x[np.arange(len(x)), pos[:, 0]] == s[:, 0]) and np.all(x[np.arange(len(x)), pos[:, 2]] == s[:, 2])
    print(t0, "p_gap grid %.6g  brute %.6g +- %.2g" % (L.p_gap, len(a) / n, np.sqrt(len(a)) / n))
    print("  mean ranks law ", np.round(s[:, :5].mean(0), 3), "\n  mean ranks brute", np.round(a[:, :5].mean(0), 3), "(n=%d)" % len(a))
