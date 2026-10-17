"""Unit check: exact fibre fraction vs 2e6-point dense grid on the same circle (random circles, 4 models)."""
import sys, os; import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "understand_m43"))
from zoo44 import zoo; import hopf
rng = np.random.default_rng(3); worst = 0
for H, T, s in [(4, 4, 0), (4, 4, 3), (4, 3, 0), (4, 4, 1)]:
    m = zoo(H, T, s); Wi, Wh, Wo = m
    for _ in range(5):
        a, b = np.linalg.qr(rng.standard_normal((T, 2)))[0].T
        for w in ("uniform", "abs_sin"):
            f, n = hopf.fibre(m, a, b, w)
            ph = (np.arange(2_000_000) + 0.5) * 2 * np.pi / 2_000_000; x = np.outer(np.cos(ph), a) + np.outer(np.sin(ph), b)
            h = np.zeros((len(x), H))
            for t in range(T):
                h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
            ok = (h @ Wo.T).argmax(1) == np.argsort(-x, 1)[:, 1]; wt = np.ones(len(ph)) if w == "uniform" else np.abs(np.sin(ph))
            worst = max(worst, abs(f - (ok * wt).sum() / wt.sum()))
print(f"max |exact fibre - dense grid| over 40 circles: {worst:.2e} (grid resolution ~1e-6)")
