"""Accuracy of M_16,10 under input distributions and weight-noise levels (lit/web_arc README section 4).

Run from the repo root: source .venv/bin/activate && python lit/web_arc/noise_check.py   (~10 CPU-seconds)
"""
import sys

import numpy as np

sys.path.insert(0, "m16")
from arc_eval import forward, noisy_models  # noqa: E402
from common import Wi, Wh, Wo  # noqa: E402


def acc(model, x):
    return (forward(model, x).argmax(1) == np.argsort(x, 1)[:, -2]).mean()


base = (Wi, Wh, Wo)
rng = np.random.default_rng(7)
print("gauss", acc(base, rng.standard_normal((400_000, 10))))
print("unif ", acc(base, rng.uniform(-1, 1, (400_000, 10))))
print("RMS per matrix", [round(float(np.sqrt((W ** 2).mean())), 3) for W in base])

rng = np.random.default_rng(3)
for s in (0.01, 0.05):                       # absolute noise, as ARC used for M_4,3 (s.d. 0.05)
    a = np.array([acc(tuple(W + s * rng.standard_normal(W.shape) for W in base), rng.standard_normal((100_000, 10)))
                  for _ in range(16)])
    print(f"abs sigma {s}: mean {a.mean():.4f} sd {a.std():.4f}")

rng = np.random.default_rng(5)
for s in (0.0025, 0.005, 0.01, 0.02):        # our relative noise (sigma x matrix RMS)
    a = np.array([acc(m, rng.standard_normal((100_000, 10))) for m in noisy_models(12, s, seed=1)])
    print(f"rel sigma {s}: mean {a.mean():.4f} sd {a.std():.4f}")
