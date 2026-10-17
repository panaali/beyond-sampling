"""T87 diagnostic (MC, 2e6 inputs; description only): where the error mass of zoo (4,3) vs (4,4) lies -- angular
distance from the nearest class-changing true boundary of the input's ordering chamber, and the number of hidden
neurons on at the last step. usage: python diag87b.py"""
import numpy as np
from zoo44 import zoo


def describe(model, n=2_000_000, seed=0):
    Wi, Wh, Wo = model; T = Wo.shape[0]
    x = np.random.default_rng(seed).standard_normal((n, T)); x /= np.linalg.norm(x, axis=1, keepdims=True)
    h = np.zeros((n, len(Wi)))
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0)
    wrong = (h @ Wo.T).argmax(1) != np.argsort(x, 1)[:, -2]
    xs = np.sort(x, 1)[:, ::-1]                      # descending
    d = np.minimum(xs[:, 0] - xs[:, 1], xs[:, 1] - xs[:, 2]) / np.sqrt(2)   # distance to x_a = x_b or x_b = x_c
    ang = np.degrees(np.arcsin(np.clip(d, 0, 1)))
    act = (h > 0).sum(1)
    return wrong, ang, act


for T in (3, 4):
    for s in range(5):
        w, a, act = describe(zoo(4, T, s))
        q = np.percentile(a[w], [50, 90]); qa = np.percentile(a, [50, 90])
        print(f"(4,{T}) seed {s}: error {w.mean():.3f}; angle from nearest boundary: errors median {q[0]:.1f} deg (90%: {q[1]:.1f}), "
              f"all inputs median {qa[0]:.1f}; error share within 5 deg {np.mean(a[w] < 5):.0%}; active neurons on errors "
              + ", ".join(f"{k}:{np.mean(act[w] == k):.0%}" for k in range(5)), flush=True)
